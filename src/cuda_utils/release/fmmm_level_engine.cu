#include "fmmm_workspace.h"
#include "fmmm_session.h"
#include "fmmm_level_engine.h"

#include "fmmm_advanced_uncoarsen.h"
#include "fmmm_hierarchy_contract.h"
#include "fmmm_nmm_packing.h"

#include "../oracle/continuous_crossings.h"
#include "../oracle/device_memory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lcn::oracle {
namespace {

constexpr int kPlacementOk = 0;
constexpr int kPlacementUnsupported = 1;
constexpr int kPlacementMaxNodes = 64;
constexpr int kForceMaxEdges = 256;
constexpr int kHierarchyNoMoonNeighbour = 2;

// This is the MSVC std::mt19937 + uniform_int_distribution path used by the
// pinned OGDF runner.  It stays in the serial CUDA reference so that no host
// PRNG call can change the draw order.
struct OgdfMt19937 {
    static constexpr int kWords = 624;
    std::uint32_t state[kWords];
    int index = kWords;
    std::int64_t draws_consumed = 0;

    __device__ void seed(std::uint32_t value) {
        state[0] = value;
        for (int i = 1; i < kWords; ++i) {
            state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30U)) + static_cast<std::uint32_t>(i);
        }
        index = kWords;
        draws_consumed = 0;
    }
    __device__ void twist() {
        for (int i = 0; i < kWords; ++i) {
            const std::uint32_t bits = (state[i] & 0x80000000U) | (state[(i + 1) % kWords] & 0x7fffffffU);
            state[i] = state[(i + 397) % kWords] ^ (bits >> 1U) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
        }
        index = 0;
    }
    __device__ std::uint32_t next() {
        if (index == kWords) twist();
        std::uint32_t value = state[index++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        ++draws_consumed;
        return value ^ (value >> 18U);
    }
};

__device__ std::uint32_t random_billion(OgdfMt19937& random) {
    constexpr std::uint64_t range = 1000000001ULL;
    constexpr std::uint32_t threshold = 294967292U;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * range;
        if (static_cast<std::uint32_t>(product) >= threshold) return static_cast<std::uint32_t>(product >> 32U);
    }
}

// MSVC 2022's uniform_int_distribution(0, upper) implementation: multiply
// high with the same rejection threshold used by the OGDF runner.  This is
// separate from random_billion because hierarchy selection has a changing
// inclusive range for every Set::choose() call.
__device__ std::uint32_t random_inclusive(OgdfMt19937& random, std::uint32_t upper) {
    const std::uint32_t range = upper + 1U;
    const std::uint32_t threshold = static_cast<std::uint32_t>(-range) % range;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * range;
        if (static_cast<std::uint32_t>(product) >= threshold) {
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }
}

// This is intentionally one CUDA thread. OGDF's List/Set mutation and the
// discovery order of Graph::edges() are semantic inputs to FMMM coarsening;
// a parallel implementation would be a different algorithm.  The input is
// an already normalized level graph and no CPU selection result is supplied.
__global__ void full_path_solar_hierarchy_kernel(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, std::uint64_t seed, int* status, int* parent,
    int* node_type, int* dedicated_sun, double* sun_distance,
    int* coarse_node_count, int* coarse_edge_count, int* coarse_source,
    int* coarse_target, double* coarse_length,
    std::uint8_t* plan_moon_edge, int* plan_low_adj_begin, int* plan_low_adj_edge,
    int* plan_lambda_begin, double* plan_lambda_value, int* plan_neighbour_begin,
    int* plan_neighbour_node, int* plan_moon_begin, int* plan_moon_node,
    int* plan_high_lower_node, int* plan_high_edge_span, int* plan_high_source,
    int* plan_high_target, std::uint8_t* plan_high_extra, int* plan_high_adj_begin,
    int* plan_high_adj_edge, int* draws_consumed,
    lcn::release::FmmmWorkspaceView workspace) {
    if (blockIdx.x || threadIdx.x) return;
    if (draws_consumed != nullptr) *draws_consumed = 0;
    if (node_count <= 0 || edge_count < 0) {
        *status = kPlacementUnsupported;
        return;
    }

    int* selectable = workspace.scratch_int;
    int* selectable_position = workspace.low_adj_begin;
    int* star_mass = workspace.high_adj_begin;
    int* type = workspace.lambda_begin;
    int* sun = workspace.neighbour_begin;
    int* sun_parent = workspace.moon_begin;
    int* moon_parent = workspace.cell_child4;
    double* distance_to_sun = workspace.scratch_double;
    for (int node = 0; node < node_count; ++node) {
        selectable[node] = selectable_position[node] = node;
        // The frozen profile starts every level-zero node with mass one.
        star_mass[node] = 1;
        type[node] = 0;
        sun[node] = -1;
        sun_parent[node] = -1;
        moon_parent[node] = -1;
        distance_to_sun[node] = 0.0;
        parent[node] = -1;
    }
    for (int edge = 0; edge < edge_count; ++edge) {
        const int source = edge_source[edge], target = edge_target[edge];
        if (source < 0 || source >= node_count || target < 0 || target >= node_count ||
            source == target || edge_length[edge] <= 0.0) {
            *status = kPlacementUnsupported;
            return;
        }
        ++star_mass[source];
        ++star_mass[target];
    }
    if (plan_moon_edge != nullptr) {
        for (int edge = 0; edge < edge_count; ++edge) plan_moon_edge[edge] = 0;
    }

    OgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(seed));
    int last_selectable = node_count - 1;
    int next_parent = 0;
    while (last_selectable >= 0) {
        int last_trial = last_selectable;
        int selected_index = -1;
        int selected_mass = 0;
        // RandomRandIterNr / LowerMass: retain the first equal-mass choice.
        for (int trial = 0; trial < 20 && last_trial >= 0; ++trial) {
            const int random_index = static_cast<int>(random_inclusive(
                random, static_cast<std::uint32_t>(last_trial)));
            const int candidate = selectable[random_index];
            const int candidate_mass = star_mass[candidate];
            const int tail = selectable[last_trial];
            selectable[last_trial] = candidate;
            selectable[random_index] = tail;
            selectable_position[candidate] = last_trial;
            selectable_position[tail] = random_index;
            --last_trial;
            if (trial == 0 || candidate_mass < selected_mass) {
                selected_index = last_trial + 1;
                selected_mass = candidate_mass;
            }
        }
        const int selected_sun = selectable[selected_index];
        const int tail = selectable[last_selectable];
        selectable[last_selectable] = selected_sun;
        selectable[selected_index] = tail;
        selectable_position[selected_sun] = last_selectable;
        selectable_position[tail] = selected_index;
        --last_selectable;

        type[selected_sun] = 1; // Sun
        sun[selected_sun] = selected_sun;
        sun_parent[selected_sun] = next_parent;
        parent[selected_sun] = next_parent++;

        // Ordered adjacency scan reproduces OGDF's adjacency iteration for
        // the normalized graph. A planet is deliberately assigned before it
        // is removed from the selectable Set, matching OGDF's two passes.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (source != selected_sun && target != selected_sun) continue;
            const int planet = source == selected_sun ? target : source;
            type[planet] = 2; // Planet
            sun[planet] = selected_sun;
            distance_to_sun[planet] = edge_length[edge];
            parent[planet] = sun_parent[selected_sun];
        }
        // Set::del() uses a swap with its live tail. Membership is therefore
        // represented by selectable_position <= last_selectable, not type.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (source != selected_sun && target != selected_sun) continue;
            const int planet = source == selected_sun ? target : source;
            if (selectable_position[planet] <= last_selectable) {
                const int position = selectable_position[planet];
                const int removed_tail = selectable[last_selectable];
                selectable[last_selectable] = planet;
                selectable[position] = removed_tail;
                selectable_position[planet] = last_selectable;
                selectable_position[removed_tail] = position;
                --last_selectable;
            }
        }
        // Mark the neighbours of planets unavailable. They remain type 0 and
        // become moons only in OGDF's second, shortest-incident-edge pass.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (source != selected_sun && target != selected_sun) continue;
            const int planet = source == selected_sun ? target : source;
            for (int moon_edge = 0; moon_edge < edge_count; ++moon_edge) {
                const int moon_source = edge_source[moon_edge], moon_target = edge_target[moon_edge];
                if (moon_source != planet && moon_target != planet) continue;
                const int possible_moon = moon_source == planet ? moon_target : moon_source;
                if (selectable_position[possible_moon] <= last_selectable) {
                    const int position = selectable_position[possible_moon];
                    const int removed_tail = selectable[last_selectable];
                    selectable[last_selectable] = possible_moon;
                    selectable[position] = removed_tail;
                    selectable_position[possible_moon] = last_selectable;
                    selectable_position[removed_tail] = position;
                    --last_selectable;
                }
            }
        }
    }

    for (int moon = 0; moon < node_count; ++moon) {
        if (type[moon] != 0) continue;
        int nearest = -1;
        int nearest_edge = -1;
        double nearest_length = 0.0;
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (source != moon && target != moon) continue;
            const int neighbour = source == moon ? target : source;
            if (type[neighbour] != 2 && type[neighbour] != 3) continue;
            // Strict comparison is intentional: OGDF keeps the first
            // normalized adjacency entry on tied BoundingCircle lengths.
            if (nearest == -1 || nearest_length > edge_length[edge]) {
                nearest = neighbour;
                nearest_edge = edge;
                nearest_length = edge_length[edge];
            }
        }
        if (nearest == -1) {
            *status = kHierarchyNoMoonNeighbour;
            return;
        }
        type[moon] = 4; // Moon
        sun[moon] = sun[nearest];
        distance_to_sun[moon] = nearest_length + distance_to_sun[nearest];
        parent[moon] = sun_parent[sun[moon]];
        moon_parent[moon] = nearest;
        if (plan_moon_edge != nullptr) plan_moon_edge[nearest_edge] = 1;
        type[nearest] = 3; // PlanetWithMoons
    }

    int output_edges = 0;
    for (int node = 0; node < node_count; ++node) {
        node_type[node] = type[node];
        dedicated_sun[node] = sun[node];
        sun_distance[node] = distance_to_sun[node];
    }
    for (int edge = 0; edge < edge_count; ++edge) {
        const int first_parent = parent[edge_source[edge]];
        const int second_parent = parent[edge_target[edge]];
        if (first_parent == second_parent) continue;
        // Keep OGDF newEdge(source_sun, target_sun) direction (not min/max).
        // Dedup is undirected, matching makeSimpleUndirected survivor selection:
        // first inter-sun edge of each undirected parent pair in scan order.
        const int source = first_parent;
        const int target = second_parent;
        const int lo = source < target ? source : target;
        const int hi = source < target ? target : source;
        int previous = -1;
        for (int prior = 0; prior < output_edges; ++prior) {
            const int plo = coarse_source[prior] < coarse_target[prior]
                ? coarse_source[prior] : coarse_target[prior];
            const int phi = coarse_source[prior] < coarse_target[prior]
                ? coarse_target[prior] : coarse_source[prior];
            if (plo == lo && phi == hi) {
                previous = prior;
                break;
            }
        }
        // FMMM's makeSimpleUndirected() drops parallel edges before the later
        // averaging loop, so the observable profile retains this first edge.
        if (previous == -1) {
            if (output_edges >= workspace.max_level_edges) {
                *status = kPlacementUnsupported;
                return;
            }
            // Multilevel recursion feeds these edges into the next solar pass;
            // discovery order and direction are semantic (do not sort).
            coarse_source[output_edges] = source;
            coarse_target[output_edges] = target;
            coarse_length[output_edges] = distance_to_sun[edge_source[edge]] +
                                          edge_length[edge] + distance_to_sun[edge_target[edge]];
            ++output_edges;
        }
    }
    // Export the exact ordered Multilevel attributes needed by Advanced
    // placement. This is kept inside the same serial CUDA owner: no host
    // container is permitted to rebuild lambda, moon, or adjacency order.
    if (plan_low_adj_begin != nullptr && plan_low_adj_edge != nullptr &&
            plan_lambda_begin != nullptr && plan_lambda_value != nullptr &&
            plan_neighbour_begin != nullptr && plan_neighbour_node != nullptr &&
            plan_moon_begin != nullptr && plan_moon_node != nullptr &&
            plan_high_lower_node != nullptr && plan_high_edge_span != nullptr &&
            plan_high_source != nullptr && plan_high_target != nullptr &&
            plan_high_extra != nullptr && plan_high_adj_begin != nullptr &&
            plan_high_adj_edge != nullptr) {
        int* degree = workspace.high_adj_begin;
        int* cursor = workspace.moon_begin;
        for (int node = 0; node < node_count; ++node) degree[node] = 0;
        for (int edge = 0; edge < edge_count; ++edge) {
            ++degree[edge_source[edge]]; ++degree[edge_target[edge]];
        }
        plan_low_adj_begin[0] = 0;
        for (int node = 0; node < node_count; ++node) {
            plan_low_adj_begin[node + 1] = plan_low_adj_begin[node] + degree[node];
            cursor[node] = plan_low_adj_begin[node];
        }
        for (int edge = 0; edge < edge_count; ++edge) {
            plan_low_adj_edge[cursor[edge_source[edge]]++] = edge;
            plan_low_adj_edge[cursor[edge_target[edge]]++] = edge;
        }

        int* lambda_count = workspace.scratch_int;
        for (int node = 0; node < node_count; ++node) lambda_count[node] = 0;
        for (int edge = 0; edge < edge_count; ++edge) {
            if (sun[edge_source[edge]] != sun[edge_target[edge]]) {
                ++lambda_count[edge_source[edge]]; ++lambda_count[edge_target[edge]];
            }
        }
        plan_lambda_begin[0] = plan_neighbour_begin[0] = 0;
        for (int node = 0; node < node_count; ++node) {
            plan_lambda_begin[node + 1] = plan_lambda_begin[node] + lambda_count[node];
            plan_neighbour_begin[node + 1] = plan_lambda_begin[node + 1];
            cursor[node] = plan_lambda_begin[node];
        }
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (sun[source] == sun[target]) continue;
            const double high_length = distance_to_sun[source] + edge_length[edge] + distance_to_sun[target];
            const int source_at = cursor[source]++, target_at = cursor[target]++;
            plan_lambda_value[source_at] = distance_to_sun[source] / high_length;
            plan_neighbour_node[source_at] = sun[target];
            plan_lambda_value[target_at] = distance_to_sun[target] / high_length;
            plan_neighbour_node[target_at] = sun[source];
        }

        for (int node = 0; node < node_count; ++node) degree[node] = 0;
        for (int node = 0; node < node_count; ++node) if (moon_parent[node] >= 0) ++degree[moon_parent[node]];
        plan_moon_begin[0] = 0;
        for (int node = 0; node < node_count; ++node) {
            plan_moon_begin[node + 1] = plan_moon_begin[node] + degree[node];
            cursor[node] = plan_moon_begin[node];
            plan_high_lower_node[node] = -1;
        }
        for (int node = 0; node < node_count; ++node) {
            if (moon_parent[node] >= 0) plan_moon_node[cursor[moon_parent[node]]++] = node;
            if (type[node] == 1) plan_high_lower_node[parent[node]] = node;
        }

        int high_raw_count = 0;
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = edge_source[edge], target = edge_target[edge];
            if (sun[source] == sun[target]) continue;
            plan_high_source[high_raw_count] = parent[source];
            plan_high_target[high_raw_count] = parent[target];
            plan_high_extra[high_raw_count] = 0;
            ++high_raw_count;
        }
        *plan_high_edge_span = high_raw_count;
        for (int node = 0; node < next_parent; ++node) degree[node] = 0;
        for (int edge = 0; edge < high_raw_count; ++edge) {
            const int a = plan_high_source[edge], b = plan_high_target[edge];
            const int lo = a < b ? a : b, hi = a < b ? b : a;
            bool first = true;
            for (int prior = 0; prior < edge; ++prior) {
                const int pa = plan_high_source[prior], pb = plan_high_target[prior];
                if ((pa < pb ? pa : pb) == lo && (pa < pb ? pb : pa) == hi) { first = false; break; }
            }
            if (first) { ++degree[a]; ++degree[b]; }
        }
        plan_high_adj_begin[0] = 0;
        for (int node = 0; node < next_parent; ++node) {
            plan_high_adj_begin[node + 1] = plan_high_adj_begin[node] + degree[node];
            cursor[node] = plan_high_adj_begin[node];
        }
        for (int edge = 0; edge < high_raw_count; ++edge) {
            const int a = plan_high_source[edge], b = plan_high_target[edge];
            const int lo = a < b ? a : b, hi = a < b ? b : a;
            bool first = true;
            for (int prior = 0; prior < edge; ++prior) {
                const int pa = plan_high_source[prior], pb = plan_high_target[prior];
                if ((pa < pb ? pa : pb) == lo && (pa < pb ? pb : pa) == hi) { first = false; break; }
            }
            if (first) { plan_high_adj_edge[cursor[a]++] = edge; plan_high_adj_edge[cursor[b]++] = edge; }
        }
    }
    *coarse_node_count = next_parent;
    *coarse_edge_count = output_edges;
    if (draws_consumed != nullptr) {
        *draws_consumed = static_cast<int>(random.draws_consumed);
    }
    *status = kPlacementOk;
}

__global__ void full_path_coarsest_placement_kernel(int node_count, double node_width, double node_height,
                                                     std::uint64_t seed, int* status, double* initial_box,
                                                     double* dynamic_box, double* down_left_x, double* down_left_y,
                                                     double* x, double* y) {
    if (blockIdx.x || threadIdx.x) return;
    if (node_count <= 0 || node_width < 10.0 || node_height < 10.0) {
        *status = kPlacementUnsupported;
        return;
    }
    // FMMMLayout::init_boxlength_and_cornercoordinate().
    const double box = ceil(fmax(node_count * node_width, node_count * node_height) * 1.1);
    *initial_box = box;
    OgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(seed));
    constexpr double billion = 1000000000.0;
    double xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0;
    for (int node = 0; node < node_count; ++node) {
        x[node] = static_cast<double>(random_billion(random)) / billion * (box - 2.0) + 1.0;
        y[node] = static_cast<double>(random_billion(random)) / billion * (box - 2.0) + 1.0;
        if (node == 0) {
            xmin = xmax = x[node];
            ymin = ymax = y[node];
        } else {
            xmin = fmin(xmin, x[node]); xmax = fmax(xmax, x[node]);
            ymin = fmin(ymin, y[node]); ymax = fmax(ymax, y[node]);
        }
    }
    // FMMMLayout::update_boxlength_and_cornercoordinate().
    *down_left_x = floor(xmin - 1.0);
    *down_left_y = floor(ymin - 1.0);
    *dynamic_box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
    if (*dynamic_box <= 2.0) {
        *dynamic_box = node_count * 20.0;
        *down_left_x = floor(xmin) - (*dynamic_box / 2.0);
        *down_left_y = floor(ymin) - (*dynamic_box / 2.0);
    }
    *status = kPlacementOk;
}

// FMMMLayout::set_average_ideal_edgelength(), retained as one ordered device
// summation because the frozen profile treats Graph::edges() order as state.
__global__ void average_ideal_length_kernel(int edge_count, const double* edge_length, double* average) {
    if (blockIdx.x || threadIdx.x) return;
    if (edge_count <= 0) { *average = 50.0; return; }
    double sum = 0.0;
    for (int edge = 0; edge < edge_count; ++edge) sum += edge_length[edge];
    *average = sum / static_cast<double>(edge_count);
}

// FMMMLayout::adapt_drawing_to_ideal_average_edgelength(). The final
// postprocess resize deliberately leaves the stored force box untouched.
__global__ void resize_to_ideal_length_kernel(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, double* x, double* y, int refresh_box,
    double* box, double* left_x, double* left_y, int* status) {
    if (blockIdx.x || threadIdx.x) return;
    if (node_count <= 0 || edge_count < 0) { *status = kPlacementUnsupported; return; }
    double ideal_sum = 0.0, real_sum = 0.0;
    for (int edge = 0; edge < edge_count; ++edge) {
        const int u = edge_source[edge], v = edge_target[edge];
        if (u < 0 || u >= node_count || v < 0 || v >= node_count || edge_length[edge] <= 0.0) { *status = kPlacementUnsupported; return; }
        ideal_sum += edge_length[edge];
        const double dx = x[u] - x[v], dy = y[u] - y[v];
        real_sum += sqrt(dx * dx + dy * dy);
    }
    // ENGINEERING_V1: OGDF also multiplies by resizingScalar() (default 1). No ABI
    // scalar field yet; scale is ideal_sum/real_sum only (default-1 freeze).
    const double scale = real_sum == 0.0 ? 1.0 : ideal_sum / real_sum;
    for (int node = 0; node < node_count; ++node) { x[node] *= scale; y[node] *= scale; }
    if (refresh_box != 0) {
        double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
        for (int node = 1; node < node_count; ++node) { xmin = fmin(xmin, x[node]); xmax = fmax(xmax, x[node]); ymin = fmin(ymin, y[node]); ymax = fmax(ymax, y[node]); }
        *left_x = floor(xmin - 1.0); *left_y = floor(ymin - 1.0);
        *box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
        if (*box <= 2.0) { *box = node_count * 20.0; *left_x = floor(xmin) - *box / 2.0; *left_y = floor(ymin) - *box / 2.0; }
    }
    *status = kPlacementOk;
}

__global__ void single_component_pack_export_kernel(int node_count, double* x, double* y, int* status) {
    if (blockIdx.x || threadIdx.x) return;
    if (node_count <= 0) { *status = kPlacementUnsupported; return; }
    constexpr double kPi = 3.14159265358979323846;
    double best_angle = 0.0, best_score = 0.0;
    for (int candidate = 0; candidate <= 10; ++candidate) {
        const double angle = kPi * static_cast<double>(candidate) / 22.0;
        const double c = cos(angle), s = sin(angle);
        double xmin = c * x[0] - s * y[0], xmax = xmin, ymin = s * x[0] + c * y[0], ymax = ymin;
        for (int node = 1; node < node_count; ++node) { const double px = c * x[node] - s * y[node], py = s * x[node] + c * y[node]; xmin = fmin(xmin, px); xmax = fmax(xmax, px); ymin = fmin(ymin, py); ymax = fmax(ymax, py); }
        const double width = xmax - xmin + 50.0, height = ymax - ymin + 50.0;
        const double ratio = width / height, score = width * height * (ratio < 1.0 ? 1.0 / ratio : ratio);
        if (candidate == 0 || score < best_score) { best_score = score; best_angle = angle; }
    }
    double c = cos(best_angle), s = sin(best_angle);
    double xmin = c * x[0] - s * y[0], xmax = xmin, ymin = s * x[0] + c * y[0], ymax = ymin;
    for (int node = 0; node < node_count; ++node) { const double px = c * x[node] - s * y[node], py = s * x[node] + c * y[node]; x[node] = px; y[node] = py; if (node) { xmin=fmin(xmin,px);xmax=fmax(xmax,px);ymin=fmin(ymin,py);ymax=fmax(ymax,py); } }
    if ((xmax - xmin) < (ymax - ymin)) { for (int node = 0; node < node_count; ++node) { const double px = -y[node]; y[node] = x[node]; x[node] = px; } xmin=xmax=x[0];ymin=ymax=y[0]; for(int node=1;node<node_count;++node){xmin=fmin(xmin,x[node]);xmax=fmax(xmax,x[node]);ymin=fmin(ymin,y[node]);ymax=fmax(ymax,y[node]);} }
    for (int node = 0; node < node_count; ++node) { x[node] = floor(x[node] - (xmin - 25.0)); y[node] = floor(y[node] - (ymin - 25.0)); }
    *status = kPlacementOk;
}

__global__ void exact_force_step_kernel(int node_count, int edge_count, const int* edge_source,
                                        const int* edge_target, const double* edge_length,
                                        const double* input_x, const double* input_y, double box_length,
                                        double down_left_x, double down_left_y,
                                        double average_ideal_length, int iteration, int* status,
                                        double* adjusted_x, double* adjusted_y, double* attr_x, double* attr_y,
                                        double* rep_x, double* rep_y, double* force_x, double* force_y,
                                        double* moved_x, double* moved_y, double* out_box,
                                        double* out_down_left_x, double* out_down_left_y) {
    if (blockIdx.x || threadIdx.x) return;
    if (node_count <= 0 || edge_count < 0 ||
        average_ideal_length <= 0.0 || iteration <= 0) {
        *status = kPlacementUnsupported;
        return;
    }
    double adjusted_box = box_length;
    double adjusted_left_x = down_left_x;
    double adjusted_left_y = down_left_y;
    for (int node = 0; node < node_count; ++node) {
        adjusted_x[node] = floor(input_x[node]);
        adjusted_y[node] = floor(input_y[node]);
        if (adjusted_x[node] < adjusted_left_x) { adjusted_box += 2.0; adjusted_left_x -= 2.0; }
        if (adjusted_y[node] < adjusted_left_y) { adjusted_box += 2.0; adjusted_left_y -= 2.0; }
        attr_x[node] = attr_y[node] = rep_x[node] = rep_y[node] = 0.0;
    }
    // OGDF Graph::edges order is retained exactly as supplied by the frozen
    // oracle.  All arithmetic is serial/double to preserve accumulation order.
    for (int edge = 0; edge < edge_count; ++edge) {
        const int u = edge_source[edge], v = edge_target[edge];
        if (u < 0 || u >= node_count || v < 0 || v >= node_count || u == v || edge_length[edge] <= 0.0) {
            *status = kPlacementUnsupported;
            return;
        }
        const double dx = adjusted_x[v] - adjusted_x[u];
        const double dy = adjusted_y[v] - adjusted_y[u];
        const double distance = sqrt(dx * dx + dy * dy);
        if (distance == 0.0) continue;
        const double length = edge_length[edge];
        const double scalar = log2(distance / length) * distance / (length * length * length);
        const double fx = scalar * dx;
        const double fy = scalar * dy;
        attr_x[u] += fx; attr_y[u] += fy;
        attr_x[v] -= fx; attr_y[v] -= fy;
    }
    for (int u = 0; u < node_count; ++u) {
        for (int v = u + 1; v < node_count; ++v) {
            const double dx = adjusted_x[v] - adjusted_x[u];
            const double dy = adjusted_y[v] - adjusted_y[u];
            const double distance_squared = dx * dx + dy * dy;
            if (distance_squared == 0.0) { *status = kPlacementUnsupported; return; }
            const double fx = dx / distance_squared;
            const double fy = dy / distance_squared;
            rep_x[v] += fx; rep_y[v] += fy;
            rep_x[u] -= fx; rep_y[u] -= fy;
        }
    }
    const double max_radius = iteration == 1 ? adjusted_box / 1000.0 : adjusted_box / 5.0;
    const double average_squared = average_ideal_length * average_ideal_length;
    for (int node = 0; node < node_count; ++node) {
        const double raw_x = average_squared * (attr_x[node] + rep_x[node]);
        const double raw_y = average_squared * (attr_y[node] + rep_y[node]);
        const double norm = sqrt(raw_x * raw_x + raw_y * raw_y);
        const double scale = norm == 0.0 ? 0.0 : fmin(norm * 0.05, max_radius) / norm;
        force_x[node] = scale * raw_x;
        force_y[node] = scale * raw_y;
        moved_x[node] = adjusted_x[node] + force_x[node];
        moved_y[node] = adjusted_y[node] + force_y[node];
    }
    double xmin = moved_x[0], xmax = moved_x[0], ymin = moved_y[0], ymax = moved_y[0];
    for (int node = 1; node < node_count; ++node) {
        xmin = fmin(xmin, moved_x[node]); xmax = fmax(xmax, moved_x[node]);
        ymin = fmin(ymin, moved_y[node]); ymax = fmax(ymax, moved_y[node]);
    }
    *out_down_left_x = floor(xmin - 1.0);
    *out_down_left_y = floor(ymin - 1.0);
    *out_box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
    if (*out_box <= 2.0) {
        *out_box = node_count * 20.0;
        *out_down_left_x = floor(xmin) - (*out_box / 2.0);
        *out_down_left_y = floor(ymin) - (*out_box / 2.0);
    }
    *status = kPlacementOk;
}

__global__ void exact_force_loop_kernel(int node_count, int edge_count, const int* edge_source,
                                        const int* edge_target, const double* edge_length, double* x, double* y,
                                        double box_length, double down_left_x, double down_left_y,
                                        double average_ideal_length, int iterations, int* status,
                                        double* out_box, double* out_down_left_x, double* out_down_left_y,
                                        int force_phase, lcn::release::FmmmWorkspaceView workspace) {
    if (blockIdx.x || threadIdx.x) return;
    if (node_count <= 0 || edge_count < 0 ||
        average_ideal_length <= 0.0 || iterations <= 0 || iterations > 10000 || force_phase < 0 || force_phase > 2) { *status = kPlacementUnsupported; return; }
    double* last_x = workspace.force_x;
    double* last_y = workspace.force_y;
    double* attr_x = workspace.mass;
    double* attr_y = workspace.center_x;
    double* rep_x = workspace.center_y;
    double* rep_y = workspace.scratch_double;
    for (int i = 0; i < node_count; ++i) { last_x[i] = 0.0; last_y[i] = 0.0; }
    constexpr double factors[] = {2.0, 2.0, 1.5, 1.0, 0.66666666, 0.5, 0.33333333, 0.33333333,
                                  0.5, 0.66666666, 1.0, 1.5, 2.0, 2.0};
    constexpr double pi_times_one_over_six = 0.52359878;
    double current_box = box_length;
    double current_left_x = down_left_x;
    double current_left_y = down_left_y;
    const double average_squared = average_ideal_length * average_ideal_length;
    for (int iteration = 1; iteration <= iterations; ++iteration) {
        for (int node = 0; node < node_count; ++node) {
            x[node] = floor(x[node]); y[node] = floor(y[node]);
            if (x[node] < current_left_x) { current_box += 2.0; current_left_x -= 2.0; }
            if (y[node] < current_left_y) { current_box += 2.0; current_left_y -= 2.0; }
            attr_x[node] = attr_y[node] = rep_x[node] = rep_y[node] = 0.0;
        }
        for (int edge = 0; edge < edge_count; ++edge) {
            const int u = edge_source[edge], v = edge_target[edge];
            if (u < 0 || u >= node_count || v < 0 || v >= node_count || u == v || edge_length[edge] <= 0.0) { *status = kPlacementUnsupported; return; }
            const double dx = x[v] - x[u], dy = y[v] - y[u];
            const double distance = sqrt(dx * dx + dy * dy);
            if (distance == 0.0) continue;
            const double length = edge_length[edge];
            const double scalar = log2(distance / length) * distance / (length * length * length);
            const double fx = scalar * dx, fy = scalar * dy;
            attr_x[u] += fx; attr_y[u] += fy; attr_x[v] -= fx; attr_y[v] -= fy;
        }
        for (int u = 0; u < node_count; ++u) for (int v = u + 1; v < node_count; ++v) {
            const double dx = x[v] - x[u], dy = y[v] - y[u];
            const double distance_squared = dx * dx + dy * dy;
            if (distance_squared == 0.0) { *status = kPlacementUnsupported; return; }
            const double fx = dx / distance_squared, fy = dy / distance_squared;
            rep_x[v] += fx; rep_y[v] += fy; rep_x[u] -= fx; rep_y[u] -= fy;
        }
        const double max_radius = iteration == 1 ? current_box / 1000.0 : current_box / 5.0;
        const double spring_strength = force_phase == 2 ? 2.0 : 1.0;
        const double repulsion_strength = force_phase == 2 ? fmin(0.2, 400.0 / static_cast<double>(node_count)) : 1.0;
        const double cool_factor = force_phase == 1 ? 0.1 : (force_phase == 2 ? (iteration <= iterations - 5 ? 0.2 : 0.02) : 1.0);
        for (int node = 0; node < node_count; ++node) {
            double fx = average_squared * (spring_strength * attr_x[node] + repulsion_strength * rep_x[node]);
            double fy = average_squared * (spring_strength * attr_y[node] + repulsion_strength * rep_y[node]);
            const double norm = sqrt(fx * fx + fy * fy);
            const double scale = norm == 0.0 ? 0.0 : fmin(norm * cool_factor * 0.05, max_radius) / norm;
            fx *= scale; fy *= scale;
            if (iteration > 1) {
                const double new_norm = sqrt(fx * fx + fy * fy);
                const double old_norm = sqrt(last_x[node] * last_x[node] + last_y[node] * last_y[node]);
                if (new_norm > 0.0 && old_norm > 0.0) {
                    double angle = atan2(fy, fx) - atan2(last_y[node], last_x[node]);
                    if (angle < 0.0) angle += 2.0 * 3.14159265358979323846;
                    int factor_index = static_cast<int>(ceil(angle / pi_times_one_over_six));
                    if (factor_index < 0) factor_index = 0; if (factor_index > 13) factor_index = 13;
                    const double quotient = old_norm * factors[factor_index] / new_norm;
                    if (quotient < 1.0) { fx *= quotient; fy *= quotient; }
                }
            }
            last_x[node] = fx; last_y[node] = fy; x[node] += fx; y[node] += fy;
        }
        double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
        for (int node = 1; node < node_count; ++node) { xmin = fmin(xmin, x[node]); xmax = fmax(xmax, x[node]); ymin = fmin(ymin, y[node]); ymax = fmax(ymax, y[node]); }
        current_left_x = floor(xmin - 1.0); current_left_y = floor(ymin - 1.0);
        current_box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
        if (current_box <= 2.0) { current_box = node_count * 20.0; current_left_x = floor(xmin) - current_box / 2.0; current_left_y = floor(ymin) - current_box / 2.0; }
    }
    *out_box = current_box; *out_down_left_x = current_left_x; *out_down_left_y = current_left_y; *status = kPlacementOk;
}

// G4-small: one CTA, one lane per output node.  Iterations remain a strict
// device-side dependency, but the independent node work inside an iteration
// is exposed to the warp scheduler.  The per-node edge and pair loops retain
// the serial accumulation order of exact_force_loop_kernel; only independent
// output nodes are concurrent.  This route is deliberately bounded to keep
// the production large-graph kernel unchanged.
constexpr int kFmmmSmallGraphExactThreshold = 175;

__global__ void exact_force_loop_small_kernel(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box, double* out_down_left_x, double* out_down_left_y,
    int force_phase, lcn::release::FmmmWorkspaceView workspace) {
    __shared__ double current_box;
    __shared__ double current_left_x;
    __shared__ double current_left_y;
    __shared__ int input_valid;
    __shared__ int early_stop_converged;
    // Cache coordinates in shared memory so inner repulsion loops do not hammer
    // global memory with O(N^2) DRAM reads per iteration.
    __shared__ double sh_x[256];
    __shared__ double sh_y[256];

    const int lane = threadIdx.x;
    if (lane == 0) {
        input_valid = node_count > 0 && edge_count >= 0 &&
                      average_ideal_length > 0.0 && iterations > 0 &&
                      iterations <= 10000 && force_phase >= 0 && force_phase <= 2;
        early_stop_converged = 0;
        if (input_valid) {
            for (int edge = 0; edge < edge_count; ++edge) {
                const int u = edge_source[edge], v = edge_target[edge];
                if (u < 0 || u >= node_count || v < 0 || v >= node_count ||
                    u == v || edge_length[edge] <= 0.0) {
                    input_valid = 0;
                    break;
                }
            }
        }
        *status = input_valid ? kPlacementOk : kPlacementUnsupported;
        current_box = box_length;
        current_left_x = down_left_x;
        current_left_y = down_left_y;
    }
    __syncthreads();
    if (!input_valid) return;

    double* last_x = workspace.force_x;
    double* last_y = workspace.force_y;
    double* attr_x = workspace.mass;
    double* attr_y = workspace.center_x;
    double* rep_x = workspace.center_y;
    double* rep_y = workspace.scratch_double;

    if (lane < node_count) {
        last_x[lane] = 0.0;
        last_y[lane] = 0.0;
    }
    __syncthreads();

    constexpr double factors[] = {2.0, 2.0, 1.5, 1.0, 0.66666666, 0.5, 0.33333333,
                                  0.33333333, 0.5, 0.66666666, 1.0, 1.5, 2.0, 2.0};
    constexpr double pi_times_one_over_six = 0.52359878;
    const double average_squared = average_ideal_length * average_ideal_length;

    for (int iteration = 1; iteration <= iterations; ++iteration) {
        // Flooring is independent; box expansion is intentionally kept in
        // the original node order below because it is a sticky state update.
        if (lane < node_count) {
            const double fx = floor(x[lane]);
            const double fy = floor(y[lane]);
            x[lane] = fx;
            y[lane] = fy;
            sh_x[lane] = fx;
            sh_y[lane] = fy;
        }
        __syncthreads();
        if (lane == 0) {
            for (int node = 0; node < node_count; ++node) {
                if (sh_x[node] < current_left_x) {
                    current_box += 2.0;
                    current_left_x -= 2.0;
                }
                if (sh_y[node] < current_left_y) {
                    current_box += 2.0;
                    current_left_y -= 2.0;
                }
            }
        }
        __syncthreads();

        if (lane < node_count) {
            double local_attr_x = 0.0, local_attr_y = 0.0;
            double local_rep_x = 0.0, local_rep_y = 0.0;

            // For one output node this is the same edge visitation order as
            // the serial kernel, without cross-node atomics.
            for (int edge = 0; edge < edge_count; ++edge) {
                const int u = edge_source[edge], v = edge_target[edge];
                if (u != lane && v != lane) continue;
                const double dx = sh_x[v] - sh_x[u];
                const double dy = sh_y[v] - sh_y[u];
                const double distance = sqrt(dx * dx + dy * dy);
                if (distance == 0.0) continue;
                const double length = edge_length[edge];
                const double scalar = log2(distance / length) * distance /
                                      (length * length * length);
                const double fx = scalar * dx;
                const double fy = scalar * dy;
                const double sign = u == lane ? 1.0 : -1.0;
                local_attr_x += sign * fx;
                local_attr_y += sign * fy;
            }

            // Inner repulsion loop now pulls exclusively from shared memory tiles
            // rather than thrashing global device memory.
            for (int other = 0; other < node_count; ++other) {
                if (other == lane) continue;
                const double dx = lane < other ? sh_x[other] - sh_x[lane]
                                               : sh_x[lane] - sh_x[other];
                const double dy = lane < other ? sh_y[other] - sh_y[lane]
                                               : sh_y[lane] - sh_y[other];
                const double distance_squared = dx * dx + dy * dy;
                if (distance_squared == 0.0) {
                    atomicExch(status, kPlacementUnsupported);
                    continue;
                }
                const double fx = dx / distance_squared;
                const double fy = dy / distance_squared;
                if (other < lane) {
                    local_rep_x += fx;
                    local_rep_y += fy;
                } else {
                    local_rep_x -= fx;
                    local_rep_y -= fy;
                }
            }
            attr_x[lane] = local_attr_x;
            attr_y[lane] = local_attr_y;
            rep_x[lane] = local_rep_x;
            rep_y[lane] = local_rep_y;
        }
        __syncthreads();

        const double max_radius = iteration == 1 ? current_box / 1000.0 : current_box / 5.0;
        const double spring_strength = force_phase == 2 ? 2.0 : 1.0;
        const double repulsion_strength = force_phase == 2
            ? fmin(0.2, 400.0 / static_cast<double>(node_count)) : 1.0;
        const double cool_factor = force_phase == 1
            ? 0.1 : (force_phase == 2 ? (iteration <= iterations - 5 ? 0.2 : 0.02) : 1.0);

        if (lane < node_count) {
            double fx = average_squared *
                        (spring_strength * attr_x[lane] + repulsion_strength * rep_x[lane]);
            double fy = average_squared *
                        (spring_strength * attr_y[lane] + repulsion_strength * rep_y[lane]);
            const double norm = sqrt(fx * fx + fy * fy);
            const double scale = norm == 0.0 ? 0.0 : fmin(norm * cool_factor * 0.05, max_radius) / norm;
            fx *= scale;
            fy *= scale;
            if (iteration > 1) {
                const double new_norm = sqrt(fx * fx + fy * fy);
                const double old_norm = sqrt(last_x[lane] * last_x[lane] + last_y[lane] * last_y[lane]);
                if (new_norm > 0.0 && old_norm > 0.0) {
                    double angle = atan2(fy, fx) - atan2(last_y[lane], last_x[lane]);
                    if (angle < 0.0) angle += 2.0 * 3.14159265358979323846;
                    int factor_index = static_cast<int>(ceil(angle / pi_times_one_over_six));
                    if (factor_index < 0) factor_index = 0;
                    if (factor_index > 13) factor_index = 13;
                    const double quotient = old_norm * factors[factor_index] / new_norm;
                    if (quotient < 1.0) {
                        fx *= quotient;
                        fy *= quotient;
                    }
                }
            }
            last_x[lane] = fx;
            last_y[lane] = fy;
            x[lane] += fx;
            y[lane] += fy;
        }
        __syncthreads();

        if (lane == 0) {
            double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
            double max_displacement = 0.0;
            for (int node = 0; node < node_count; ++node) {
                const double d = sqrt(last_x[node] * last_x[node] + last_y[node] * last_y[node]);
                if (d > max_displacement) max_displacement = d;
                xmin = fmin(xmin, x[node]);
                xmax = fmax(xmax, x[node]);
                ymin = fmin(ymin, y[node]);
                ymax = fmax(ymax, y[node]);
            }
            current_left_x = floor(xmin - 1.0);
            current_left_y = floor(ymin - 1.0);
            current_box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
            if (current_box <= 2.0) {
                current_box = node_count * 20.0;
                current_left_x = floor(xmin) - current_box / 2.0;
                current_left_y = floor(ymin) - current_box / 2.0;
            }
            // Dynamic convergence check: if layout displacement drops below
            // a fraction of ideal length after a minimal warmup, bail out early.
            // No point burning GPU cycles on sub-atomic node jitter.
            if (iteration >= 5 && max_displacement < 0.001 * average_ideal_length) {
                early_stop_converged = 1;
            }
        }
        __syncthreads();
        if (early_stop_converged) break;
    }

    if (lane == 0) {
        *out_box = current_box;
        *out_down_left_x = current_left_x;
        *out_down_left_y = current_left_y;
    }
}

bool g4_small_route_enabled() {
    // The first bounded candidate did not meet the E2E acceptance gate. Keep
    // the proven serial route as the production default until a later small
    // graph candidate demonstrates a clean same-binary win.
    const char* enabled = std::getenv("LCN_FMMM_G4_SMALL_ENABLE");
    return enabled != nullptr && enabled[0] == '1';
}

void launch_exact_force_loop_kernel_dispatch(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box, double* out_down_left_x, double* out_down_left_y,
    int force_phase, const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream) {
    if (g4_small_route_enabled() && node_count > 0 &&
        node_count < kFmmmSmallGraphExactThreshold) {
        exact_force_loop_small_kernel<<<1, 256, 0, stream>>>(
            node_count, edge_count, edge_source, edge_target, edge_length, x, y,
            box_length, down_left_x, down_left_y, average_ideal_length, iterations,
            status, out_box, out_down_left_x, out_down_left_y, force_phase, workspace);
    } else {
        exact_force_loop_kernel<<<1, 1, 0, stream>>>(
            node_count, edge_count, edge_source, edge_target, edge_length, x, y,
            box_length, down_left_x, down_left_y, average_ideal_length, iterations,
            status, out_box, out_down_left_x, out_down_left_y, force_phase, workspace);
    }
}

// Floor positions, expand box if needed, compute attractive (spring) forces only.
// Repulsion is supplied later by G5 fmmm_nmm_force into rep_x/rep_y.
__global__ void nmm_force_prep_kernel(int node_count, int edge_count, const int* edge_source,
                                      const int* edge_target, const double* edge_length, double* x,
                                      double* y, double box_length, double down_left_x,
                                      double down_left_y, double* attr_x, double* attr_y,
                                      double* box_count_x, double* box_count_y,
                                      double* out_box, double* out_down_left_x,
                                      double* out_down_left_y, int* status) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) *status = kPlacementOk;
    if (node_count <= 0 || edge_count < 0) {
        if (tid == 0) *status = kPlacementUnsupported;
        return;
    }
    for (int node = tid; node < node_count; node += blockDim.x * gridDim.x) {
        x[node] = floor(x[node]);
        y[node] = floor(y[node]);
        box_count_x[node] = x[node] < down_left_x ? 1.0 : 0.0;
        box_count_y[node] = y[node] < down_left_y ? 1.0 : 0.0;
        attr_x[node] = 0.0;
        attr_y[node] = 0.0;
        // Keep the OGDF edge order for each node.  This avoids atomics and
        // preserves the original per-node summation contract while exposing
        // independent node work to the grid.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int u = edge_source[edge], v = edge_target[edge];
            if (u < 0 || u >= node_count || v < 0 || v >= node_count || u == v ||
                edge_length[edge] <= 0.0) {
                *status = kPlacementUnsupported;
                continue;
            }
            if (u != node && v != node) continue;
            const double dx = x[v] - x[u], dy = y[v] - y[u];
            const double distance = sqrt(dx * dx + dy * dy);
            if (distance == 0.0) continue;
            const double length = edge_length[edge];
            const double scalar = log2(distance / length) * distance /
                                  (length * length * length);
            const double sign = u == node ? 1.0 : -1.0;
            attr_x[node] += sign * scalar * dx;
            attr_y[node] += sign * scalar * dy;
        }
    }
}

// The original prep expands the box by two for every out-of-box coordinate.
// Count flags are reduced in a fixed single-block tree so the result is
// deterministic and does not require floating-point atomics.
__global__ void nmm_force_prep_box_reduce_kernel(
    int node_count, double box_length, double down_left_x, double down_left_y,
    const double* count_x, const double* count_y, double* out_box,
    double* out_down_left_x, double* out_down_left_y) {
    __shared__ double sx[256];
    __shared__ double sy[256];
    const int lane = threadIdx.x;
    double local_x = 0.0, local_y = 0.0;
    for (int node = lane; node < node_count; node += blockDim.x) {
        local_x += count_x[node];
        local_y += count_y[node];
    }
    sx[lane] = local_x;
    sy[lane] = local_y;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sx[lane] += sx[lane + stride];
            sy[lane] += sy[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        *out_box = box_length + 2.0 * (sx[0] + sy[0]);
        *out_down_left_x = down_left_x - 2.0 * sx[0];
        *out_down_left_y = down_left_y - 2.0 * sy[0];
    }
}

// Combine attr + NMM rep, cool, prevent oscillations, move, refresh box.
__global__ void nmm_force_apply_kernel(int node_count, double* x, double* y, double box_length,
                                       double down_left_x, double down_left_y,
                                       double average_ideal_length, int iteration, int iterations,
                                       int force_phase, const double* attr_x, const double* attr_y,
                                       const double* rep_x, const double* rep_y, double* last_x,
                                       double* last_y, double* out_box, double* out_down_left_x,
                                       double* out_down_left_y, double* bbox_x,
                                       double* bbox_y, int* status) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) *status = kPlacementOk;
    if (node_count <= 0 || average_ideal_length <= 0.0 || iteration <= 0 || iterations <= 0 ||
        force_phase < 0 || force_phase > 2) {
        if (tid == 0) *status = kPlacementUnsupported;
        return;
    }
    constexpr double factors[] = {2.0, 2.0, 1.5, 1.0, 0.66666666, 0.5, 0.33333333, 0.33333333,
                                  0.5, 0.66666666, 1.0, 1.5, 2.0, 2.0};
    constexpr double pi_times_one_over_six = 0.52359878;
    const double average_squared = average_ideal_length * average_ideal_length;
    const double max_radius = iteration == 1 ? box_length / 1000.0 : box_length / 5.0;
    const double spring_strength = force_phase == 2 ? 2.0 : 1.0;
    const double repulsion_strength =
        force_phase == 2 ? fmin(0.2, 400.0 / static_cast<double>(node_count)) : 1.0;
    const double cool_factor =
        force_phase == 1 ? 0.1
                         : (force_phase == 2 ? (iteration <= iterations - 5 ? 0.2 : 0.02) : 1.0);
    for (int node = tid; node < node_count; node += blockDim.x * gridDim.x) {
        double fx =
            average_squared * (spring_strength * attr_x[node] + repulsion_strength * rep_x[node]);
        double fy =
            average_squared * (spring_strength * attr_y[node] + repulsion_strength * rep_y[node]);
        const double norm = sqrt(fx * fx + fy * fy);
        const double scale = norm == 0.0 ? 0.0 : fmin(norm * cool_factor * 0.05, max_radius) / norm;
        fx *= scale;
        fy *= scale;
        if (iteration > 1) {
            const double new_norm = sqrt(fx * fx + fy * fy);
            const double old_norm = sqrt(last_x[node] * last_x[node] + last_y[node] * last_y[node]);
            if (new_norm > 0.0 && old_norm > 0.0) {
                double angle = atan2(fy, fx) - atan2(last_y[node], last_x[node]);
                if (angle < 0.0) angle += 2.0 * 3.14159265358979323846;
                int factor_index = static_cast<int>(ceil(angle / pi_times_one_over_six));
                if (factor_index < 0) factor_index = 0;
                if (factor_index > 13) factor_index = 13;
                const double quotient = old_norm * factors[factor_index] / new_norm;
                if (quotient < 1.0) {
                    fx *= quotient;
                    fy *= quotient;
                }
            }
        }
        last_x[node] = fx;
        last_y[node] = fy;
        x[node] += fx;
        y[node] += fy;
        bbox_x[node] = x[node];
        bbox_y[node] = y[node];
    }
}

__global__ void nmm_force_apply_box_reduce_kernel(
    int node_count, const double* bbox_x, const double* bbox_y,
    double* out_box, double* out_down_left_x, double* out_down_left_y) {
    __shared__ double min_x[256], max_x[256], min_y[256], max_y[256];
    const int lane = threadIdx.x;
    double local_min_x = 1.0e300, local_max_x = -1.0e300;
    double local_min_y = 1.0e300, local_max_y = -1.0e300;
    for (int node = lane; node < node_count; node += blockDim.x) {
        const double px = bbox_x[node], py = bbox_y[node];
        local_min_x = fmin(local_min_x, px);
        local_max_x = fmax(local_max_x, px);
        local_min_y = fmin(local_min_y, py);
        local_max_y = fmax(local_max_y, py);
    }
    min_x[lane] = local_min_x; max_x[lane] = local_max_x;
    min_y[lane] = local_min_y; max_y[lane] = local_max_y;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            min_x[lane] = fmin(min_x[lane], min_x[lane + stride]);
            max_x[lane] = fmax(max_x[lane], max_x[lane + stride]);
            min_y[lane] = fmin(min_y[lane], min_y[lane + stride]);
            max_y[lane] = fmax(max_y[lane], max_y[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        const double xmin = min_x[0], xmax = max_x[0];
        const double ymin = min_y[0], ymax = max_y[0];
    double current_left_x = floor(xmin - 1.0);
    double current_left_y = floor(ymin - 1.0);
    double current_box = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
    if (current_box <= 2.0) {
        current_box = node_count * 20.0;
        current_left_x = floor(xmin) - current_box / 2.0;
        current_left_y = floor(ymin) - current_box / 2.0;
    }
    *out_box = current_box;
    *out_down_left_x = current_left_x;
    *out_down_left_y = current_left_y;
    }
}

} // namespace

// Launcher for G2 call-export: 1-CC pack_subGraph_drawings + Integer floor.
void fmmm_launch_single_component_pack_export(int node_count, double* x, double* y, int* status) {
    single_component_pack_export_kernel<<<1, 1>>>(node_count, x, y, status);
}

void launch_resize_to_ideal_length_kernel(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    int refresh_box, double* out_box, double* out_down_left_x, double* out_down_left_y,
    int* status, cudaStream_t stream) {
    resize_to_ideal_length_kernel<<<1, 1, 0, stream>>>(
        node_count, edge_count, edge_source, edge_target, edge_length,
        x, y, refresh_box, out_box, out_down_left_x, out_down_left_y, status);
}

FmmmFullPathPlacementDiagnostic fmmm_full_path_coarsest_placement(
    int node_count, double node_width, double node_height, std::uint64_t seed) {
    FmmmFullPathPlacementDiagnostic result;
    if (node_count <= 0 || node_width < 10.0 || node_height < 10.0) return result;
    DeviceBuffer<int> status(1);
    DeviceBuffer<double> initial_box(1), dynamic_box(1), down_left_x(1), down_left_y(1);
    DeviceBuffer<double> x(static_cast<std::size_t>(node_count)), y(static_cast<std::size_t>(node_count));
    full_path_coarsest_placement_kernel<<<1, 1>>>(node_count, node_width, node_height, seed, status.get(),
        initial_box.get(), dynamic_box.get(), down_left_x.get(), down_left_y.get(), x.get(), y.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path coarsest placement");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM full-path coarsest placement");
    status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;
    initial_box.download(&result.initial_box_length, 1);
    dynamic_box.download(&result.dynamic_box_length, 1);
    down_left_x.download(&result.down_left_x, 1);
    down_left_y.download(&result.down_left_y, 1);
    std::vector<double> xs(static_cast<std::size_t>(node_count)), ys(static_cast<std::size_t>(node_count));
    x.download(xs.data(), xs.size());
    y.download(ys.data(), ys.size());
    for (int node = 0; node < node_count; ++node) result.positions.emplace_back(xs[node], ys[node]);
    return result;
}

FmmmExactForceStepDiagnostic fmmm_full_path_exact_force_step(
    const std::vector<std::pair<double, double>>& positions,
    const std::vector<int>& edge_source, const std::vector<int>& edge_target,
    const std::vector<double>& edge_length, double box_length, double down_left_x,
    double down_left_y, double average_ideal_length, int iteration) {
    FmmmExactForceStepDiagnostic result;
    const int node_count = static_cast<int>(positions.size());
    const int edge_count = static_cast<int>(edge_source.size());
    if (node_count <= 0 ||
        edge_target.size() != edge_source.size() || edge_length.size() != edge_source.size()) return result;
    std::vector<double> input_x(static_cast<std::size_t>(node_count)), input_y(static_cast<std::size_t>(node_count));
    for (int node = 0; node < node_count; ++node) { input_x[node] = positions[node].first; input_y[node] = positions[node].second; }
    DeviceBuffer<int> source(static_cast<std::size_t>(edge_count)), target(static_cast<std::size_t>(edge_count)), status(1);
    DeviceBuffer<double> length(static_cast<std::size_t>(edge_count)), x(static_cast<std::size_t>(node_count)), y(static_cast<std::size_t>(node_count));
    DeviceBuffer<double> adjusted_x(static_cast<std::size_t>(node_count)), adjusted_y(static_cast<std::size_t>(node_count)), attr_x(static_cast<std::size_t>(node_count)), attr_y(static_cast<std::size_t>(node_count)), rep_x(static_cast<std::size_t>(node_count)), rep_y(static_cast<std::size_t>(node_count)), force_x(static_cast<std::size_t>(node_count)), force_y(static_cast<std::size_t>(node_count)), moved_x(static_cast<std::size_t>(node_count)), moved_y(static_cast<std::size_t>(node_count));
    DeviceBuffer<double> out_box(1), out_left_x(1), out_left_y(1);
    if (edge_count > 0) { source.upload(edge_source.data(), edge_source.size()); target.upload(edge_target.data(), edge_target.size()); length.upload(edge_length.data(), edge_length.size()); }
    x.upload(input_x.data(), input_x.size()); y.upload(input_y.data(), input_y.size());
    exact_force_step_kernel<<<1, 1>>>(node_count, edge_count, source.get(), target.get(), length.get(), x.get(), y.get(),
        box_length, down_left_x, down_left_y, average_ideal_length, iteration, status.get(), adjusted_x.get(), adjusted_y.get(),
        attr_x.get(), attr_y.get(), rep_x.get(), rep_y.get(), force_x.get(), force_y.get(), moved_x.get(), moved_y.get(),
        out_box.get(), out_left_x.get(), out_left_y.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path exact force step");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM full-path exact force step");
    status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;
    std::vector<double> ax(node_count), ay(node_count), atx(node_count), aty(node_count), rx(node_count), ry(node_count), fx(node_count), fy(node_count), mx(node_count), my(node_count);
    adjusted_x.download(ax.data(), ax.size()); adjusted_y.download(ay.data(), ay.size()); attr_x.download(atx.data(), atx.size()); attr_y.download(aty.data(), aty.size()); rep_x.download(rx.data(), rx.size()); rep_y.download(ry.data(), ry.size()); force_x.download(fx.data(), fx.size()); force_y.download(fy.data(), fy.size()); moved_x.download(mx.data(), mx.size()); moved_y.download(my.data(), my.size());
    out_box.download(&result.box_length, 1); out_left_x.download(&result.down_left_x, 1); out_left_y.download(&result.down_left_y, 1);
    for (int node = 0; node < node_count; ++node) { result.adjusted_positions.emplace_back(ax[node], ay[node]); result.attraction.emplace_back(atx[node], aty[node]); result.repulsion.emplace_back(rx[node], ry[node]); result.capped_force.emplace_back(fx[node], fy[node]); result.moved_positions.emplace_back(mx[node], my[node]); }
    return result;
}

FmmmExactForceLoopDiagnostic fmmm_full_path_exact_force_loop(
    const std::vector<std::pair<double, double>>& positions,
    const std::vector<int>& edge_source, const std::vector<int>& edge_target,
    const std::vector<double>& edge_length, double box_length, double down_left_x,
    double down_left_y, double average_ideal_length, int iterations) {
    FmmmExactForceLoopDiagnostic result;
    const int node_count = static_cast<int>(positions.size()), edge_count = static_cast<int>(edge_source.size());
    if (node_count <= 0 || edge_target.size() != edge_source.size() || edge_length.size() != edge_source.size()) return result;
    std::vector<double> host_x(node_count), host_y(node_count); for (int node = 0; node < node_count; ++node) { host_x[node] = positions[node].first; host_y[node] = positions[node].second; }
    DeviceBuffer<int> source(edge_count), target(edge_count), status(1); DeviceBuffer<double> length(edge_count), x(node_count), y(node_count), out_box(1), out_left_x(1), out_left_y(1);
    if (edge_count > 0) { source.upload(edge_source.data(), edge_source.size()); target.upload(edge_target.data(), edge_target.size()); length.upload(edge_length.data(), edge_length.size()); }
    x.upload(host_x.data(), host_x.size()); y.upload(host_y.data(), host_y.size());
    lcn::release::FmmmWorkspaceShape shape; shape.node_count = node_count; shape.edge_count = edge_count; shape.level_count = 1; shape.cell_capacity = node_count; shape.max_level_nodes = node_count; shape.max_level_edges = edge_count; shape.max_aux_entries = 512;
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { result.status = 1; return result; }
    launch_exact_force_loop_kernel_dispatch(node_count, edge_count, source.get(), target.get(), length.get(), x.get(), y.get(), box_length, down_left_x, down_left_y, average_ideal_length, iterations, status.get(), out_box.get(), out_left_x.get(), out_left_y.get(), 0, workspace.view(), nullptr);
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path exact force loop"); cuda_check(cudaDeviceSynchronize(), "CUDA FMMM full-path exact force loop"); status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;
    x.download(host_x.data(), host_x.size()); y.download(host_y.data(), host_y.size()); out_box.download(&result.box_length, 1); out_left_x.download(&result.down_left_x, 1); out_left_y.download(&result.down_left_y, 1);
    for (int node = 0; node < node_count; ++node) result.final_positions.emplace_back(host_x[node], host_y[node]);
    return result;
}

void fmmm_full_path_exact_force_loop_device(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, double* position_x, double* position_y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box_length, double* out_down_left_x, double* out_down_left_y,
    int force_phase) {
    // We must pass a workspace here. Wait, this function is only called from native coordinator.
    lcn::release::FmmmWorkspaceShape shape; shape.node_count = node_count; shape.edge_count = edge_count; shape.level_count = 1; shape.cell_capacity = node_count; shape.max_level_nodes = node_count; shape.max_level_edges = edge_count; shape.max_aux_entries = 512;
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { *status = 1; return; }
    launch_exact_force_loop_kernel_dispatch(node_count, edge_count, edge_source, edge_target, edge_length,
        position_x, position_y, box_length, down_left_x, down_left_y, average_ideal_length,
        iterations, status, out_box_length, out_down_left_x, out_down_left_y, force_phase, workspace.view(), nullptr);
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path exact force device loop");
}

FmmmFullPathHierarchyDiagnostic fmmm_full_path_solar_hierarchy(
    int node_count, const std::vector<int>& edge_source,
    const std::vector<int>& edge_target, const std::vector<double>& edge_length,
    std::uint64_t seed) {
    FmmmFullPathHierarchyDiagnostic result;
    if (node_count <= 0 ||
        edge_source.size() != edge_target.size() || edge_source.size() != edge_length.size()) return result;
    const int edge_count = static_cast<int>(edge_source.size());
    DeviceBuffer<int> source(static_cast<std::size_t>(edge_count));
    DeviceBuffer<int> target(static_cast<std::size_t>(edge_count));
    DeviceBuffer<double> length(static_cast<std::size_t>(edge_count));
    if (edge_count > 0) {
        source.upload(edge_source.data(), edge_source.size());
        target.upload(edge_target.data(), edge_target.size());
        length.upload(edge_length.data(), edge_length.size());
    }
    DeviceBuffer<int> status(1), parent(static_cast<std::size_t>(node_count)),
        node_type(static_cast<std::size_t>(node_count)), dedicated_sun(static_cast<std::size_t>(node_count)),
        coarse_node_count(1), coarse_edge_count(1), coarse_source(edge_count),
        coarse_target(edge_count);
    DeviceBuffer<double> sun_distance(static_cast<std::size_t>(node_count)), coarse_length(edge_count);
    lcn::release::FmmmWorkspaceShape shape; shape.node_count = node_count; shape.edge_count = edge_count; shape.level_count = 1; shape.cell_capacity = node_count; shape.max_level_nodes = node_count; shape.max_level_edges = edge_count; shape.max_aux_entries = 512; shape.low_adj_count = edge_count * 2; shape.high_adj_count = edge_count * 2; shape.lambda_count = edge_count * 2; shape.neighbour_count = edge_count * 2; shape.moon_count = edge_count * 2;
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { result.status = 1; return result; }
    full_path_solar_hierarchy_kernel<<<1, 1>>>(
        node_count, edge_count, source.get(), target.get(), length.get(), seed, status.get(), parent.get(),
        node_type.get(), dedicated_sun.get(), sun_distance.get(), coarse_node_count.get(),
        coarse_edge_count.get(), coarse_source.get(), coarse_target.get(), coarse_length.get(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, workspace.view());
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path solar hierarchy");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM full-path solar hierarchy");
    status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;

    int output_edge_count = 0;
    coarse_node_count.download(&result.coarse_node_count, 1);
    coarse_edge_count.download(&output_edge_count, 1);
    if (result.coarse_node_count <= 0 || result.coarse_node_count > node_count ||
        output_edge_count < 0 || output_edge_count > kForceMaxEdges) {
        result.status = kPlacementUnsupported;
        result.coarse_node_count = 0;
        return result;
    }
    result.parent.resize(static_cast<std::size_t>(node_count));
    result.node_type.resize(static_cast<std::size_t>(node_count));
    result.dedicated_sun.resize(static_cast<std::size_t>(node_count));
    result.sun_distance.resize(static_cast<std::size_t>(node_count));
    parent.download(result.parent.data(), result.parent.size());
    node_type.download(result.node_type.data(), result.node_type.size());
    dedicated_sun.download(result.dedicated_sun.data(), result.dedicated_sun.size());
    sun_distance.download(result.sun_distance.data(), result.sun_distance.size());
    if (output_edge_count > 0) {
        std::vector<int> result_source(static_cast<std::size_t>(output_edge_count));
        std::vector<int> result_target(static_cast<std::size_t>(output_edge_count));
        std::vector<double> result_length(static_cast<std::size_t>(output_edge_count));
        cuda_check(cudaMemcpy(result_source.data(), coarse_source.get(),
                               result_source.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM hierarchy sources");
        cuda_check(cudaMemcpy(result_target.data(), coarse_target.get(),
                               result_target.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM hierarchy targets");
        cuda_check(cudaMemcpy(result_length.data(), coarse_length.get(),
                               result_length.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM hierarchy lengths");
        result.coarse_edges.reserve(static_cast<std::size_t>(output_edge_count));
        for (int edge = 0; edge < output_edge_count; ++edge) {
            // Witness canonicalization: undirected min/max + lexicographic sort.
            // Multilevel recursion keeps directed discovery order on device.
            const int a = result_source[edge], b = result_target[edge];
            result.coarse_edges.push_back({a < b ? a : b, a < b ? b : a, result_length[edge]});
        }
        std::sort(result.coarse_edges.begin(), result.coarse_edges.end(),
                  [](const FmmmFullPathHierarchyEdge& a, const FmmmFullPathHierarchyEdge& b) {
                      return a.source < b.source || (a.source == b.source && a.target < b.target);
                  });
    }
    return result;
}

FmmmAdvancedPlanDiagnostic fmmm_full_path_advanced_plan(
    int node_count, const std::vector<int>& edge_source,
    const std::vector<int>& edge_target, const std::vector<double>& edge_length,
    std::uint64_t seed) {
    FmmmAdvancedPlanDiagnostic result;
    if (node_count <= 0 ||
            edge_source.size() != edge_target.size() || edge_source.size() != edge_length.size()) return result;
    const int edge_count = static_cast<int>(edge_source.size());
    DeviceBuffer<int> source(edge_count), target(edge_count), status(1), parent(node_count), type(node_count), sun(node_count),
        coarse_nodes(1), coarse_edges(1), coarse_source(edge_count), coarse_target(edge_count),
        low_adj_begin(node_count + 1), low_adj_edge(2 * edge_count), lambda_begin(node_count + 1), neighbour_begin(node_count + 1),
        neighbour_node(2 * edge_count), moon_begin(node_count + 1), moon_node(node_count), high_lower(node_count), high_span(1),
        high_source(edge_count), high_target(edge_count), high_adj_begin(node_count + 1), high_adj_edge(2 * edge_count);
    DeviceBuffer<std::uint8_t> moon_edge(edge_count), high_extra(edge_count);
    DeviceBuffer<double> length(edge_count), distance(node_count), coarse_length(edge_count), lambda_value(2 * edge_count);
    if (edge_count > 0) { source.upload(edge_source.data(), edge_source.size()); target.upload(edge_target.data(), edge_target.size()); length.upload(edge_length.data(), edge_length.size()); }
    lcn::release::FmmmWorkspaceShape shape; shape.node_count = node_count; shape.edge_count = edge_count; shape.level_count = 1; shape.cell_capacity = node_count; shape.max_level_nodes = node_count; shape.max_level_edges = edge_count; shape.max_aux_entries = 512; shape.low_adj_count = edge_count * 2; shape.high_adj_count = edge_count * 2; shape.lambda_count = edge_count * 2; shape.neighbour_count = edge_count * 2; shape.moon_count = edge_count * 2;
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { result.status = 1; return result; }
    full_path_solar_hierarchy_kernel<<<1, 1>>>(node_count, edge_count, source.get(), target.get(), length.get(), seed,
        status.get(), parent.get(), type.get(), sun.get(), distance.get(), coarse_nodes.get(), coarse_edges.get(),
        coarse_source.get(), coarse_target.get(), coarse_length.get(), moon_edge.get(), low_adj_begin.get(), low_adj_edge.get(),
        lambda_begin.get(), lambda_value.get(), neighbour_begin.get(), neighbour_node.get(), moon_begin.get(), moon_node.get(),
        high_lower.get(), high_span.get(), high_source.get(), high_target.get(), high_extra.get(), high_adj_begin.get(),
        high_adj_edge.get(), nullptr, workspace.view());
    cuda_check(cudaGetLastError(), "CUDA FMMM full-path Advanced-plan hierarchy"); cuda_check(cudaDeviceSynchronize(), "CUDA FMMM full-path Advanced-plan hierarchy");
    status.download(&result.status, 1); if (result.status != kPlacementOk) return result;
    int high_count = 0, high_edge_span = 0;
    coarse_nodes.download(&high_count, 1); high_span.download(&high_edge_span, 1);
    if (high_count <= 0 || high_count > node_count || high_edge_span < 0 || high_edge_span > edge_count) { result.status = kPlacementUnsupported; return result; }
    result.lower_node_count = node_count; result.lower_edge_ordinal_span = edge_count; result.high_node_count = high_count; result.high_edge_ordinal_span = high_edge_span;
    result.lower_type.resize(node_count); result.lower_dedicated_sun.resize(node_count); result.lower_dedicated_sun_distance.resize(node_count); result.high_lower_node.resize(high_count);
    type.download(result.lower_type.data(), result.lower_type.size()); sun.download(result.lower_dedicated_sun.data(), result.lower_dedicated_sun.size()); distance.download(result.lower_dedicated_sun_distance.data(), result.lower_dedicated_sun_distance.size());
    std::vector<int> high_lower_full(static_cast<size_t>(node_count)); high_lower.download(high_lower_full.data(), high_lower_full.size()); std::copy_n(high_lower_full.begin(), high_count, result.high_lower_node.begin());
    result.lower_edge_source = edge_source; result.lower_edge_target = edge_target; result.lower_edge_length = edge_length; result.lower_edge_is_moon.resize(edge_count);
    result.lower_adjacency_begin.resize(node_count + 1); result.lambda_begin.resize(node_count + 1); result.neighbour_sun_begin.resize(node_count + 1); result.moon_begin.resize(node_count + 1); result.high_adjacency_begin.resize(high_count + 1);
    low_adj_begin.download(result.lower_adjacency_begin.data(), result.lower_adjacency_begin.size()); lambda_begin.download(result.lambda_begin.data(), result.lambda_begin.size()); neighbour_begin.download(result.neighbour_sun_begin.data(), result.neighbour_sun_begin.size()); moon_begin.download(result.moon_begin.data(), result.moon_begin.size());
    std::vector<int> high_adjacency_full(static_cast<size_t>(node_count + 1)); high_adj_begin.download(high_adjacency_full.data(), high_adjacency_full.size()); std::copy_n(high_adjacency_full.begin(), high_count + 1, result.high_adjacency_begin.begin());
    const int low_adj_count = result.lower_adjacency_begin.back(), lambda_count = result.lambda_begin.back(), moon_count = result.moon_begin.back(), high_adj_count = result.high_adjacency_begin.back();
    if (low_adj_count < 0 || low_adj_count > 2 * edge_count || lambda_count < 0 || lambda_count > 2 * edge_count || moon_count < 0 || moon_count > node_count || high_adj_count < 0 || high_adj_count > 2 * edge_count) { result.status = kPlacementUnsupported; return result; }
    result.lower_adjacency_edge.resize(low_adj_count); result.lambda_values.resize(lambda_count); result.neighbour_sun_nodes.resize(lambda_count); result.moon_nodes.resize(moon_count); result.high_edge_source.resize(high_edge_span); result.high_edge_target.resize(high_edge_span); result.high_edge_is_extra.resize(high_edge_span); result.high_adjacency_edge.resize(high_adj_count);
    if (low_adj_count) low_adj_edge.download(result.lower_adjacency_edge.data(), result.lower_adjacency_edge.size());
    std::vector<double> lambda_full(static_cast<size_t>(2 * edge_count)); std::vector<int> neighbour_full(static_cast<size_t>(2 * edge_count)), moon_full(static_cast<size_t>(node_count)), high_source_full(static_cast<size_t>(edge_count)), high_target_full(static_cast<size_t>(edge_count)), high_adj_full(static_cast<size_t>(2 * edge_count)); std::vector<std::uint8_t> moon_edge_full(static_cast<size_t>(edge_count)), high_extra_full(static_cast<size_t>(edge_count));
    if (edge_count) { moon_edge.download(moon_edge_full.data(), moon_edge_full.size()); lambda_value.download(lambda_full.data(), lambda_full.size()); neighbour_node.download(neighbour_full.data(), neighbour_full.size()); high_source.download(high_source_full.data(), high_source_full.size()); high_target.download(high_target_full.data(), high_target_full.size()); high_extra.download(high_extra_full.data(), high_extra_full.size()); high_adj_edge.download(high_adj_full.data(), high_adj_full.size()); }
    if (node_count) moon_node.download(moon_full.data(), moon_full.size());
    for (int edge = 0; edge < edge_count; ++edge) result.lower_edge_is_moon[edge] = moon_edge_full[edge];
    std::copy_n(lambda_full.begin(), lambda_count, result.lambda_values.begin()); std::copy_n(neighbour_full.begin(), lambda_count, result.neighbour_sun_nodes.begin()); std::copy_n(moon_full.begin(), moon_count, result.moon_nodes.begin()); std::copy_n(high_source_full.begin(), high_edge_span, result.high_edge_source.begin()); std::copy_n(high_target_full.begin(), high_edge_span, result.high_edge_target.begin()); for (int edge = 0; edge < high_edge_span; ++edge) result.high_edge_is_extra[edge] = high_extra_full[edge]; std::copy_n(high_adj_full.begin(), high_adj_count, result.high_adjacency_edge.begin());
    return result;
}

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_hierarchy_force_uncoarsen(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, const std::vector<std::pair<double, double>>& coarse_initial,
    double coarse_box_length, double coarse_down_left_x, double coarse_down_left_y,
    double average_ideal_length, int force_iterations, std::uint64_t seed,
    int placement_draws_consumed, int fine_normal_iterations,
    int postprocess_iterations, int fine_tuning_iterations, int pack_single_component) {
    FmmmNativeUncoarsenDiagnostic result;
    if (node_count <= 0 || fine_source.empty()
            || fine_source.size() != fine_target.size() || fine_source.size() != fine_length.size()
            || (!coarse_initial.empty() && coarse_box_length <= 0.0)
            || average_ideal_length <= 0.0 || force_iterations <= 0
            || placement_draws_consumed < 0 || fine_normal_iterations < 0 || fine_normal_iterations > 100
            || postprocess_iterations < 0 || postprocess_iterations > 100
            || fine_tuning_iterations < 0 || fine_tuning_iterations > 100 || (pack_single_component != 0 && pack_single_component != 1)) return result;

    // All graph-derived planes are allocated once and stay resident across
    // hierarchy, exact coarse force, and Advanced uncoarsening.  The two
    // scalar count downloads below only select bounded launch dimensions;
    // no hierarchy or coordinate state is reconstructed on the host.
    const int edge_count = static_cast<int>(fine_source.size());
    DeviceBuffer<int> source(edge_count), target(edge_count), hierarchy_status(1), parent(node_count), type(node_count), sun(node_count),
        coarse_nodes(1), coarse_edges(1), coarse_source(edge_count), coarse_target(edge_count),
        low_adj_begin(node_count + 1), low_adj_edge(2 * edge_count), lambda_begin(node_count + 1), neighbour_begin(node_count + 1),
        neighbour_node(2 * edge_count), moon_begin(node_count + 1), moon_node(node_count), high_lower(node_count), high_span(1),
        high_source(edge_count), high_target(edge_count), high_adj_begin(node_count + 1), high_adj_edge(2 * edge_count);
    DeviceBuffer<std::uint8_t> moon_edge(edge_count), high_extra(edge_count);
    DeviceBuffer<double> length(edge_count), distance(node_count), coarse_length(edge_count), lambda_value(2 * edge_count), fine_average(1);
    source.upload(fine_source.data(), fine_source.size());
    target.upload(fine_target.data(), fine_target.size());
    length.upload(fine_length.data(), fine_length.size());
    average_ideal_length_kernel<<<1, 1>>>(edge_count, length.get(), fine_average.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM native fine average launch");

    lcn::release::FmmmWorkspaceShape shape; shape.node_count = node_count; shape.edge_count = edge_count; shape.level_count = 1; shape.cell_capacity = node_count; shape.max_level_nodes = node_count; shape.max_level_edges = edge_count; shape.max_aux_entries = 512; shape.low_adj_count = edge_count * 2; shape.high_adj_count = edge_count * 2; shape.lambda_count = edge_count * 2; shape.neighbour_count = edge_count * 2; shape.moon_count = edge_count * 2;
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { result.status = 1; return result; }
    full_path_solar_hierarchy_kernel<<<1, 1>>>(
        node_count, edge_count, source.get(), target.get(), length.get(), seed,
        hierarchy_status.get(), parent.get(), type.get(), sun.get(), distance.get(),
        coarse_nodes.get(), coarse_edges.get(), coarse_source.get(), coarse_target.get(), coarse_length.get(),
        moon_edge.get(), low_adj_begin.get(), low_adj_edge.get(), lambda_begin.get(), lambda_value.get(),
        neighbour_begin.get(), neighbour_node.get(), moon_begin.get(), moon_node.get(), high_lower.get(),
        high_span.get(), high_source.get(), high_target.get(), high_extra.get(), high_adj_begin.get(),
        high_adj_edge.get(), nullptr, workspace.view());
    cuda_check(cudaGetLastError(), "CUDA FMMM native hierarchy launch");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native hierarchy");
    hierarchy_status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;

    int high_count = 0, high_edge_span = 0, high_force_edges = 0;
    coarse_nodes.download(&high_count, 1);
    high_span.download(&high_edge_span, 1);
    coarse_edges.download(&high_force_edges, 1);
    if (high_count <= 0 || high_count > node_count || high_edge_span < 0 || high_edge_span > edge_count
            || high_force_edges < 0
            || (!coarse_initial.empty() && coarse_initial.size() != static_cast<std::size_t>(high_count))) {
        result.status = kPlacementUnsupported;
        return result;
    }

    DeviceBuffer<double> high_x(node_count), high_y(node_count), force_box(1), force_left_x(1), force_left_y(1);
    DeviceBuffer<int> force_status(1);
    double force_box_length = coarse_box_length, force_down_left_x = coarse_down_left_x, force_down_left_y = coarse_down_left_y;
    int effective_placement_draws = placement_draws_consumed;
    if (coarse_initial.empty()) {
        // This branch is the frozen 20x20 coarsest-level placement profile.
        // It seeds the exact same device MT stream used by the hierarchy and
        // leaves the generated coordinate planes resident for force.
        DeviceBuffer<int> placement_status(1);
        DeviceBuffer<double> placement_initial_box(1), placement_dynamic_box(1), placement_left_x(1), placement_left_y(1);
        full_path_coarsest_placement_kernel<<<1, 1>>>(high_count, 20.0, 20.0, seed, placement_status.get(),
            placement_initial_box.get(), placement_dynamic_box.get(), placement_left_x.get(), placement_left_y.get(), high_x.get(), high_y.get());
        cuda_check(cudaGetLastError(), "CUDA FMMM native coarsest placement launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native coarsest placement");
        placement_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
        placement_dynamic_box.download(&force_box_length, 1);
        placement_left_x.download(&force_down_left_x, 1);
        placement_left_y.download(&force_down_left_y, 1);
        effective_placement_draws = 2 * high_count;
    } else {
        // The initial coarse positions are the frozen oracle input for this
        // first assembly seam.  They are uploaded once; exact force overwrites
        // the same allocation that Advanced placement reads below.
        std::vector<double> high_x_host(static_cast<std::size_t>(node_count), 0.0);
        std::vector<double> high_y_host(static_cast<std::size_t>(node_count), 0.0);
        for (int node = 0; node < high_count; ++node) {
            high_x_host[node] = coarse_initial[node].first;
            high_y_host[node] = coarse_initial[node].second;
        }
        high_x.upload(high_x_host.data(), high_x_host.size());
        high_y.upload(high_y_host.data(), high_y_host.size());
    }
    fmmm_full_path_exact_force_loop_device(
        high_count, high_force_edges, coarse_source.get(), coarse_target.get(), coarse_length.get(),
        high_x.get(), high_y.get(), force_box_length, force_down_left_x, force_down_left_y,
        average_ideal_length, force_iterations, force_status.get(), force_box.get(), force_left_x.get(), force_left_y.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM native coarse force launch");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native coarse force");
    force_status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;

    DeviceBuffer<double> position_x(node_count), position_y(node_count), angle_1(node_count), angle_2(node_count),
        output_box(1), output_left_x(1), output_left_y(1);
    DeviceBuffer<std::uint8_t> placed(node_count);
    DeviceBuffer<int> uncoarsen_status(1);
    const FmmmAdvancedUncoarsenPlanView plan{
        node_count, edge_count, high_count, high_edge_span,
        type.get(), sun.get(), distance.get(), high_lower.get(), high_x.get(), high_y.get(),
        source.get(), target.get(), length.get(), moon_edge.get(), low_adj_begin.get(), low_adj_edge.get(),
        high_source.get(), high_target.get(), high_extra.get(), high_adj_begin.get(), high_adj_edge.get(),
        lambda_begin.get(), lambda_value.get(), neighbour_begin.get(), neighbour_node.get(), moon_begin.get(), moon_node.get()};
    const FmmmAdvancedUncoarsenStateView state{
        position_x.get(), position_y.get(), placed.get(), angle_1.get(), angle_2.get(), output_box.get(),
        output_left_x.get(), output_left_y.get(), uncoarsen_status.get()};
    fmmm_advanced_uncoarsen_reference(plan, state, {seed, effective_placement_draws});
    cuda_check(cudaGetLastError(), "CUDA FMMM native Advanced uncoarsen launch");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native Advanced uncoarsen");
    uncoarsen_status.download(&result.status, 1);
    if (result.status != kPlacementOk) return result;

    if (fine_normal_iterations != 0) {
        double lower_box = 0.0, lower_left_x = 0.0, lower_left_y = 0.0, lower_average = 0.0;
        output_box.download(&lower_box, 1);
        output_left_x.download(&lower_left_x, 1);
        output_left_y.download(&lower_left_y, 1);
        fine_average.download(&lower_average, 1);
        DeviceBuffer<int> fine_status(1);
        DeviceBuffer<double> fine_box(1), fine_left_x(1), fine_left_y(1);
        fmmm_full_path_exact_force_loop_device(
            node_count, edge_count, source.get(), target.get(), length.get(), position_x.get(), position_y.get(),
            lower_box, lower_left_x, lower_left_y, lower_average, fine_normal_iterations,
            fine_status.get(), fine_box.get(), fine_left_x.get(), fine_left_y.get());
        cuda_check(cudaGetLastError(), "CUDA FMMM native fine normal force launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native fine normal force");
        fine_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
        fine_box.download(&result.box_length, 1);
        fine_left_x.download(&result.down_left_x, 1);
        fine_left_y.download(&result.down_left_y, 1);
    } else {
        output_box.download(&result.box_length, 1);
        output_left_x.download(&result.down_left_x, 1);
        output_left_y.download(&result.down_left_y, 1);
    }

    if (postprocess_iterations != 0) {
        double lower_average = 0.0;
        fine_average.download(&lower_average, 1);
        DeviceBuffer<int> post_status(1);
        DeviceBuffer<double> post_box(1), post_left_x(1), post_left_y(1);
        fmmm_full_path_exact_force_loop_device(
            node_count, edge_count, source.get(), target.get(), length.get(), position_x.get(), position_y.get(),
            result.box_length, result.down_left_x, result.down_left_y, lower_average, postprocess_iterations,
            post_status.get(), post_box.get(), post_left_x.get(), post_left_y.get(), 1);
        cuda_check(cudaGetLastError(), "CUDA FMMM native postprocess force launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native postprocess force");
        post_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
        post_box.download(&result.box_length, 1); post_left_x.download(&result.down_left_x, 1); post_left_y.download(&result.down_left_y, 1);

        resize_to_ideal_length_kernel<<<1, 1>>>(node_count, edge_count, source.get(), target.get(), length.get(),
            position_x.get(), position_y.get(), 1, post_box.get(), post_left_x.get(), post_left_y.get(), post_status.get());
        cuda_check(cudaGetLastError(), "CUDA FMMM native first resize launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native first resize");
        post_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
        post_box.download(&result.box_length, 1); post_left_x.download(&result.down_left_x, 1); post_left_y.download(&result.down_left_y, 1);

        if (fine_tuning_iterations != 0) {
            fmmm_full_path_exact_force_loop_device(
                node_count, edge_count, source.get(), target.get(), length.get(), position_x.get(), position_y.get(),
                result.box_length, result.down_left_x, result.down_left_y, lower_average, fine_tuning_iterations,
                post_status.get(), post_box.get(), post_left_x.get(), post_left_y.get(), 2);
            cuda_check(cudaGetLastError(), "CUDA FMMM native fine-tuning force launch");
            cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native fine-tuning force");
            post_status.download(&result.status, 1);
            if (result.status != kPlacementOk) return result;
            post_box.download(&result.box_length, 1); post_left_x.download(&result.down_left_x, 1); post_left_y.download(&result.down_left_y, 1);
        }

        // OGDF does this second resize without update_boxlength..., so the
        // published raw coordinates are scaled while the stored box remains
        // the fine-tuning endpoint box.
        resize_to_ideal_length_kernel<<<1, 1>>>(node_count, edge_count, source.get(), target.get(), length.get(),
            position_x.get(), position_y.get(), 0, post_box.get(), post_left_x.get(), post_left_y.get(), post_status.get());
        cuda_check(cudaGetLastError(), "CUDA FMMM native final resize launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM native final resize");
        post_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
    }

    if (pack_single_component != 0) {
        DeviceBuffer<int> pack_status(1);
        single_component_pack_export_kernel<<<1, 1>>>(node_count, position_x.get(), position_y.get(), pack_status.get());
        cuda_check(cudaGetLastError(), "CUDA FMMM single-component pack/export launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM single-component pack/export");
        pack_status.download(&result.status, 1);
        if (result.status != kPlacementOk) return result;
    }

    std::vector<double> x(static_cast<std::size_t>(node_count)), y(static_cast<std::size_t>(node_count));
    position_x.download(x.data(), x.size());
    position_y.download(y.data(), y.size());
    result.positions.reserve(static_cast<std::size_t>(node_count));
    for (int node = 0; node < node_count; ++node) result.positions.emplace_back(x[node], y[node]);
    return result;
}

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed) {
    return fmmm_full_path_native_hierarchy_force_uncoarsen(
        node_count, fine_source, fine_target, fine_length, {}, 0.0, 0.0, 0.0,
        average_ideal_length, force_iterations, seed, 0, 0);
}

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_normal_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed) {
    return fmmm_full_path_native_hierarchy_force_uncoarsen(
        node_count, fine_source, fine_target, fine_length, {}, 0.0, 0.0, 0.0,
        average_ideal_length, force_iterations, seed, 0, force_iterations);
}

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_postprocessed_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed) {
    return fmmm_full_path_native_hierarchy_force_uncoarsen(
        node_count, fine_source, fine_target, fine_length, {}, 0.0, 0.0, 0.0,
        average_ideal_length, force_iterations, seed, 0, force_iterations, 10, 20);
}

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_packed_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed) {
    return fmmm_full_path_native_hierarchy_force_uncoarsen(
        node_count, fine_source, fine_target, fine_length, {}, 0.0, 0.0, 0.0,
        average_ideal_length, force_iterations, seed, 0, force_iterations, 10, 20, 1);
}



struct FinalizeState {
    double min_x = 0;
    double max_x = 0;
    double min_y = 0;
    double max_y = 0;
    int first_bad_node = -1;
    int status_code = 0; // 0 = ok, 1 = extent, 2 = non-finite, 3 = range, 4 = duplicate
};

__global__ void fmmm_finalize_minmax_kernel(int node_count, const double* raw_x, const double* raw_y, int extent_x, int extent_y, FinalizeState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    state->status_code = 0;
    state->first_bad_node = -1;
    if (extent_x < 3 || extent_y < 3) {
        state->status_code = 1;
        state->first_bad_node = 0;
        return;
    }
    double minx = 0, maxx = 0, miny = 0, maxy = 0;
    for (int i = 0; i < node_count; ++i) {
        double vx = raw_x[i], vy = raw_y[i];
        if (!isfinite(vx) || !isfinite(vy)) {
            state->status_code = 2;
            state->first_bad_node = i;
            return;
        }
        if (i == 0) {
            minx = maxx = vx; miny = maxy = vy;
        } else {
            if (vx < minx) minx = vx;
            if (vx > maxx) maxx = vx;
            if (vy < miny) miny = vy;
            if (vy > maxy) maxy = vy;
        }
    }
    state->min_x = minx;
    state->max_x = maxx;
    state->min_y = miny;
    state->max_y = maxy;
}

__global__ void fmmm_finalize_quantize_kernel(int node_count, const double* raw_x, const double* raw_y, int extent_x, int extent_y, FinalizeState* state, std::int64_t* out_coords) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count || state->status_code != 0) return;
    double vx = raw_x[i], vy = raw_y[i];
    
    std::int64_t qx;
    if (state->max_x == state->min_x) qx = (extent_x - 1) / 2;
    else qx = static_cast<std::int64_t>(floor(1.0 + (vx - state->min_x) * (extent_x - 2) / (state->max_x - state->min_x) + 0.5));
    
    std::int64_t qy;
    if (state->max_y == state->min_y) qy = (extent_y - 1) / 2;
    else qy = static_cast<std::int64_t>(floor(1.0 + (vy - state->min_y) * (extent_y - 2) / (state->max_y - state->min_y) + 0.5));
    
    out_coords[2 * i] = qx;
    out_coords[2 * i + 1] = qy;
}

// Continuous affine: same channel as integer quantize, without floor(+0.5).
__global__ void fmmm_finalize_affine_continuous_kernel(int node_count, const double* raw_x,
                                                       const double* raw_y, int extent_x, int extent_y,
                                                       FinalizeState* state, double* out_x,
                                                       double* out_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count || state->status_code != 0) return;
    const double vx = raw_x[i], vy = raw_y[i];
    double qx;
    if (state->max_x == state->min_x) qx = static_cast<double>(extent_x) * 0.5;
    else qx = 1.0 + (vx - state->min_x) * static_cast<double>(extent_x - 2) /
                         (state->max_x - state->min_x);
    double qy;
    if (state->max_y == state->min_y) qy = static_cast<double>(extent_y) * 0.5;
    else qy = 1.0 + (vy - state->min_y) * static_cast<double>(extent_y - 2) /
                         (state->max_y - state->min_y);
    out_x[i] = qx;
    out_y[i] = qy;
}

__global__ void fmmm_finalize_validate_kernel(int node_count, const std::int64_t* out_coords, int extent_x, int extent_y, FinalizeState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || state->status_code != 0) return;
    for (int i = 0; i < node_count; ++i) {
        std::int64_t qx = out_coords[2 * i], qy = out_coords[2 * i + 1];
        if (qx <= 0 || qx >= extent_x || qy <= 0 || qy >= extent_y) {
            state->status_code = 3;
            state->first_bad_node = i;
            return;
        }
        for (int j = 0; j < i; ++j) {
            if (out_coords[2 * j] == qx && out_coords[2 * j + 1] == qy) {
                state->status_code = 4;
                state->first_bad_node = i;
                return;
            }
        }
    }
}

__global__ void fmmm_finalize_split_coords(int node_count, const std::int64_t* coords, std::int64_t* out_x, std::int64_t* out_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        out_x[i] = coords[2 * i];
        out_y[i] = coords[2 * i + 1];
    }
}

FmmmFinalResult fmmm_finalize_ogdf_profile_integer(
    DeviceGraph& graph,
    const DeviceBuffer<double>& raw_x,
    const DeviceBuffer<double>& raw_y,
    std::int64_t width,
    std::int64_t height,
    bool layers_enabled) {
    FmmmFinalResult res;
    res.status = FmmmFinalStatus::backend_error;
    
    int nodes = graph.node_count();
    if (nodes == 0) {
        res.status = FmmmFinalStatus::invalid_input;
        res.reason = "unsupported empty graph";
        return res;
    }
    if (width > 1000000000LL || height > 1000000000LL || width < 0 || height < 0) {
        res.status = FmmmFinalStatus::invalid_input;
        res.reason = "overflow";
        return res;
    }

    try {
        DeviceBuffer<FinalizeState> state(1);
        fmmm_finalize_minmax_kernel<<<1, 1>>>(nodes, raw_x.get(), raw_y.get(), static_cast<int>(width), static_cast<int>(height), state.get());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return res;

        DeviceBuffer<std::int64_t> cand_coords(nodes * 2);
        int threads = 256;
        int blocks = (nodes + threads - 1) / threads;
        fmmm_finalize_quantize_kernel<<<blocks, threads>>>(nodes, raw_x.get(), raw_y.get(), static_cast<int>(width), static_cast<int>(height), state.get(), cand_coords.get());
        if (cudaGetLastError() != cudaSuccess) return res;

        fmmm_finalize_validate_kernel<<<1, 1>>>(nodes, cand_coords.get(), static_cast<int>(width), static_cast<int>(height), state.get());
        if (cudaGetLastError() != cudaSuccess) return res;
        cudaDeviceSynchronize();

        FinalizeState h_state;
        state.download(&h_state, 1);
        if (h_state.status_code != 0) {
            res.status = FmmmFinalStatus::invalid_input;
            res.first_bad_node = h_state.first_bad_node;
            if (h_state.status_code == 1) res.reason = "extent < 3";
            else if (h_state.status_code == 2) res.reason = "non-finite coordinate";
            else if (h_state.status_code == 3) res.reason = "strict interior violation";
            else if (h_state.status_code == 4) res.reason = "quantization collision";
            return res;
        }

        DeviceBuffer<std::int64_t> cand_x(nodes);
        DeviceBuffer<std::int64_t> cand_y(nodes);
        fmmm_finalize_split_coords<<<blocks, threads>>>(nodes, cand_coords.get(), cand_x.get(), cand_y.get());
        if (cudaGetLastError() != cudaSuccess) return res;
        cudaDeviceSynchronize();

        DeviceGraphView view = graph.view();
        view.x = cand_x.get();
        view.y = cand_y.get();
        
        ScoreResult score = evaluate_full_square_candidate(view, layers_enabled);
        if (!score.valid) {
            res.status = FmmmFinalStatus::rejected_by_admission;
            res.reason = "rejected by gpu full square";
            return res;
        }
        
        graph.commit_device_coordinates(cand_coords);
        
        std::vector<std::int64_t> h_coords(nodes * 2);
        cand_coords.download(h_coords.data(), h_coords.size());
        for (int i = 0; i < nodes; ++i) {
            res.coordinates.push_back({static_cast<double>(h_coords[2 * i]),
                                       static_cast<double>(h_coords[2 * i + 1])});
        }
        res.score = score;
        res.status = FmmmFinalStatus::ok;
        res.reason = "ok";
    } catch (...) {
        res.status = FmmmFinalStatus::backend_error;
        res.reason = "exception";
    }
    
    return res;
}

FmmmFinalResult fmmm_finalize_ogdf_profile(
    DeviceGraph& graph,
    const DeviceBuffer<double>& raw_x,
    const DeviceBuffer<double>& raw_y,
    std::int64_t width,
    std::int64_t height,
    bool layers_enabled) {
    DeviceGraphView view = graph.view();
    return fmmm_finalize_ogdf_profile_raw(
        graph.node_count(), graph.edge_count(), view.source, view.target, view.layer,
        raw_x.get(), raw_y.get(), width, height, layers_enabled);
}

FmmmFinalResult fmmm_finalize_ogdf_profile_raw(
    int nodes,
    int edges_count,
    const int* d_source,
    const int* d_target,
    const int* d_layer,
    const double* raw_x,
    const double* raw_y,
    std::int64_t width,
    std::int64_t height,
    bool layers_enabled) {
    FmmmFinalResult res;
    res.status = FmmmFinalStatus::backend_error;

    if (nodes == 0) {
        res.status = FmmmFinalStatus::invalid_input;
        res.reason = "unsupported empty graph";
        return res;
    }
    if (width > 1000000000LL || height > 1000000000LL || width < 0 || height < 0) {
        res.status = FmmmFinalStatus::invalid_input;
        res.reason = "overflow";
        return res;
    }

    try {
        DeviceBuffer<FinalizeState> state(1);
        fmmm_finalize_minmax_kernel<<<1, 1>>>(nodes, raw_x, raw_y,
                                              static_cast<int>(width), static_cast<int>(height),
                                              state.get());
        if (cudaGetLastError() != cudaSuccess) return res;
        cudaDeviceSynchronize();

        FinalizeState h_state{};
        state.download(&h_state, 1);
        if (h_state.status_code != 0) {
            res.status = FmmmFinalStatus::invalid_input;
            res.first_bad_node = h_state.first_bad_node;
            if (h_state.status_code == 1) res.reason = "extent < 3";
            else if (h_state.status_code == 2) res.reason = "non-finite coordinate";
            else res.reason = "invalid_input";
            return res;
        }

        DeviceBuffer<double> mapped_x(nodes);
        DeviceBuffer<double> mapped_y(nodes);
        const int threads = 256;
        const int blocks = (nodes + threads - 1) / threads;
        fmmm_finalize_affine_continuous_kernel<<<blocks, threads>>>(
            nodes, raw_x, raw_y, static_cast<int>(width), static_cast<int>(height),
            state.get(), mapped_x.get(), mapped_y.get());
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return res;
        }

        ScoreResult score = evaluate_continuous_crossings(
            nodes, edges_count, mapped_x.get(), mapped_y.get(), d_source, d_target,
            d_layer, layers_enabled);
        if (!score.valid) {
            res.status = FmmmFinalStatus::backend_error;
            res.reason = "continuous score failed";
            return res;
        }

        std::vector<double> hx(static_cast<std::size_t>(nodes));
        std::vector<double> hy(static_cast<std::size_t>(nodes));
        mapped_x.download(hx.data(), hx.size());
        mapped_y.download(hy.data(), hy.size());
        res.coordinates.reserve(static_cast<std::size_t>(nodes));
        for (int i = 0; i < nodes; ++i) {
            res.coordinates.emplace_back(hx[static_cast<std::size_t>(i)],
                                         hy[static_cast<std::size_t>(i)]);
        }
        res.score = score;
        res.status = FmmmFinalStatus::ok;
        res.reason = "ok";
    } catch (...) {
        res.status = FmmmFinalStatus::backend_error;
        res.reason = "exception";
    }
    return res;
}

} // namespace lcn::oracle
namespace lcn::release {

FmmmStageResult fmmm_run_single_level(FmmmLevelState& state, const FmmmGraphView& view, const FmmmProfile& profile) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    result.completed_stage = FmmmPipelineStage::none;
    result.reason = "none";

    if (state.node_count < 0 || view.node_count != state.node_count || view.edge_count < 0) {
        result.reason = "invalid counts";
        return result;
    }
    if (state.position_x == nullptr || state.position_y == nullptr) {
        result.reason = "null positions";
        return result;
    }
    // OGDF skips the entire force/post/fine/resize path when n <= 1.
    if (state.node_count <= 1) {
        state.iteration = 0;
        state.phase = 0;
        state.cooling = 1.0;
        result.status = FmmmPipelineStatus::ok;
        result.completed_stage = FmmmPipelineStage::none;
        result.reason = "ok";
        return result;
    }
    if (view.edge_count > 0 &&
        (view.edge_source == nullptr || view.edge_target == nullptr || view.edge_length == nullptr)) {
        result.reason = "null edge arrays";
        return result;
    }
    if (profile.normal_iterations < 0 || profile.postprocess_iterations < 0 ||
        profile.fine_tuning_iterations < 0) {
        result.reason = "negative iterations";
        return result;
    }

    const bool needs_force = profile.normal_iterations > 0 || profile.postprocess_iterations > 0 ||
                             profile.fine_tuning_iterations > 0;
    const bool needs_resize = profile.resize_drawing;
    const bool needs_pack = profile.apply_single_component_pack && state.node_count > 1;
    if (needs_force || needs_resize) {
        if (state.stage_status == nullptr || state.stage_box == nullptr ||
            state.stage_left_x == nullptr || state.stage_left_y == nullptr) {
            result.reason = "null stage staging";
            return result;
        }
    } else if (needs_pack && state.stage_status == nullptr) {
        result.reason = "null stage staging";
        return result;
    }
    if (needs_force) {
        if (state.last_movement_x == nullptr || state.last_movement_y == nullptr ||
            state.attraction_x == nullptr || state.attraction_y == nullptr ||
            state.repulsion_x == nullptr || state.repulsion_y == nullptr) {
            result.reason = "null force scratch";
            return result;
        }
        if (!(profile.average_ideal_length > 0.0)) {
            result.reason = "nonpositive average_ideal_length";
            return result;
        }
    }

    int current_status = 0;
    auto sync_and_check = [&]() -> bool {
        if (cudaGetLastError() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "kernel launch error";
            return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "kernel sync error";
            return false;
        }
        if (cudaMemcpy(&current_status, state.stage_status, sizeof(int), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "status download error";
            return false;
        }
        if (current_status != 0) {
            result.status = FmmmPipelineStatus::backend_error;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "kernel logical error";
            return false;
        }
        return true;
    };

    auto download_box = [&]() -> bool {
        if (cudaMemcpy(&state.box_length, state.stage_box, sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(&state.down_left_x, state.stage_left_x, sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&state.down_left_y, state.stage_left_y, sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "box download error";
            return false;
        }
        return true;
    };

    auto apply_resize = [&](int refresh_box) -> bool {
        lcn::oracle::resize_to_ideal_length_kernel<<<1, 1>>>(
            state.node_count, view.edge_count, view.edge_source, view.edge_target, view.edge_length,
            state.position_x, state.position_y, refresh_box, state.stage_box, state.stage_left_x,
            state.stage_left_y, state.stage_status);
        if (!sync_and_check()) return false;
        if (refresh_box && !download_box()) return false;
        result.completed_stage = FmmmPipelineStage::resize;
        state.phase = 3;
        return true;
    };

    auto run_force = [&](int iterations, int force_phase, FmmmPipelineStage stage) -> bool {
        if (iterations <= 0) return true;
        const bool use_nmm = profile.repulsive_method == FmmmRepulsiveMethod::nmm;
        if (use_nmm) {
            if (state.node_count < 175) {
                result.status = FmmmPipelineStatus::invalid_input;
                result.completed_stage = FmmmPipelineStage::none;
                result.reason = "nmm selected but node_count < 175";
                return false;
            }
            FmmmStageResult nmm_loop = fmmm_run_nmm_force_loop(
                state.node_count, view.edge_count, view.edge_source, view.edge_target,
                view.edge_length, state.position_x, state.position_y, state.box_length,
                state.down_left_x, state.down_left_y, profile.average_ideal_length, iterations,
                state.stage_status, state.stage_box, state.stage_left_x, state.stage_left_y,
                force_phase, state.last_movement_x, state.last_movement_y, state.attraction_x,
                state.attraction_y, state.repulsion_x, state.repulsion_y, profile.seed, state.nmm_workspace);
            if (nmm_loop.status != FmmmPipelineStatus::ok) {
                result = nmm_loop;
                return false;
            }
            if (!download_box()) return false;
        } else {
            lcn::release::FmmmWorkspaceView workspace{};
            workspace.force_x = state.last_movement_x;
            workspace.force_y = state.last_movement_y;
            workspace.mass = state.attraction_x;
            workspace.center_x = state.attraction_y;
            workspace.center_y = state.repulsion_x;
            workspace.scratch_double = state.repulsion_y;
            lcn::oracle::exact_force_loop_kernel<<<1, 1>>>(
                state.node_count, view.edge_count, view.edge_source, view.edge_target,
                view.edge_length, state.position_x, state.position_y, state.box_length,
                state.down_left_x, state.down_left_y, profile.average_ideal_length, iterations,
                state.stage_status, state.stage_box, state.stage_left_x, state.stage_left_y,
                force_phase, workspace);
            if (!sync_and_check()) return false;
            if (!download_box()) return false;
        }
        result.completed_stage = stage;
        state.phase = force_phase;
        state.iteration = iterations;
        // Published cooling under frozen coolTemperature=false: phase0=1.0, post=0.1,
        // fine=0.2 (or 0.02 if iters<=5). cool_temperature=true only updates this
        // host field for phase 0; exact kernel phase-0 cool_factor remains 1.0.
        if (force_phase == 0) {
            state.cooling = profile.cool_temperature ? profile.cool_value : 1.0;
        } else if (force_phase == 1) {
            state.cooling = 0.1;
        } else {
            state.cooling = iterations > 5 ? 0.2 : 0.02;
        }
        return true;
    };

    if (!run_force(profile.normal_iterations, 0, FmmmPipelineStage::normal_force)) return result;
    if (!run_force(profile.postprocess_iterations, 1, FmmmPipelineStage::postprocess)) return result;
    if (needs_resize && !apply_resize(1)) return result;
    if (!run_force(profile.fine_tuning_iterations, 2, FmmmPipelineStage::fine_tuning)) return result;
    if (needs_resize && profile.apply_final_resize && !apply_resize(0)) return result;

    // FMMMLayout::call always pack_subGraph_drawings (even for 1 CC) then
    // AllowedPositions::Integer adjust_positions. Reuse the existing 1-CC
    // rotate/tip/translate/floor kernel so route finals match full-call gold.
    if (profile.apply_single_component_pack) {
        if (state.stage_status == nullptr) {
            result.status = FmmmPipelineStatus::invalid_input;
            result.completed_stage = FmmmPipelineStage::none;
            result.reason = "null stage staging";
            return result;
        }
        lcn::oracle::fmmm_launch_single_component_pack_export(
            state.node_count, state.position_x, state.position_y, state.stage_status);
        if (!sync_and_check()) return result;
        result.completed_stage = FmmmPipelineStage::packing;
        state.phase = 4;
    }

    // Status-gated publish: consumers must ignore device buffers unless status==ok.
    // Mid-schedule backend_error leaves earlier stage mutations on device but sets
    // completed_stage=none and boundary_published=false.
    result.status = FmmmPipelineStatus::ok;
    result.reason = "ok";
    return result;
}

namespace {

constexpr double kFrozenNodeWidth = 20.0;
constexpr double kFrozenNodeHeight = 20.0;
constexpr double kFrozenUnitEdgeLength = 20.0;
constexpr int kFrozenMaxIterFactor = 10;
constexpr int kFrozenPostprocessIters = 10;
constexpr int kFrozenFineTuningIters = 20;
constexpr int kNmmMinNodes = 175;

double frozen_bounding_circle_length() {
    const double half_w = kFrozenNodeWidth * 0.5;
    const double half_h = kFrozenNodeHeight * 0.5;
    const double radius = std::sqrt(half_w * half_w + half_h * half_h);
    return kFrozenUnitEdgeLength + 2.0 * radius;
}

int frozen_get_max_mult_iter(int act_level, int max_level, int node_nr, int fixed_iterations) {
    int iter = 0;
    if (max_level == 0) {
        iter = kFrozenMaxIterFactor * fixed_iterations;
    } else {
        iter = fixed_iterations
            + static_cast<int>((static_cast<double>(act_level) / static_cast<double>(max_level))
                               * static_cast<double>(kFrozenMaxIterFactor - 1)
                               * static_cast<double>(fixed_iterations));
    }
    if (node_nr <= 500 && iter < 100) return 100;
    return iter;
}

struct SimpleEdge {
    int source = 0;
    int target = 0;
    double length = 0.0;
};

void make_simple_loopfree(const ::lcn::oracle::GraphSpec& spec, std::vector<SimpleEdge>& out) {
    out.clear();
    std::set<std::pair<int, int>> seen;
    const double length = frozen_bounding_circle_length();
    for (const auto& edge : spec.edges) {
        int u = edge.source;
        int v = edge.target;
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        if (!seen.insert({u, v}).second) continue;
        out.push_back({edge.source, edge.target, length});
    }
}

void connected_components(int node_count, const std::vector<SimpleEdge>& edges,
                          std::vector<std::vector<int>>& members) {
    members.clear();
    if (node_count <= 0) return;
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(node_count));
    for (const auto& e : edges) {
        if (e.source < 0 || e.source >= node_count || e.target < 0 || e.target >= node_count) continue;
        adj[static_cast<std::size_t>(e.source)].push_back(e.target);
        adj[static_cast<std::size_t>(e.target)].push_back(e.source);
    }
    std::vector<int> comp(static_cast<std::size_t>(node_count), -1);
    for (int v = 0; v < node_count; ++v) {
        if (comp[static_cast<std::size_t>(v)] >= 0) continue;
        const int cid = static_cast<int>(members.size());
        members.emplace_back();
        std::vector<int> stack{v};
        comp[static_cast<std::size_t>(v)] = cid;
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            members[static_cast<std::size_t>(cid)].push_back(u);
            for (int w : adj[static_cast<std::size_t>(u)]) {
                if (comp[static_cast<std::size_t>(w)] >= 0) continue;
                comp[static_cast<std::size_t>(w)] = cid;
                stack.push_back(w);
            }
        }
        std::sort(members[static_cast<std::size_t>(cid)].begin(),
                  members[static_cast<std::size_t>(cid)].end());
    }
}

__global__ void integer_floor_positions_kernel(int node_count, double* x, double* y) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < node_count; ++i) {
        x[i] = floor(x[i]);
        y[i] = floor(y[i]);
    }
}

__global__ void scatter_component_positions_kernel(int local_n, const int* global_ids,
                                                   const double* local_x, const double* local_y,
                                                   double* global_x, double* global_y) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < local_n; ++i) {
        const int g = global_ids[i];
        global_x[g] = local_x[i];
        global_y[g] = local_y[i];
    }
}

FmmmStageResult run_component_g2(int local_n, int edge_count, const int* d_local_source,
                                 const int* d_local_target,
                                 const double* d_local_length, std::uint64_t seed,
                                 int fixed_iterations, bool apply_pack,
                                 double* out_x, double* out_y, void* nmm_workspace) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (local_n <= 0 || out_x == nullptr || out_y == nullptr) {
        result.reason = "g2 component: invalid input";
        return result;
    }
    if (local_n == 1) {
        const double zero = 0.0;
        if (cudaMemcpy(out_x, &zero, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(out_y, &zero, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "g2 component: one-node upload failed";
            return result;
        }
        result.status = FmmmPipelineStatus::ok;
        result.reason = "ok";
        return result;
    }

    const int* d_source_ptr = d_local_source;
    const int* d_target_ptr = d_local_target;
    const double* d_length_ptr = d_local_length;

    ::lcn::oracle::DeviceBuffer<double> d_x(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_y(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_last_x(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_last_y(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_attr_x(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_attr_y(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_rep_x(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<double> d_rep_y(static_cast<std::size_t>(local_n));
    ::lcn::oracle::DeviceBuffer<int> d_stage_status(1);
    ::lcn::oracle::DeviceBuffer<double> d_stage_box(1);
    ::lcn::oracle::DeviceBuffer<double> d_stage_left_x(1);
    ::lcn::oracle::DeviceBuffer<double> d_stage_left_y(1);
    ::lcn::oracle::DeviceBuffer<int> d_place_status(1);
    ::lcn::oracle::DeviceBuffer<double> d_initial_box(1);
    ::lcn::oracle::DeviceBuffer<double> d_dynamic_box(1);
    ::lcn::oracle::DeviceBuffer<double> d_dlx(1);
    ::lcn::oracle::DeviceBuffer<double> d_dly(1);

    launch_full_path_coarsest_placement_kernel(
        local_n, kFrozenNodeWidth, kFrozenNodeHeight, seed, d_place_status.get(),
        d_initial_box.get(), d_dynamic_box.get(), d_dlx.get(), d_dly.get(), d_x.get(), d_y.get());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "g2 component: placement launch failed";
        return result;
    }
    int place_status = 1;
    d_place_status.download(&place_status, 1);
    if (place_status != 0) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "g2 component: placement rejected";
        return result;
    }

    double box = 0.0, dlx = 0.0, dly = 0.0;
    d_dynamic_box.download(&box, 1);
    d_dlx.download(&dlx, 1);
    d_dly.download(&dly, 1);

    double avg = frozen_bounding_circle_length();

    FmmmGraphView view;
    view.edge_source = edge_count > 0 ? d_source_ptr : nullptr;
    view.edge_target = edge_count > 0 ? d_target_ptr : nullptr;
    view.edge_length = edge_count > 0 ? d_length_ptr : nullptr;
    view.node_count = local_n;
    view.edge_count = edge_count;

    FmmmProfile profile;
    profile.normal_iterations = frozen_get_max_mult_iter(0, 0, local_n, fixed_iterations);
    profile.postprocess_iterations = kFrozenPostprocessIters;
    profile.fine_tuning_iterations = kFrozenFineTuningIters;
    profile.average_ideal_length = avg;
    profile.initial_box_length = box;
    profile.initial_down_left_x = dlx;
    profile.initial_down_left_y = dly;
    profile.cool_temperature = false;
    profile.resize_drawing = true;
    profile.apply_final_resize = true;
    profile.apply_single_component_pack = apply_pack;
    profile.seed = seed;
    profile.max_iter_factor = kFrozenMaxIterFactor;
    profile.repulsive_method =
        (local_n >= kNmmMinNodes) ? FmmmRepulsiveMethod::nmm : FmmmRepulsiveMethod::exact;

    FmmmLevelState state;
    state.position_x = d_x.get();
    state.position_y = d_y.get();
    state.force_x = d_last_x.get();
    state.force_y = d_last_y.get();
    state.attraction_x = d_attr_x.get();
    state.attraction_y = d_attr_y.get();
    state.repulsion_x = d_rep_x.get();
    state.repulsion_y = d_rep_y.get();
    state.nmm_workspace = nmm_workspace;
    state.last_movement_x = d_last_x.get();
    state.last_movement_y = d_last_y.get();
    state.stage_status = d_stage_status.get();
    state.stage_box = d_stage_box.get();
    state.stage_left_x = d_stage_left_x.get();
    state.stage_left_y = d_stage_left_y.get();
    state.box_length = box;
    state.down_left_x = dlx;
    state.down_left_y = dly;
    state.resize_scale = 1.0;
    state.cooling = 1.0;
    state.iteration = 0;
    state.phase = 0;
    state.node_count = local_n;

    result = fmmm_run_single_level(state, view, profile);
    if (result.status != FmmmPipelineStatus::ok) return result;

    if (cudaMemcpy(out_x, d_x.get(), static_cast<std::size_t>(local_n) * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess ||
        cudaMemcpy(out_y, d_y.get(), static_cast<std::size_t>(local_n) * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "g2 component: D2D publish failed";
        return result;
    }
    return result;
}

FmmmStageResult run_component_g4(int local_n, int edge_count, const int* d_local_source,
                                 const int* d_local_target,
                                 const double* d_local_length, std::uint64_t seed,
                                 int fixed_iterations, bool apply_pack,
                                 double* out_x, double* out_y, void* nmm_workspace) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (local_n <= 1 || out_x == nullptr || out_y == nullptr) {
        result.reason = "g4 component: invalid input";
        return result;
    }

    const int* d_source_ptr = d_local_source;
    const int* d_target_ptr = d_local_target;
    const double* d_length_ptr = d_local_length;

    FmmmGraphView view;
    view.edge_source = edge_count > 0 ? d_source_ptr : nullptr;
    view.edge_target = edge_count > 0 ? d_target_ptr : nullptr;
    view.edge_length = edge_count > 0 ? d_length_ptr : nullptr;
    view.node_count = local_n;
    view.edge_count = edge_count;

    FmmmProfile profile;
    profile.normal_iterations = fixed_iterations;
    profile.max_iter_factor = kFrozenMaxIterFactor;
    profile.postprocess_iterations = kFrozenPostprocessIters;
    profile.fine_tuning_iterations = kFrozenFineTuningIters;
    profile.resize_drawing = true;
    profile.seed = seed;
    profile.repulsive_method =
        (local_n >= kNmmMinNodes) ? FmmmRepulsiveMethod::nmm : FmmmRepulsiveMethod::exact;

    FmmmHierarchyState hierarchy;
    hierarchy.capacity = 32;
    std::vector<FmmmHierarchyLevel> levels(static_cast<std::size_t>(hierarchy.capacity));
    hierarchy.levels = levels.data();

    auto release_guard = [&hierarchy]() { fmmm_release_hierarchy(hierarchy); };

    result = fmmm_build_hierarchy(hierarchy, view, profile);
    if (result.status != FmmmPipelineStatus::ok) {
        release_guard();
        return result;
    }
    for (int i = 0; i < hierarchy.level_count; ++i) {
        hierarchy.levels[i].state.nmm_workspace = nmm_workspace;
    }
    result = fmmm_execute_hierarchy(hierarchy);
    if (result.status != FmmmPipelineStatus::ok) {
        release_guard();
        return result;
    }
    if (hierarchy.level_count <= 0 || hierarchy.levels[0].state.position_x == nullptr ||
        hierarchy.levels[0].state.position_y == nullptr) {
        release_guard();
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "g4 component: missing finest positions";
        return result;
    }

    if (cudaMemcpy(out_x, hierarchy.levels[0].state.position_x,
                   static_cast<std::size_t>(local_n) * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess ||
        cudaMemcpy(out_y, hierarchy.levels[0].state.position_y,
                   static_cast<std::size_t>(local_n) * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        release_guard();
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "g4 component: D2D publish failed";
        return result;
    }
    release_guard();

    if (apply_pack) {
        ::lcn::oracle::DeviceBuffer<int> pack_status(1);
        ::lcn::oracle::fmmm_launch_single_component_pack_export(local_n, out_x, out_y, pack_status.get());
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "g4 component: pack launch failed";
            return result;
        }
        int status = 1;
        pack_status.download(&status, 1);
        if (status != 0) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "g4 component: pack rejected";
            return result;
        }
        result.completed_stage = FmmmPipelineStage::packing;
    }
    result.status = FmmmPipelineStatus::ok;
    result.reason = "ok";
    return result;
}

} // namespace

__global__ void count_component_edges_kernel(int num_edges, const int* src, const int* comp, int target_comp, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        if (comp[src[i]] == target_comp) {
            atomicAdd(out_count, 1);
        }
    }
}

__global__ void extract_component_edges_kernel(int num_edges, const int* src, const int* tgt, const double* len, const int* comp, const int* local_node_ids, int target_comp, int* out_src, int* out_tgt, double* out_len, int* counter) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int pos = 0;
    for (int i = 0; i < num_edges; ++i) {
        if (comp[src[i]] == target_comp) {
            out_src[pos] = local_node_ids[src[i]];
            out_tgt[pos] = local_node_ids[tgt[i]];
            out_len[pos] = len[i];
            pos++;
        }
    }
    *counter = pos;
}

__global__ void init_pack_sizes_kernel(int node_count, double* widths, double* heights, double w, double h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        widths[i] = w;
        heights[i] = h;
    }
}

// REFERENCE/ORACLE ONLY.  This historical host component coordinator is kept
// in the source for differential archaeology, but is compile-disabled and has
// no production reachability.  The production ABI seam is implemented below
// by the session-owned device schedule dispatcher.
#if 0
FmmmCoordinatorResult fmmm_run_ogdf_coordinator_reference_only(DeviceFmmmSession& session, std::uint64_t seed, int fixed_iterations, bool layers_enabled) {
    FmmmCoordinatorResult out;
    out.stage.status = FmmmPipelineStatus::invalid_input;
    out.stage.reason = "none";
    out.finalization.status = ::lcn::oracle::FmmmFinalStatus::invalid_input;
    out.finalization.reason = "none";

    int node_count = session.raw_nodes_x.valid_elements;
    int component_count = session.component_count;

    if (node_count <= 0) {
        out.stage.reason = "empty graph";
        out.finalization.reason = "empty graph";
        return out;
    }
    if (fixed_iterations < 0) {
        out.stage.reason = "negative fixed_iterations";
        out.finalization.reason = "negative fixed_iterations";
        return out;
    }
    if (component_count <= 0) {
        out.stage.reason = "no components";
        out.finalization.reason = "no components";
        return out;
    }

    ::lcn::oracle::DeviceBuffer<double> raw_x(node_count);
    ::lcn::oracle::DeviceBuffer<double> raw_y(node_count);
    const bool multi_cc = component_count > 1;
    bool used_g4 = false;
    bool used_nmm_profile = false;
    
    std::vector<int> h_comp_offsets(component_count + 1);
    cudaMemcpy(h_comp_offsets.data(), session.comp_offsets.d_ptr, (component_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int c = 0; c < component_count; ++c) {
        int local_n = h_comp_offsets[c+1] - h_comp_offsets[c];
        
        int* d_edge_count; cudaMalloc(&d_edge_count, sizeof(int));
        cudaMemset(d_edge_count, 0, sizeof(int));
        
        int total_norm_edges = session.normalized_edge_count;
        if (total_norm_edges > 0) {
            count_component_edges_kernel<<<(total_norm_edges + 127)/128, 128>>>(total_norm_edges, session.norm_edges_src.d_ptr, session.node_comp.d_ptr, c, d_edge_count);
            cudaDeviceSynchronize();
        }
        
        int local_edge_count = 0;
        cudaMemcpy(&local_edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        ::lcn::oracle::DeviceBuffer<int> d_local_source(local_edge_count > 0 ? local_edge_count : 1);
        ::lcn::oracle::DeviceBuffer<int> d_local_target(local_edge_count > 0 ? local_edge_count : 1);
        ::lcn::oracle::DeviceBuffer<double> d_local_length(local_edge_count > 0 ? local_edge_count : 1);
        
        if (total_norm_edges > 0) {
            cudaMemset(d_edge_count, 0, sizeof(int));
            extract_component_edges_kernel<<<(total_norm_edges + 127)/128, 128>>>(total_norm_edges, session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr, session.norm_edges_len.d_ptr, session.node_comp.d_ptr, session.local_node_ids.d_ptr, c, d_local_source.get(), d_local_target.get(), d_local_length.get(), d_edge_count);
            cudaDeviceSynchronize();
        }
        cudaFree(d_edge_count);
        
        ::lcn::oracle::DeviceBuffer<double> comp_x(local_n);
        ::lcn::oracle::DeviceBuffer<double> comp_y(local_n);
        const bool apply_pack = !multi_cc;
        FmmmStageResult stage;
        
        if (fmmm_is_single_level_route(local_n)) {
            stage = run_component_g2(local_n, local_edge_count, d_local_source.get(), d_local_target.get(), d_local_length.get(), seed, fixed_iterations, apply_pack, comp_x.get(), comp_y.get(), session.nmm_workspace);
        } else {
            used_g4 = true;
            if (local_n >= kNmmMinNodes) used_nmm_profile = true;
            stage = run_component_g4(local_n, local_edge_count, d_local_source.get(), d_local_target.get(), d_local_length.get(), seed, fixed_iterations, apply_pack, comp_x.get(), comp_y.get(), session.nmm_workspace);
        }
        
        if (stage.status != FmmmPipelineStatus::ok) {
            out.stage = stage;
            out.finalization.status = ::lcn::oracle::FmmmFinalStatus::backend_error;
            out.finalization.reason = stage.reason;
            out.route = used_g4 ? (used_nmm_profile ? "g4_g5_nmm" : "g4_hierarchy") : "g2_single_level";
            return out;
        }
        
        scatter_component_positions_kernel<<<1, 1>>>(
            local_n, session.comp_members.d_ptr + h_comp_offsets[c], comp_x.get(), comp_y.get(), raw_x.get(), raw_y.get());
        
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            out.stage.status = FmmmPipelineStatus::backend_error;
            out.stage.reason = "component scatter failed";
            out.finalization.status = ::lcn::oracle::FmmmFinalStatus::backend_error;
            out.finalization.reason = "component scatter failed";
            return out;
        }
    }
    
    if (multi_cc) {
        ::lcn::oracle::DeviceBuffer<double> d_widths(node_count);
        ::lcn::oracle::DeviceBuffer<double> d_heights(node_count);
        init_pack_sizes_kernel<<<(node_count + 127)/128, 128>>>(node_count, d_widths.get(), d_heights.get(), kFrozenNodeWidth, kFrozenNodeHeight);
        cudaDeviceSynchronize();
        
        ::lcn::oracle::DeviceBuffer<double> d_rect_left(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_rect_bottom(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_rect_right(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_rect_top(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_old_dlc_x(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_old_dlc_y(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_new_dlc_x(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_new_dlc_y(component_count);
        ::lcn::oracle::DeviceBuffer<int> d_tipped(component_count);
        ::lcn::oracle::DeviceBuffer<int> d_chosen(component_count);
        ::lcn::oracle::DeviceBuffer<int> d_row(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_angle(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_tx(component_count);
        ::lcn::oracle::DeviceBuffer<double> d_ty(component_count);

        FmmmPackingState pack_state;
        pack_state.component_begin = session.comp_offsets.d_ptr;
        pack_state.component_nodes = session.comp_members.d_ptr;
        pack_state.node_width = d_widths.get();
        pack_state.node_height = d_heights.get();
        pack_state.rect_left = d_rect_left.get();
        pack_state.rect_bottom = d_rect_bottom.get();
        pack_state.rect_right = d_rect_right.get();
        pack_state.rect_top = d_rect_top.get();
        pack_state.old_dlc_x = d_old_dlc_x.get();
        pack_state.old_dlc_y = d_old_dlc_y.get();
        pack_state.new_dlc_x = d_new_dlc_x.get();
        pack_state.new_dlc_y = d_new_dlc_y.get();
        pack_state.tipped = d_tipped.get();
        pack_state.chosen_candidate = d_chosen.get();
        pack_state.row_index = d_row.get();
        pack_state.chosen_angle = d_angle.get();
        pack_state.translate_x = d_tx.get();
        pack_state.translate_y = d_ty.get();
        pack_state.node_count = node_count;
        pack_state.component_count = component_count;
        pack_state.component_node_count = node_count;

        FmmmPackingOptions pack_opts;
        pack_state.position_x = raw_x.get();
        pack_state.position_y = raw_y.get();

        FmmmStageResult pack_res = fmmm_pack_components(pack_state, pack_opts);
        if (pack_res.status != FmmmPipelineStatus::ok) {
            out.stage = pack_res;
            out.finalization.status = ::lcn::oracle::FmmmFinalStatus::backend_error;
            out.finalization.reason = pack_res.reason;
            out.route = used_g4 ? (used_nmm_profile ? "g4_g5_nmm_pack" : "g4_hierarchy_pack") : "g2_single_level_pack";
            return out;
        }

        integer_floor_positions_kernel<<<1, 1>>>(node_count, raw_x.get(), raw_y.get());
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            out.stage.status = FmmmPipelineStatus::backend_error;
            out.stage.reason = "floor kernel failed";
            out.finalization.status = ::lcn::oracle::FmmmFinalStatus::backend_error;
            out.finalization.reason = "floor kernel failed";
            return out;
        }
    }

    out.finalization = ::lcn::oracle::fmmm_finalize_ogdf_profile_raw(
        node_count, session.raw_edges_src.valid_elements, session.raw_edges_src.d_ptr, session.raw_edges_tgt.d_ptr, session.raw_edges_layer.d_ptr,
        raw_x.get(), raw_y.get(), session.max_width, session.max_height, layers_enabled);
    
    if (out.finalization.status == ::lcn::oracle::FmmmFinalStatus::ok) {
        out.stage.status = FmmmPipelineStatus::ok;
        out.stage.reason = "ok";
        if (multi_cc) {
            out.route = used_g4 ? (used_nmm_profile ? "g4_g5_nmm_pack" : "g4_hierarchy_pack") : "g2_single_level_pack";
        } else {
            out.route = used_g4 ? (used_nmm_profile ? "g4_g5_nmm" : "g4_hierarchy") : "g2_single_level";
        }
    }
    return out;
}
#endif

namespace {

::lcn::oracle::FmmmFinalStatus device_publication_status(
    const DeviceFmmmPublication& publication) {
    if (publication.status == "OK") return ::lcn::oracle::FmmmFinalStatus::ok;
    if (publication.status == "INVALID_INPUT") return ::lcn::oracle::FmmmFinalStatus::invalid_input;
    if (publication.status == "REJECTED_BY_ADMISSION") {
        return ::lcn::oracle::FmmmFinalStatus::rejected_by_admission;
    }
    return ::lcn::oracle::FmmmFinalStatus::backend_error;
}

void copy_device_publication_to_legacy_result(
    const DeviceFmmmPublication& publication,
    FmmmCoordinatorResult& out) {
    out.finalization.status = device_publication_status(publication);
    // FmmmFinalResult stores a borrowed reason pointer.  Do not retain the
    // temporary publication string beyond this conversion.
    out.finalization.reason = publication.reason.empty()
        ? "device publication"
        : "device publication reported a session result";
    out.finalization.coordinates.clear();
    out.finalization.coordinates.reserve(publication.coordinates.size());
    for (const auto& point : publication.coordinates) {
        out.finalization.coordinates.emplace_back(
            static_cast<double>(point.first), static_cast<double>(point.second));
    }

    out.finalization.score.edge_crossings = publication.edge_crossings;
    out.finalization.score.K = publication.K;
    out.finalization.score.nK = publication.nK;
    out.finalization.score.Phi = publication.Phi;
    out.finalization.score.C = publication.C;
    out.finalization.score.valid =
        out.finalization.status == ::lcn::oracle::FmmmFinalStatus::ok;
    out.finalization.score.exact_valid = out.finalization.score.valid;
    out.finalization.score.status = out.finalization.score.valid
        ? ::lcn::oracle::ProposalStatus::accepted
        : ::lcn::oracle::ProposalStatus::rejected_invalid;
    out.finalization.score.fallback_required =
        out.finalization.status == ::lcn::oracle::FmmmFinalStatus::backend_error;
    out.finalization.score.fallback_reason = out.finalization.score.fallback_required
        ? "device_schedule_publication_error"
        : "none";
}

} // namespace

FmmmCoordinatorResult fmmm_run_ogdf_coordinator(
    DeviceFmmmSession& session, std::uint64_t seed, int fixed_iterations,
    bool layers_enabled) {
    FmmmCoordinatorResult out;
    out.route = "g0_device_schedule";
    out.stage.status = FmmmPipelineStatus::invalid_input;
    out.stage.completed_stage = FmmmPipelineStage::none;
    out.stage.reason = "device schedule not submitted";
    out.finalization.status = ::lcn::oracle::FmmmFinalStatus::invalid_input;
    out.finalization.reason = "device schedule not submitted";

    // Compile/runtime contract: resident_api.cu has already completed the
    // single upload, normalization, and initial hierarchy calls before this
    // ABI seam.  The session configuration is immutable after create, so the
    // compatibility arguments must agree with it rather than being used to
    // steer a host-side schedule.
    if (session.config.seed != seed || session.config.iterations != fixed_iterations ||
        session.config.layers_enabled != layers_enabled) {
        out.stage.reason = "device schedule configuration mismatch";
        out.finalization.reason = out.stage.reason;
        return out;
    }
    if (!session.hierarchy_built) {
        out.stage.status = FmmmPipelineStatus::backend_error;
        out.stage.reason = "device schedule requires built hierarchy";
        out.finalization.status = ::lcn::oracle::FmmmFinalStatus::backend_error;
        out.finalization.reason = out.stage.reason;
        return out;
    }

    // One call per semantic phase.  These are integration hooks owned by
    // fmmm_session.cu; no host state is inspected between phases and no
    // legacy coordinator/helper is called from the production route.
    const auto submit_phase = [&](const char* phase,
                                  FmmmDeviceStatus (*submit)(DeviceFmmmSession&, cudaStream_t)) {
        const FmmmDeviceStatus status = submit(session, nullptr);
        if (status == FmmmDeviceStatus::ok) return true;
        out.stage.status = status == FmmmDeviceStatus::invalid_input
            ? FmmmPipelineStatus::invalid_input
            : FmmmPipelineStatus::backend_error;
        out.stage.reason = phase;
        out.finalization.status = status == FmmmDeviceStatus::invalid_input
            ? ::lcn::oracle::FmmmFinalStatus::invalid_input
            : ::lcn::oracle::FmmmFinalStatus::backend_error;
        out.finalization.reason = phase;
        return false;
    };

    if (!submit_phase("device NMM phase failed", &fmmm_session_build_nmm) ||
        !submit_phase("device schedule phase failed", &fmmm_session_run_schedule) ||
        !submit_phase("device score/admission phase failed", &fmmm_session_score_and_admit) ||
        !submit_phase("device publication phase failed", &fmmm_session_publish_device)) {
        return out;
    }

    // The session API owns the sole bounded publication download.  Mapping it
    // into the pre-existing coordinator ABI is a compatibility conversion;
    // it does not download positions, tree state, score state, or phase state
    // separately.
    const DeviceFmmmPublication publication = fmmm_session_download_publication(session);
    copy_device_publication_to_legacy_result(publication, out);
    if (out.finalization.status == ::lcn::oracle::FmmmFinalStatus::ok) {
        out.stage.status = FmmmPipelineStatus::ok;
        out.stage.completed_stage = FmmmPipelineStage::packing;
        out.stage.reason = "ok";
    } else if (out.finalization.status == ::lcn::oracle::FmmmFinalStatus::invalid_input) {
        out.stage.status = FmmmPipelineStatus::invalid_input;
        out.stage.reason = out.finalization.reason;
    } else {
        out.stage.status = FmmmPipelineStatus::backend_error;
        out.stage.reason = out.finalization.reason;
    }
    return out;
}

} // namespace lcn::release

namespace lcn::release {

void launch_full_path_solar_hierarchy_kernel(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, std::uint64_t seed, int* status, int* parent,
    int* node_type, int* dedicated_sun, double* sun_distance,
    int* coarse_node_count, int* coarse_edge_count, int* coarse_source,
    int* coarse_target, double* coarse_length,
    std::uint8_t* plan_moon_edge, int* plan_low_adj_begin, int* plan_low_adj_edge,
    int* plan_lambda_begin, double* plan_lambda_value, int* plan_neighbour_begin,
    int* plan_neighbour_node, int* plan_moon_begin, int* plan_moon_node,
    int* plan_high_lower_node, int* plan_high_edge_span, int* plan_high_source,
    int* plan_high_target, std::uint8_t* plan_high_extra, int* plan_high_adj_begin,
    int* plan_high_adj_edge, int* draws_consumed,
    const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream) {
    lcn::oracle::full_path_solar_hierarchy_kernel<<<1, 1, 0, stream>>>(
        node_count, edge_count, edge_source, edge_target, edge_length, seed, status, parent,
        node_type, dedicated_sun, sun_distance, coarse_node_count, coarse_edge_count, coarse_source,
        coarse_target, coarse_length, plan_moon_edge, plan_low_adj_begin, plan_low_adj_edge,
        plan_lambda_begin, plan_lambda_value, plan_neighbour_begin, plan_neighbour_node,
        plan_moon_begin, plan_moon_node, plan_high_lower_node, plan_high_edge_span, plan_high_source,
        plan_high_target, plan_high_extra, plan_high_adj_begin, plan_high_adj_edge, draws_consumed,
        workspace);
}

void launch_full_path_coarsest_placement_kernel(
    int node_count, double node_width, double node_height,
    std::uint64_t seed, int* status, double* initial_box,
    double* dynamic_box, double* down_left_x, double* down_left_y,
    double* x, double* y, cudaStream_t stream) {
    lcn::oracle::full_path_coarsest_placement_kernel<<<1, 1, 0, stream>>>(
        node_count, node_width, node_height, seed, status, initial_box,
        dynamic_box, down_left_x, down_left_y, x, y);
}

void launch_exact_force_loop_kernel(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box, double* out_down_left_x, double* out_down_left_y,
    int force_phase, const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream) {
    lcn::oracle::launch_exact_force_loop_kernel_dispatch(
        node_count, edge_count, edge_source, edge_target, edge_length, x, y,
        box_length, down_left_x, down_left_y, average_ideal_length, iterations, status,
        out_box, out_down_left_x, out_down_left_y, force_phase, workspace, stream);
}

namespace {

void free_nmm_package_tree(FmmmNmmState& state) {
    auto free_d = [](double*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    auto free_i = [](int*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    free_d(state.cell_downleft_x);
    free_d(state.cell_downleft_y);
    free_d(state.cell_box_length);
    free_i(state.cell_level);
    free_i(state.cell_parent);
    free_i(state.cell_child4);
    free_i(state.cell_particle_count);
    free_d(state.cell_center_x);
    free_d(state.cell_center_y);
    free_i(state.contained_begin);
    free_i(state.contained_nodes);
    free_i(state.I_begin);
    free_i(state.I_nodes);
    free_i(state.D1_begin);
    free_i(state.D1_nodes);
    free_i(state.D2_begin);
    free_i(state.D2_nodes);
    free_i(state.M_begin);
    free_i(state.M_nodes);
    free_d(state.multipole_re);
    free_d(state.multipole_im);
    free_d(state.local_re);
    free_d(state.local_im);
    state.cell_count = 0;
    state.contained_node_count = 0;
    state.I_count = 0;
    state.D1_count = 0;
    state.D2_count = 0;
    state.M_count = 0;
    state.precision = 0;
    state.expansion_stride = 0;
}

} // namespace

FmmmStageResult fmmm_run_nmm_force_loop(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, double* x, double* y, double box_length, double down_left_x,
    double down_left_y, double average_ideal_length, int iterations, int* status, double* out_box,
    double* out_down_left_x, double* out_down_left_y, int force_phase, double* last_movement_x,
    double* last_movement_y, double* attraction_x, double* attraction_y, double* repulsion_x,
    double* repulsion_y, std::uint64_t seed, void* persistent_workspace) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    result.completed_stage = FmmmPipelineStage::none;
    result.reason = "none";

    if (node_count < 175 || edge_count < 0 || iterations <= 0 || iterations > 10000 ||
        force_phase < 0 || force_phase > 2 || !(average_ideal_length > 0.0) || x == nullptr ||
        y == nullptr || status == nullptr || out_box == nullptr || out_down_left_x == nullptr ||
        out_down_left_y == nullptr || last_movement_x == nullptr || last_movement_y == nullptr ||
        attraction_x == nullptr || attraction_y == nullptr || repulsion_x == nullptr ||
        repulsion_y == nullptr) {
        result.reason = "nmm force loop: invalid input";
        return result;
    }
    if (edge_count > 0 &&
        (edge_source == nullptr || edge_target == nullptr || edge_length == nullptr)) {
        result.reason = "nmm force loop: null edge arrays";
        return result;
    }

    cudaMemset(last_movement_x, 0, static_cast<std::size_t>(node_count) * sizeof(double));
    cudaMemset(last_movement_y, 0, static_cast<std::size_t>(node_count) * sizeof(double));

    double current_box = box_length;
    double current_left_x = down_left_x;
    double current_left_y = down_left_y;
    const int block_threads = 256;
    const int grid_blocks = std::max(1, (node_count + block_threads - 1) / block_threads);

    for (int iteration = 1; iteration <= iterations; ++iteration) {
        lcn::oracle::nmm_force_prep_kernel<<<grid_blocks, block_threads>>>(
            node_count, edge_count, edge_source, edge_target, edge_length, x, y, current_box,
            current_left_x, current_left_y, attraction_x, attraction_y, repulsion_x, repulsion_y,
            out_box, out_down_left_x, out_down_left_y, status);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm prep kernel error";
            return result;
        }
        lcn::oracle::nmm_force_prep_box_reduce_kernel<<<1, block_threads>>>(
            node_count, current_box, current_left_x, current_left_y,
            repulsion_x, repulsion_y, out_box, out_down_left_x, out_down_left_y);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm prep box reduction error";
            return result;
        }
        int st = 1;
        if (cudaMemcpy(&st, status, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || st != 0) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm prep logical error";
            return result;
        }
        if (cudaMemcpy(&current_box, out_box, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&current_left_x, out_down_left_x, sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(&current_left_y, out_down_left_y, sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm prep box download error";
            return result;
        }

        FmmmNmmState nmm_state{};
        nmm_state.position_x = x;
        nmm_state.position_y = y;
        nmm_state.force_x = repulsion_x;
        nmm_state.force_y = repulsion_y;
        nmm_state.edge_source = edge_source;
        nmm_state.edge_target = edge_target;
        nmm_state.edge_length = edge_length;
        nmm_state.node_count = node_count;
        nmm_state.edge_count = edge_count;
        nmm_state.cell_count = 0;
        nmm_state.graph_epoch = seed;
        nmm_state.level_epoch = seed;
        nmm_state.particle_order_epoch = seed;
        nmm_state.tree_epoch = seed;
        nmm_state.map_epoch = seed;
        const FmmmStageResult bind_res =
                fmmm_nmm_bind_workspace(nmm_state, persistent_workspace);
        if (bind_res.status != FmmmPipelineStatus::ok) {
            result = bind_res;
            return result;
        }
        FmmmNmmOptions nmm_opts{};
        nmm_opts.box_length = current_box;
        nmm_opts.down_left_x = current_left_x;
        nmm_opts.down_left_y = current_left_y;
        nmm_opts.precision = 4;
        nmm_opts.particles_in_leaf = 25;
        nmm_opts.tree_construction = FmmmNmmTreeConstruction::subtree_by_subtree;
        nmm_opts.small_cell = FmmmNmmSmallCell::iteratively;
        nmm_opts.using_nmm = true;
        nmm_opts.seed = seed;

        FmmmStageResult nmm_res = fmmm_nmm_force(nmm_state, nmm_opts);
        // free_nmm_package_tree(nmm_state);
        if (nmm_res.status != FmmmPipelineStatus::ok) {
            result = nmm_res;
            if (result.reason == nullptr || result.reason[0] == '\0') {
                result.reason = "nmm force failed";
            }
            return result;
        }

        lcn::oracle::nmm_force_apply_kernel<<<grid_blocks, block_threads>>>(
            node_count, x, y, current_box, current_left_x, current_left_y, average_ideal_length,
            iteration, iterations, force_phase, attraction_x, attraction_y, repulsion_x, repulsion_y,
            last_movement_x, last_movement_y, out_box, out_down_left_x, out_down_left_y,
            attraction_x, attraction_y, status);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm apply kernel error";
            return result;
        }
        lcn::oracle::nmm_force_apply_box_reduce_kernel<<<1, block_threads>>>(
            node_count, attraction_x, attraction_y, out_box, out_down_left_x, out_down_left_y);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm apply box reduction error";
            return result;
        }
        if (cudaMemcpy(&st, status, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || st != 0) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm apply logical error";
            return result;
        }
        if (cudaMemcpy(&current_box, out_box, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&current_left_x, out_down_left_x, sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(&current_left_y, out_down_left_y, sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "nmm apply box download error";
            return result;
        }
    }

    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "ok";
    return result;
}

} // namespace lcn::release

