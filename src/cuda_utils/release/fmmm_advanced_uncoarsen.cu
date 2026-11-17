#include "fmmm_workspace.h"
#include "fmmm_advanced_uncoarsen.h"

#include "../oracle/device_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace lcn::oracle {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kLargestFiniteDouble = 1.7976931348623157e308;
constexpr int kStatusOk = static_cast<int>(FmmmAdvancedUncoarsenStatus::Ok);
constexpr int kStatusUnsupported = static_cast<int>(FmmmAdvancedUncoarsenStatus::Unsupported);
constexpr int kStatusInvalidPlan = static_cast<int>(FmmmAdvancedUncoarsenStatus::InvalidPlan);

// This is the same intentionally local MSVC mt19937/uniform_int_distribution
// representation used by fmmm_level_engine.cu. Keeping it in this owner
// prevents a host RNG call from changing the continuation boundary.
struct OgdfMt19937 {
    static constexpr int kWords = 624;
    std::uint32_t state[kWords];
    int index = kWords;

    __device__ void seed(std::uint32_t value) {
        state[0] = value;
        for (int i = 1; i < kWords; ++i) {
            state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30U))
                    + static_cast<std::uint32_t>(i);
        }
        index = kWords;
    }
    __device__ void twist() {
        for (int i = 0; i < kWords; ++i) {
            const std::uint32_t bits = (state[i] & 0x80000000U)
                    | (state[(i + 1) % kWords] & 0x7fffffffU);
            state[i] = state[(i + 397) % kWords] ^ (bits >> 1U)
                    ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
        }
        index = 0;
    }
    __device__ std::uint32_t next() {
        if (index == kWords) twist();
        std::uint32_t value = state[index++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        return value ^ (value >> 18U);
    }
};

// Implements frozen MSVC OGDF's randomNumber(1, BILLION) before its
// lower-bound shift. The distribution has BILLION (= 1,000,000,000) values
// 0..BILLION-1; callers apply the +1 seen in Multilevel.cpp. Its rejection
// boundary matters: in Auto2 the first rejected raw word occurs while
// placing lower node 7, so both its range and threshold are semantic.
__device__ std::uint32_t random_billion(OgdfMt19937& random, int& draws) {
    ++draws;
    constexpr std::uint64_t range = 1000000000ULL;
    constexpr std::uint32_t threshold = 294967296U;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * range;
        if (static_cast<std::uint32_t>(product) >= threshold) {
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }
}

// FMMMLayout::create_initial_placement_random uses randomNumber(0, BILLION)
// (BILLION+1 outcomes). Skip those draws with this distribution so Advanced
// uncoarsen continues from the same mt19937 state OGDF has after coarsest
// placement. Mixing the (1, BILLION) Multilevel path here desyncs three-level
// Auto2 L1/L0 waggle.
__device__ void skip_initial_placement_draw(OgdfMt19937& random) {
    constexpr std::uint64_t range = 1000000001ULL;
    constexpr std::uint32_t threshold = 294967292U;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * range;
        if (static_cast<std::uint32_t>(product) >= threshold) {
            return;
        }
    }
}

__device__ bool valid_node(int node, int count) { return node >= 0 && node < count; }

__device__ double point_angle(double px, double py, double qx, double qy,
                              double rx, double ry) {
    const double dx1 = qx - px, dy1 = qy - py;
    const double dx2 = rx - px, dy2 = ry - py;
    if ((dx1 == 0.0 && dy1 == 0.0) || (dx2 == 0.0 && dy2 == 0.0)) return 0.0;
    double result = atan2(dy2, dx2) - atan2(dy1, dx1);
    if (result < 0.0) result += kTwoPi;
    return result;
}

__device__ void random_position(OgdfMt19937& random, int& draws, double center_x, double center_y,
                                double radius, double angle_1, double angle_2,
                                double& out_x, double& out_y) {
    const double random_unit = static_cast<double>(random_billion(random, draws) + 1U) / 1000000002.0;
    const double angle = angle_1 + (angle_2 - angle_1) * random_unit;
    out_x = center_x + cos(angle) * radius;
    out_y = center_y + sin(angle) * radius;
}

__device__ void waggled_inbetween_position(OgdfMt19937& random, int& draws, double sx, double sy,
                                            double tx, double ty, double lambda,
                                            double& out_x, double& out_y) {
    const double inbetween_x = sx + lambda * (tx - sx);
    const double inbetween_y = sy + lambda * (ty - sy);
    const double radius = 0.05 * sqrt((tx - sx) * (tx - sx) + (ty - sy) * (ty - sy));
    const double random_unit = static_cast<double>(random_billion(random, draws) + 1U) / 1000000002.0;
    random_position(random, draws, inbetween_x, inbetween_y, radius * random_unit, 0.0, kTwoPi,
                    out_x, out_y);
}

__device__ void calculated_position(OgdfMt19937& random, int& draws, double px, double py, double qx,
                                     double qy, double distance_p, double distance_q,
                                     double& out_x, double& out_y) {
    const double distance_pq = sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
    // The oracle profile never supplies coincident positions here. Do not
    // substitute an epsilon: OGDF divides directly, so reject instead.
    if (distance_pq == 0.0) { out_x = NAN; out_y = NAN; return; }
    const double lambda = (distance_p + (distance_pq - distance_p - distance_q) / 2.0)
            / distance_pq;
    waggled_inbetween_position(random, draws, px, py, qx, qy, lambda, out_x, out_y);
}

__device__ bool append_point(double* xs, double* ys, int& count, double x, double y) {
    if (count >= kFmmmAdvancedUncoarsenMaxAuxEntries) return false;
    xs[count] = x;
    ys[count] = y;
    ++count;
    return true;
}

__device__ bool place_barycenter(double* xs, double* ys, int count, double& out_x,
                                 double& out_y) {
    if (count == 0) return false;
    double sum_x = 0.0, sum_y = 0.0;
    for (int i = 0; i < count; ++i) { sum_x += xs[i]; sum_y += ys[i]; }
    out_x = sum_x / static_cast<double>(count);
    out_y = sum_y / static_cast<double>(count);
    return true;
}

// These two kernels cover only state initialization and read-only contract
// validation.  RNG selection, sector order, and all position writes remain in
// advanced_uncoarsen_kernel below, preserving the OGDF draw/order contract.
__global__ void uncoarsen_init_kernel(FmmmAdvancedUncoarsenStateView state,
                                       int node_count, int initial_draws) {
    if (state.status == nullptr) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *state.status = kStatusOk;
        if (state.consumed_draws != nullptr) *state.consumed_draws = initial_draws;
    }
    (void)node_count; // output arrays are shadow-owned by the transaction
}

__global__ void uncoarsen_validate_kernel(FmmmAdvancedUncoarsenPlanView plan,
                                           FmmmAdvancedUncoarsenStateView state) {
    if (state.status == nullptr) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (plan.lower_node_count <= 0 || plan.lower_node_count > kFmmmAdvancedUncoarsenMaxNodes
                || plan.high_node_count <= 0 || plan.high_node_count > kFmmmAdvancedUncoarsenMaxNodes
                || plan.lower_edge_ordinal_span < 0 || plan.lower_edge_ordinal_span > kFmmmAdvancedUncoarsenMaxEdges
                || plan.high_edge_ordinal_span < 0 || plan.high_edge_ordinal_span > kFmmmAdvancedUncoarsenMaxEdges
                || plan.lower_type == nullptr || plan.lower_dedicated_sun == nullptr
                || plan.lower_dedicated_sun_distance == nullptr || plan.high_lower_node == nullptr
                || plan.high_position_x == nullptr || plan.high_position_y == nullptr
                || plan.lower_edge_source == nullptr || plan.lower_edge_target == nullptr
                || plan.lower_edge_length == nullptr || plan.lower_edge_is_moon == nullptr
                || plan.lower_adjacency_begin == nullptr || plan.lower_adjacency_edge == nullptr
                || plan.high_edge_source == nullptr || plan.high_edge_target == nullptr
                || plan.high_edge_is_extra == nullptr || plan.high_adjacency_begin == nullptr
                || plan.high_adjacency_edge == nullptr || plan.lambda_begin == nullptr
                || plan.neighbour_sun_begin == nullptr || plan.moon_begin == nullptr
                || state.position_x == nullptr || state.position_y == nullptr
                || state.placed == nullptr || state.angle_1 == nullptr || state.angle_2 == nullptr
                || state.box_length == nullptr || state.down_left_x == nullptr || state.down_left_y == nullptr) {
            atomicExch(state.status, kStatusInvalidPlan);
        }
    }
    __syncthreads();
    if (*state.status != kStatusOk) return;

    const int lower_adjacency_count = plan.lower_adjacency_begin[plan.lower_node_count];
    const int high_adjacency_count = plan.high_adjacency_begin[plan.high_node_count];
    const int lambda_count = plan.lambda_begin[plan.lower_node_count];
    const int neighbour_count = plan.neighbour_sun_begin[plan.lower_node_count];
    const int moon_count = plan.moon_begin[plan.lower_node_count];
    if (lower_adjacency_count < 0 || lower_adjacency_count > 2 * plan.lower_edge_ordinal_span
            || high_adjacency_count < 0 || high_adjacency_count > 2 * plan.high_edge_ordinal_span
            || lambda_count < 0 || lambda_count > kFmmmAdvancedUncoarsenMaxAuxEntries
            || neighbour_count < 0 || neighbour_count > kFmmmAdvancedUncoarsenMaxAuxEntries
            || moon_count < 0 || moon_count > kFmmmAdvancedUncoarsenMaxAuxEntries
            || (lambda_count > 0 && plan.lambda_values == nullptr)
            || (neighbour_count > 0 && plan.neighbour_sun_nodes == nullptr)
            || (moon_count > 0 && plan.moon_nodes == nullptr)) {
        atomicExch(state.status, kStatusUnsupported);
    }
    __syncthreads();
    if (*state.status != kStatusOk) return;

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int node = tid; node < plan.lower_node_count; node += stride) {
        if (!valid_node(plan.lower_dedicated_sun[node], plan.lower_node_count)
                || plan.lower_adjacency_begin[node] < 0
                || plan.lower_adjacency_begin[node] > plan.lower_adjacency_begin[node + 1]
                || plan.lambda_begin[node] < 0
                || plan.lambda_begin[node] > plan.lambda_begin[node + 1]
                || plan.neighbour_sun_begin[node] < 0
                || plan.neighbour_sun_begin[node] > plan.neighbour_sun_begin[node + 1]
                || plan.moon_begin[node] < 0
                || plan.moon_begin[node] > plan.moon_begin[node + 1]) {
            atomicExch(state.status, kStatusInvalidPlan);
        }
        const int low_first = plan.lower_adjacency_begin[node];
        const int low_last = plan.lower_adjacency_begin[node + 1];
        for (int at = low_first; at < low_last; ++at) {
            const int edge = plan.lower_adjacency_edge[at];
            if (edge < 0 || edge >= plan.lower_edge_ordinal_span) {
                atomicExch(state.status, kStatusInvalidPlan);
            } else if (!valid_node(plan.lower_edge_source[edge], plan.lower_node_count)
                       || !valid_node(plan.lower_edge_target[edge], plan.lower_node_count)) {
                atomicExch(state.status, kStatusInvalidPlan);
            }
        }
        for (int at = plan.lambda_begin[node]; at < plan.lambda_begin[node + 1]; ++at) {
            if (!isfinite(plan.lambda_values[at])) atomicExch(state.status, kStatusInvalidPlan);
        }
        for (int at = plan.neighbour_sun_begin[node]; at < plan.neighbour_sun_begin[node + 1]; ++at) {
            if (!valid_node(plan.neighbour_sun_nodes[at], plan.lower_node_count)) {
                atomicExch(state.status, kStatusInvalidPlan);
            }
        }
        for (int at = plan.moon_begin[node]; at < plan.moon_begin[node + 1]; ++at) {
            if (!valid_node(plan.moon_nodes[at], plan.lower_node_count)) {
                atomicExch(state.status, kStatusInvalidPlan);
            }
        }
    }
    for (int high = tid; high < plan.high_node_count; high += stride) {
        if (!valid_node(plan.high_lower_node[high], plan.lower_node_count)
                || plan.high_adjacency_begin[high] < 0
                || plan.high_adjacency_begin[high] > plan.high_adjacency_begin[high + 1]) {
            atomicExch(state.status, kStatusInvalidPlan);
        }
        const int first = plan.high_adjacency_begin[high];
        const int last = plan.high_adjacency_begin[high + 1];
        for (int at = first; at < last; ++at) {
            const int edge = plan.high_adjacency_edge[at];
            if (edge < 0 || edge >= plan.high_edge_ordinal_span) {
                atomicExch(state.status, kStatusInvalidPlan);
            } else if (!valid_node(plan.high_edge_source[edge], plan.high_node_count)
                       || !valid_node(plan.high_edge_target[edge], plan.high_node_count)) {
                atomicExch(state.status, kStatusInvalidPlan);
            }
        }
        for (int other = high + 1; other < plan.high_node_count; ++other) {
            if (plan.high_lower_node[high] == plan.high_lower_node[other]) {
                atomicExch(state.status, kStatusInvalidPlan);
            }
        }
    }
    // Every lower membership must be represented by exactly one high sun.
    for (int node = tid; node < plan.lower_node_count; node += stride) {
        const int sun = plan.lower_dedicated_sun[node];
        bool represented = false;
        for (int high = 0; high < plan.high_node_count; ++high) {
            if (plan.high_lower_node[high] == sun) { represented = true; break; }
        }
        if (!represented) atomicExch(state.status, kStatusInvalidPlan);
    }
}

__global__ void uncoarsen_commit_kernel(FmmmAdvancedUncoarsenStateView shadow,
                                         FmmmAdvancedUncoarsenStateView state,
                                         int node_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || state.status == nullptr) return;
    if (shadow.status == nullptr || *shadow.status != kStatusOk) {
        if (shadow.status != nullptr) *state.status = *shadow.status;
        return;
    }
    for (int node = 0; node < node_count; ++node) {
        state.position_x[node] = shadow.position_x[node];
        state.position_y[node] = shadow.position_y[node];
        state.placed[node] = shadow.placed[node];
        state.angle_1[node] = shadow.angle_1[node];
        state.angle_2[node] = shadow.angle_2[node];
    }
    *state.box_length = *shadow.box_length;
    *state.down_left_x = *shadow.down_left_x;
    *state.down_left_y = *shadow.down_left_y;
    if (state.consumed_draws != nullptr && shadow.consumed_draws != nullptr)
        *state.consumed_draws = *shadow.consumed_draws;
    *state.status = kStatusOk;
}

FmmmAdvancedUncoarsenStateView make_shadow_state(
        FmmmAdvancedUncoarsenStateView state,
        const lcn::release::FmmmWorkspaceView& workspace,
        int node_count) {
    FmmmAdvancedUncoarsenStateView shadow = state;
    shadow.position_x = workspace.position_x;
    shadow.position_y = workspace.position_y;
    shadow.angle_1 = workspace.center_x;
    shadow.angle_2 = workspace.center_y;
    // The tail is reserved for byte-sized placed flags; the head is used by
    // the ordered kernel for its int PM-node list.  max_aux_entries is sized
    // to make these ranges disjoint.
    shadow.placed = reinterpret_cast<std::uint8_t*>(workspace.scratch_int)
        + workspace.max_aux_entries * sizeof(int) - node_count;
    shadow.box_length = workspace.sun_distance;
    shadow.down_left_x = workspace.sun_distance + 1;
    shadow.down_left_y = workspace.sun_distance + 2;
    shadow.consumed_draws = workspace.scratch_int;
    return shadow;
}

__global__ void uncoarsen_mark_invalid_kernel(int* status) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && status != nullptr)
        *status = kStatusUnsupported;
}

__global__ void advanced_uncoarsen_kernel(FmmmAdvancedUncoarsenPlanView plan,
                                           FmmmAdvancedUncoarsenStateView state,
                                           FmmmAdvancedUncoarsenPrng prng,
                                           lcn::release::FmmmWorkspaceView workspace) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (state.status == nullptr) return;
    if (*state.status != kStatusOk) return;
    if (plan.lower_node_count <= 0 || plan.lower_node_count > kFmmmAdvancedUncoarsenMaxNodes
            || plan.high_node_count <= 0 || plan.high_node_count > kFmmmAdvancedUncoarsenMaxNodes
            || plan.lower_edge_ordinal_span < 0 || plan.lower_edge_ordinal_span > kFmmmAdvancedUncoarsenMaxEdges
            || plan.high_edge_ordinal_span < 0 || plan.high_edge_ordinal_span > kFmmmAdvancedUncoarsenMaxEdges
            || prng.consumed_draws < 0
            || plan.lower_type == nullptr || plan.lower_dedicated_sun == nullptr
            || plan.lower_dedicated_sun_distance == nullptr || plan.high_lower_node == nullptr
            || plan.high_position_x == nullptr || plan.high_position_y == nullptr
            || plan.lower_edge_source == nullptr || plan.lower_edge_target == nullptr
            || plan.lower_edge_length == nullptr || plan.lower_edge_is_moon == nullptr
            || plan.lower_adjacency_begin == nullptr || plan.lower_adjacency_edge == nullptr
            || plan.high_edge_source == nullptr || plan.high_edge_target == nullptr
            || plan.high_edge_is_extra == nullptr || plan.high_adjacency_begin == nullptr
            || plan.high_adjacency_edge == nullptr || plan.lambda_begin == nullptr
            || plan.neighbour_sun_begin == nullptr || plan.moon_begin == nullptr
            || state.position_x == nullptr || state.position_y == nullptr || state.placed == nullptr
            || state.angle_1 == nullptr || state.angle_2 == nullptr || state.box_length == nullptr
            || state.down_left_x == nullptr || state.down_left_y == nullptr) {
        *state.status = kStatusInvalidPlan;
        return;
    }

    const int lower_adjacency_count = plan.lower_adjacency_begin[plan.lower_node_count];
    const int high_adjacency_count = plan.high_adjacency_begin[plan.high_node_count];
    const int lambda_count = plan.lambda_begin[plan.lower_node_count];
    const int neighbour_count = plan.neighbour_sun_begin[plan.lower_node_count];
    const int moon_count = plan.moon_begin[plan.lower_node_count];
    if (lower_adjacency_count < 0 || lower_adjacency_count > 2 * kFmmmAdvancedUncoarsenMaxEdges
            || high_adjacency_count < 0 || high_adjacency_count > 2 * kFmmmAdvancedUncoarsenMaxEdges
            || lambda_count < 0 || lambda_count > kFmmmAdvancedUncoarsenMaxAuxEntries
            || neighbour_count < 0 || neighbour_count > kFmmmAdvancedUncoarsenMaxAuxEntries
            || moon_count < 0 || moon_count > kFmmmAdvancedUncoarsenMaxAuxEntries) {
        *state.status = kStatusUnsupported;
        return;
    }
    if ((lambda_count > 0 && plan.lambda_values == nullptr)
            || (neighbour_count > 0 && plan.neighbour_sun_nodes == nullptr)
            || (moon_count > 0 && plan.moon_nodes == nullptr)) {
        *state.status = kStatusInvalidPlan;
        return;
    }

    for (int node = 0; node < plan.lower_node_count; ++node) {
        if (plan.lower_adjacency_begin[node] < 0
                || plan.lower_adjacency_begin[node] > plan.lower_adjacency_begin[node + 1]
                || plan.lambda_begin[node] < 0 || plan.lambda_begin[node] > plan.lambda_begin[node + 1]
                || plan.neighbour_sun_begin[node] < 0
                || plan.neighbour_sun_begin[node] > plan.neighbour_sun_begin[node + 1]
                || plan.moon_begin[node] < 0 || plan.moon_begin[node] > plan.moon_begin[node + 1]
                || !valid_node(plan.lower_dedicated_sun[node], plan.lower_node_count)) {
            *state.status = kStatusInvalidPlan;
            return;
        }
        state.placed[node] = 0;
    }
    for (int node = 0; node < plan.high_node_count; ++node) {
        if (plan.high_adjacency_begin[node] < 0
                || plan.high_adjacency_begin[node] > plan.high_adjacency_begin[node + 1]
                || !valid_node(plan.high_lower_node[node], plan.lower_node_count)) {
            *state.status = kStatusInvalidPlan;
            return;
        }
    }

    OgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(prng.seed));
    int draws = prng.consumed_draws;
    // Replay RNG in OGDF order: coarsest create_initial_placement_random uses
    // randomNumber(0, BILLION); Multilevel waggle uses randomNumber(1, BILLION).
    const int placement_prefix = (prng.initial_placement_draws < 0)
            ? prng.consumed_draws
            : prng.initial_placement_draws;
    for (int draw = 0; draw < prng.consumed_draws; ++draw) {
        if (draw < placement_prefix) {
            skip_initial_placement_draw(random);
        } else {
            int dummy = 0;
            (void)random_billion(random, dummy);
        }
    }

    // set_initial_positions_of_sun_nodes(), in graph node order at level k+1.
    for (int high = 0; high < plan.high_node_count; ++high) {
        const int sun = plan.high_lower_node[high];
        state.position_x[sun] = plan.high_position_x[high];
        state.position_y[sun] = plan.high_position_y[high];
        state.placed[sun] = 1;
    }

    // create_all_placement_sectors(). The high adjacency flattening is part
    // of the oracle because OGDF's adjEntry iteration order is semantic.
    double* adjacent_x = workspace.force_x;
    double* adjacent_y = workspace.force_y;
    for (int high = 0; high < plan.high_node_count; ++high) {
        const double px = plan.high_position_x[high], py = plan.high_position_y[high];
        int adjacent_count = 0;
        for (int at = plan.high_adjacency_begin[high]; at < plan.high_adjacency_begin[high + 1]; ++at) {
            const int edge = plan.high_adjacency_edge[at];
            if (edge < 0 || edge >= plan.high_edge_ordinal_span) { *state.status = kStatusInvalidPlan; return; }
            if (plan.high_edge_is_extra[edge] != 0) continue;
            const int source = plan.high_edge_source[edge], target = plan.high_edge_target[edge];
            if (!valid_node(source, plan.high_node_count) || !valid_node(target, plan.high_node_count)) {
                *state.status = kStatusInvalidPlan; return;
            }
            const int other = source == high ? target : (target == high ? source : -1);
            if (other < 0 || adjacent_count == kFmmmAdvancedUncoarsenMaxNodes) {
                *state.status = kStatusInvalidPlan; return;
            }
            adjacent_x[adjacent_count] = plan.high_position_x[other];
            adjacent_y[adjacent_count] = plan.high_position_y[other];
            ++adjacent_count;
        }
        double angle_1 = 0.0, angle_2 = 0.0;
        if (adjacent_count == 0) {
            angle_2 = kTwoPi;
        } else if (adjacent_count == 1) {
            angle_1 = point_angle(px, py, px + 1.0, py, adjacent_x[0], adjacent_y[0]);
            angle_2 = angle_1 + kPi;
        } else {
            // Multilevel::create_all_placement_sectors usual case:
            // do { ...; if (it.valid() && it.succ().valid()) it = cyclicSucc(it); steps++; }
            // while (steps <= MAX && it.valid() && it.succ().valid());
            // ListIterator::succ() is non-cyclic, so the last adjacency is never a
            // sector start and the walk never wraps — unlike step % n for MAX steps.
            const int kMaxSectorSteps = 10;
            int steps = 1;
            int it = 0;
            do {
                const double candidate_1 = point_angle(px, py, px + 1.0, py,
                                                       adjacent_x[it], adjacent_y[it]);
                double minimum_next = kLargestFiniteDouble;
                for (int next = 0; next < adjacent_count; ++next) {
                    if (adjacent_x[it] == adjacent_x[next] && adjacent_y[it] == adjacent_y[next]) continue;
                    const double gap = point_angle(px, py, adjacent_x[it], adjacent_y[it],
                                                   adjacent_x[next], adjacent_y[next]);
                    if (gap < minimum_next) minimum_next = gap;
                }
                if (minimum_next == kLargestFiniteDouble) { *state.status = kStatusInvalidPlan; return; }
                if (it == 0 || minimum_next > angle_2 - angle_1) {
                    angle_1 = candidate_1;
                    angle_2 = candidate_1 + minimum_next;
                }
                if (it + 1 < adjacent_count) {
                    ++it;
                }
                ++steps;
            } while (steps <= kMaxSectorSteps && it + 1 < adjacent_count);
            if (angle_1 == angle_2) angle_2 = angle_1 + kPi;
        }
        const int sun = plan.high_lower_node[high];
        state.angle_1[sun] = angle_1;
        state.angle_2[sun] = angle_2;
    }
    for (int node = 0; node < plan.lower_node_count; ++node) {
        const int sun = plan.lower_dedicated_sun[node];
        state.angle_1[node] = state.angle_1[sun];
        state.angle_2[node] = state.angle_2[sun];
    }

    // set_initial_positions_of_planet_and_moon_nodes(). Type 3 PM nodes are
    // deliberately deferred into the ordered pm_nodes list.
    int* pm_nodes = workspace.scratch_int;
    int pm_count = 0;
    double* contribution_x = workspace.scratch_double;
    double* contribution_y = workspace.mass;
    for (int node = 0; node < plan.lower_node_count; ++node) {
        const int type = plan.lower_type[node];
        if (type == 3) {
            if (pm_count == kFmmmAdvancedUncoarsenMaxNodes) { *state.status = kStatusUnsupported; return; }
            pm_nodes[pm_count++] = node;
            continue;
        }
        if (type != 2 && type != 4) continue;
        const int sun = plan.lower_dedicated_sun[node];
        const double sun_x = state.position_x[sun], sun_y = state.position_y[sun];
        const double sun_distance = plan.lower_dedicated_sun_distance[node];
        int contribution_count = 0;
        for (int at = plan.lower_adjacency_begin[node]; at < plan.lower_adjacency_begin[node + 1]; ++at) {
            const int edge = plan.lower_adjacency_edge[at];
            if (edge < 0 || edge >= plan.lower_edge_ordinal_span) { *state.status = kStatusInvalidPlan; return; }
            const int source = plan.lower_edge_source[edge], target = plan.lower_edge_target[edge];
            const int adjacent = source == node ? target : (target == node ? source : -1);
            if (!valid_node(adjacent, plan.lower_node_count)) { *state.status = kStatusInvalidPlan; return; }
            if (plan.lower_dedicated_sun[adjacent] == sun && plan.lower_type[adjacent] != 1
                    && state.placed[adjacent] != 0) {
                double x, y;
                calculated_position(random, draws, sun_x, sun_y, state.position_x[adjacent],
                                    state.position_y[adjacent], sun_distance,
                                    plan.lower_edge_length[edge], x, y);
                if (!isfinite(x) || !append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                    *state.status = kStatusUnsupported; return;
                }
            }
        }
        const int lambda_first = plan.lambda_begin[node], lambda_last = plan.lambda_begin[node + 1];
        if (lambda_first == lambda_last) {
            if (contribution_count == 0) {
                double x, y;
                random_position(random, draws, sun_x, sun_y, sun_distance, state.angle_1[node], state.angle_2[node], x, y);
                if (!append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                    *state.status = kStatusUnsupported; return;
                }
            }
        } else {
            const int neighbours_first = plan.neighbour_sun_begin[node];
            const int neighbours_last = plan.neighbour_sun_begin[node + 1];
            for (int pos = neighbours_first, lambda_pos = lambda_first; pos < neighbours_last; ++pos) {
                const int adjacent_sun = plan.neighbour_sun_nodes[pos];
                if (!valid_node(adjacent_sun, plan.lower_node_count)) { *state.status = kStatusInvalidPlan; return; }
                double x, y;
                waggled_inbetween_position(random, draws, sun_x, sun_y, state.position_x[adjacent_sun],
                                            state.position_y[adjacent_sun], plan.lambda_values[lambda_pos], x, y);
                if (!append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                    *state.status = kStatusUnsupported; return;
                }
                if (lambda_pos != lambda_last - 1) ++lambda_pos;
            }
        }
        if (!place_barycenter(contribution_x, contribution_y, contribution_count,
                              state.position_x[node], state.position_y[node])) {
            *state.status = kStatusInvalidPlan; return;
        }
        state.placed[node] = 1;
    }

    // set_initial_positions_of_pm_nodes(), preserving pm_nodes and every
    // adjacency/list order captured by the oracle.
    for (int pm_index = 0; pm_index < pm_count; ++pm_index) {
        const int node = pm_nodes[pm_index];
        const int sun = plan.lower_dedicated_sun[node];
        const double sun_x = state.position_x[sun], sun_y = state.position_y[sun];
        const double sun_distance = plan.lower_dedicated_sun_distance[node];
        int contribution_count = 0;
        for (int at = plan.lower_adjacency_begin[node]; at < plan.lower_adjacency_begin[node + 1]; ++at) {
            const int edge = plan.lower_adjacency_edge[at];
            if (edge < 0 || edge >= plan.lower_edge_ordinal_span) { *state.status = kStatusInvalidPlan; return; }
            const int source = plan.lower_edge_source[edge], target = plan.lower_edge_target[edge];
            const int adjacent = source == node ? target : (target == node ? source : -1);
            if (!valid_node(adjacent, plan.lower_node_count)) { *state.status = kStatusInvalidPlan; return; }
            if (plan.lower_edge_is_moon[edge] == 0 && plan.lower_dedicated_sun[adjacent] == sun
                    && plan.lower_type[adjacent] != 1 && state.placed[adjacent] != 0) {
                double x, y;
                calculated_position(random, draws, sun_x, sun_y, state.position_x[adjacent],
                                    state.position_y[adjacent], sun_distance,
                                    plan.lower_edge_length[edge], x, y);
                if (!isfinite(x) || !append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                    *state.status = kStatusUnsupported; return;
                }
            }
        }
        for (int pos = plan.moon_begin[node]; pos < plan.moon_begin[node + 1]; ++pos) {
            const int moon = plan.moon_nodes[pos];
            if (!valid_node(moon, plan.lower_node_count)) { *state.status = kStatusInvalidPlan; return; }
            const double moon_distance = plan.lower_dedicated_sun_distance[moon];
            if (moon_distance == 0.0) { *state.status = kStatusInvalidPlan; return; }
            double x, y;
            waggled_inbetween_position(random, draws, sun_x, sun_y, state.position_x[moon], state.position_y[moon],
                                        sun_distance / moon_distance, x, y);
            if (!append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                *state.status = kStatusUnsupported; return;
            }
        }
        const int lambda_first = plan.lambda_begin[node], lambda_last = plan.lambda_begin[node + 1];
        if (lambda_first != lambda_last) {
            const int neighbours_first = plan.neighbour_sun_begin[node];
            const int neighbours_last = plan.neighbour_sun_begin[node + 1];
            for (int pos = neighbours_first, lambda_pos = lambda_first; pos < neighbours_last; ++pos) {
                const int adjacent_sun = plan.neighbour_sun_nodes[pos];
                if (!valid_node(adjacent_sun, plan.lower_node_count)) { *state.status = kStatusInvalidPlan; return; }
                double x, y;
                waggled_inbetween_position(random, draws, sun_x, sun_y, state.position_x[adjacent_sun],
                                            state.position_y[adjacent_sun], plan.lambda_values[lambda_pos], x, y);
                if (!append_point(contribution_x, contribution_y, contribution_count, x, y)) {
                    *state.status = kStatusUnsupported; return;
                }
                if (lambda_pos != lambda_last - 1) ++lambda_pos;
            }
        }
        if (!place_barycenter(contribution_x, contribution_y, contribution_count,
                              state.position_x[node], state.position_y[node])) {
            *state.status = kStatusInvalidPlan; return;
        }
        state.placed[node] = 1;
    }

    double xmin = state.position_x[0], xmax = state.position_x[0];
    double ymin = state.position_y[0], ymax = state.position_y[0];
    for (int node = 0; node < plan.lower_node_count; ++node) {
        if (state.placed[node] == 0) { *state.status = kStatusInvalidPlan; return; }
        xmin = fmin(xmin, state.position_x[node]); xmax = fmax(xmax, state.position_x[node]);
        ymin = fmin(ymin, state.position_y[node]); ymax = fmax(ymax, state.position_y[node]);
    }
    *state.down_left_x = floor(xmin - 1.0);
    *state.down_left_y = floor(ymin - 1.0);
    *state.box_length = ceil(fmax(ymax - ymin, xmax - xmin) * 1.01 + 2.0);
    if (*state.box_length <= 2.0) {
        *state.box_length = plan.lower_node_count * 20.0;
        *state.down_left_x = floor(xmin) - *state.box_length / 2.0;
        *state.down_left_y = floor(ymin) - *state.box_length / 2.0;
    }
    *state.status = kStatusOk;
    if (state.consumed_draws != nullptr) *state.consumed_draws = draws;
}

} // namespace

void fmmm_advanced_uncoarsen_reference(FmmmAdvancedUncoarsenPlanView plan,
                                       FmmmAdvancedUncoarsenStateView state,
                                       FmmmAdvancedUncoarsenPrng prng) {
    lcn::release::FmmmWorkspaceShape shape; shape.node_count = plan.lower_node_count; shape.edge_count = plan.lower_edge_ordinal_span; shape.level_count = 1; shape.cell_capacity = plan.lower_node_count; shape.max_level_nodes = plan.lower_node_count; shape.max_level_edges = plan.lower_edge_ordinal_span; shape.max_aux_entries = std::max<std::size_t>(kFmmmAdvancedUncoarsenMaxAuxEntries, static_cast<std::size_t>(std::max(plan.lower_node_count, 1)));
    lcn::release::FmmmWorkspace workspace; if (lcn::release::FmmmWorkspace::create(shape, workspace) != lcn::release::FmmmPipelineStatus::ok) { if (state.status) *state.status = 1; return; }
    const auto view = workspace.view();
    const auto shadow = make_shadow_state(state, view, plan.lower_node_count);
    uncoarsen_init_kernel<<<1, 128>>>(shadow, plan.lower_node_count, prng.consumed_draws);
    uncoarsen_validate_kernel<<<1, 128>>>(plan, shadow);
    advanced_uncoarsen_kernel<<<1, 1>>>(plan, shadow, prng, view);
    uncoarsen_commit_kernel<<<1, 1>>>(shadow, state, plan.lower_node_count);
    cuda_check(cudaGetLastError(), "CUDA FMMM Advanced uncoarsen");
}

void launch_advanced_uncoarsen_guards(FmmmAdvancedUncoarsenPlanView plan,
                                      FmmmAdvancedUncoarsenStateView state,
                                      FmmmAdvancedUncoarsenPrng prng,
                                      cudaStream_t stream) {
    uncoarsen_init_kernel<<<1, 128, 0, stream>>>(state, plan.lower_node_count, prng.consumed_draws);
    uncoarsen_validate_kernel<<<1, 128, 0, stream>>>(plan, state);
}

void launch_advanced_uncoarsen_transaction(
        FmmmAdvancedUncoarsenPlanView plan,
        FmmmAdvancedUncoarsenStateView state,
        FmmmAdvancedUncoarsenPrng prng,
        const lcn::release::FmmmWorkspaceView& workspace,
        cudaStream_t stream) {
    const std::size_t n = static_cast<std::size_t>(std::max(plan.lower_node_count, 0));
    const std::size_t required_ints = n + (n + 3u) / 4u;
    if (workspace.position_x == nullptr || workspace.position_y == nullptr
            || workspace.center_x == nullptr || workspace.center_y == nullptr
            || workspace.scratch_int == nullptr || workspace.sun_distance == nullptr
            || workspace.max_aux_entries < required_ints) {
        uncoarsen_mark_invalid_kernel<<<1, 1, 0, stream>>>(state.status);
        return;
    }
    const auto shadow = make_shadow_state(state, workspace, plan.lower_node_count);
    uncoarsen_init_kernel<<<1, 128, 0, stream>>>(shadow, plan.lower_node_count, prng.consumed_draws);
    uncoarsen_validate_kernel<<<1, 128, 0, stream>>>(plan, shadow);
    advanced_uncoarsen_kernel<<<1, 1, 0, stream>>>(plan, shadow, prng, workspace);
    uncoarsen_commit_kernel<<<1, 1, 0, stream>>>(shadow, state, plan.lower_node_count);
}

} // namespace lcn::oracle

namespace lcn::release {
void launch_advanced_uncoarsen_kernel(
    lcn::oracle::FmmmAdvancedUncoarsenPlanView plan,
    lcn::oracle::FmmmAdvancedUncoarsenStateView state,
    lcn::oracle::FmmmAdvancedUncoarsenPrng prng,
    const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream) {
    lcn::oracle::launch_advanced_uncoarsen_transaction(plan, state, prng, workspace, stream);
}
} // namespace lcn::release

