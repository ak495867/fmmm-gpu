#include "../oracle/full_square.h"
#include "fmmm_device_state.h"
#include "fmmm_advanced_uncoarsen.h"
#include "fmmm_level_engine.h"
#include "fmmm_workspace.h"
#include "fmmm_hierarchy_contract.h"
#include "fmmm_session.h"
#include "fmmm_nmm_packing.h"

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace lcn::oracle {
namespace {

// Host graph parser for Python input tuples.
// Throws standard pybind exceptions if the tuples are malformed or coordinate
// values are non-integral for discrete oracle validation.
GraphSpec parse_graph(const py::iterable& nodes, const py::iterable& edges, std::int64_t width, std::int64_t height) {
    GraphSpec spec; spec.max_width = width; spec.max_height = height;
    for (const py::handle value : nodes) {
        const auto p = py::reinterpret_borrow<py::sequence>(value);
        if (p.size() != 2) throw py::value_error("each node must be an (x, y) pair");
        spec.nodes.push_back({py::cast<std::int64_t>(p[0]), py::cast<std::int64_t>(p[1])});
    }
    for (const py::handle value : edges) {
        const auto e = py::reinterpret_borrow<py::sequence>(value);
        if (e.size() != 3) throw py::value_error("each edge must be a (source, target, layer) triple");
        spec.edges.push_back({py::cast<std::int32_t>(e[0]), py::cast<std::int32_t>(e[1]), py::cast<std::int32_t>(e[2])});
    }
    return spec;
}

py::dict score_dict(const ScoreResult& value) {
    py::dict out;
    out["valid"] = value.valid; out["exact_valid"] = value.exact_valid;
    out["K"] = value.K; out["nK"] = value.nK; out["Phi"] = value.Phi; out["C"] = value.C;
    out["edge_crossings"] = value.edge_crossings;
    out["proposal_status"] = status_name(value.status);
    out["fallback_required"] = value.fallback_required; out["fallback_reason"] = value.fallback_reason;
    return out;
}

py::dict gpu_identity() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "read CUDA device");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "read CUDA device properties");
    py::dict out;
    out["device"] = device; out["name"] = std::string(prop.name);
    out["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    out["total_global_memory"] = static_cast<std::uint64_t>(prop.totalGlobalMem);
    return out;
}

__device__ inline long long fmmm_candidate_x(DeviceGraphView g, int node, int moved, long long x) {
    return node == moved ? x : g.x[node];
}
__device__ inline long long fmmm_candidate_y(DeviceGraphView g, int node, int moved, long long y) {
    return node == moved ? y : g.y[node];
}

// This is the admission predicate needed while generating a FMMM layout, not
// a host-side score. It deliberately matches the geometry rejection portion
// of FULL_SQUARE_GPU so force moves cannot leave projection without a valid
// starting state.
__device__ bool fmmm_candidate_is_valid(DeviceGraphView g, int moved, long long x, long long y) {
    for (int i = 0; i < g.node_count; ++i) {
        const long long ix = fmmm_candidate_x(g, i, moved, x);
        const long long iy = fmmm_candidate_y(g, i, moved, y);
        if (ix <= 0 || ix >= g.max_width || iy <= 0 || iy >= g.max_height) return false;
        for (int j = 0; j < i; ++j) {
            if (ix == fmmm_candidate_x(g, j, moved, x) &&
                iy == fmmm_candidate_y(g, j, moved, y)) return false;
        }
    }
    for (int node = 0; node < g.node_count; ++node) {
        for (int edge = 0; edge < g.edge_count; ++edge) {
            const int a = g.source[edge], b = g.target[edge];
            if (node == a || node == b) continue;
            if (on_r(fmmm_candidate_x(g, node, moved, x), fmmm_candidate_y(g, node, moved, y),
                     fmmm_candidate_x(g, a, moved, x), fmmm_candidate_y(g, a, moved, y),
                     fmmm_candidate_x(g, b, moved, x), fmmm_candidate_y(g, b, moved, y))) return false;
        }
    }
    return true;
}

// This is the standard MT19937 transition and tempering rule used by OGDF's
// std::mt19937. Keeping it local to the intentionally serial Phase-1 kernel
// makes every draw order explicit and avoids a host PRNG dependency.
struct FmmmOgdfMt19937 {
    static constexpr int kStateWords = 624;
    std::uint32_t state[kStateWords];
    int index = kStateWords;

    __device__ void seed(std::uint32_t value) {
        state[0] = value;
        for (int i = 1; i < kStateWords; ++i) {
            state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30U)) + static_cast<std::uint32_t>(i);
        }
        index = kStateWords;
    }

    __device__ void twist() {
        for (int i = 0; i < kStateWords; ++i) {
            const std::uint32_t bits = (state[i] & 0x80000000U) | (state[(i + 1) % kStateWords] & 0x7fffffffU);
            state[i] = state[(i + 397) % kStateWords] ^ (bits >> 1U) ^ ((bits & 1U) == 0U ? 0U : 0x9908b0dfU);
        }
        index = 0;
    }

    __device__ std::uint32_t next() {
        if (index == kStateWords) twist();
        std::uint32_t value = state[index++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        return value ^ (value >> 18U);
    }
};

// MSVC's uniform_int_distribution uses multiply-high with rejection. The
// threshold is 2^32 mod 1,000,000,001 and makes this match the runner's
// std::uniform_int_distribution(0, 1'000'000'000) draws exactly.
__device__ std::uint32_t fmmm_ogdf_random_billion(FmmmOgdfMt19937& random) {
    constexpr std::uint64_t kRange = 1000000001ULL;
    constexpr std::uint32_t kRejectionThreshold = 294967292U;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * kRange;
        if (static_cast<std::uint32_t>(product) >= kRejectionThreshold) {
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }
}

// This is the same VS 2022 inclusive integer distribution rule as the
// coarsest-placement helper above, generalized for Set::randomNumber(0,last).
// The rejection is normally rare for the small hierarchy witness, but keeping
// it makes the device draw stream independent of an implementation accident.
__device__ std::uint32_t fmmm_ogdf_random_inclusive(FmmmOgdfMt19937& random,
                                                     std::uint32_t upper) {
    const std::uint32_t range = upper + 1U;
    const std::uint32_t rejection_threshold = static_cast<std::uint32_t>(-range) % range;
    while (true) {
        const std::uint64_t product = static_cast<std::uint64_t>(random.next()) * range;
        if (static_cast<std::uint32_t>(product) >= rejection_threshold) {
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }
}

// OGDF's profile selects RandomRandIterNr: FMMM resets its mt19937 to the
// frozen seed and places every coarsest node from two draws in [0, 1e9]. The
// CUDA graph owns integer coordinates, so the frozen half-up normalization
// projects that continuous box into the strict device interior. Caller
// coordinates are deliberately not a layout hint in this profile.
__global__ void cuda_fmmm_initial_placement(DeviceGraphView g, unsigned long long seed) {
    if (blockIdx.x || threadIdx.x) return;
    if (g.node_count == 0) return;
    constexpr std::uint64_t kBillion = 1000000000ULL;
    FmmmOgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(seed));
    for (int v = 0; v < g.node_count; ++v) {
        const std::uint64_t raw_x = fmmm_ogdf_random_billion(random);
        const std::uint64_t raw_y = fmmm_ogdf_random_billion(random);
        g.x[v] = 1 + static_cast<std::int64_t>((raw_x * static_cast<std::uint64_t>(g.max_width - 2) + kBillion / 2U) / kBillion);
        g.y[v] = 1 + static_cast<std::int64_t>((raw_y * static_cast<std::uint64_t>(g.max_height - 2) + kBillion / 2U) / kBillion);
    }
}
__global__ void fmmm_copy_positions(DeviceGraphView graph, float* x, float* y) { const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node<graph.node_count){x[node]=static_cast<float>(graph.x[node]);y[node]=static_cast<float>(graph.y[node]);} }
__global__ void fmmm_apply_bh_force(DeviceGraphView graph,const float* fx,const float* fy) { if(blockIdx.x||threadIdx.x)return;for(int node=0;node<graph.node_count;++node){const auto nx=graph.x[node]+static_cast<std::int64_t>((fx[node]>0.0F)-(fx[node]<0.0F));const auto ny=graph.y[node]+static_cast<std::int64_t>((fy[node]>0.0F)-(fy[node]<0.0F));if(fmmm_candidate_is_valid(graph,node,nx,ny)){graph.x[node]=nx;graph.y[node]=ny;}} }
__global__ void fmmm_apply_continuous_force(float* x, float* y, const float* rep_x, const float* rep_y,
                                            const float* attr_x, const float* attr_y, int node_count,
                                            int phase, int iteration, float box_length) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count) return;
    const float spring = phase == 2 ? 2.0F : 1.0F;
    const float repulsion = phase == 2 ? fminf(0.2F, 400.0F / static_cast<float>(node_count)) : 1.0F;
    const float cooling = phase == 1 ? 0.1F : (phase == 2 ? (iteration <= 15 ? 0.2F : 0.02F) : 1.0F);
    float fx = 400.0F * (spring * attr_x[node] + repulsion * rep_x[node]);
    float fy = 400.0F * (spring * attr_y[node] + repulsion * rep_y[node]);
    const float norm = sqrtf(fx * fx + fy * fy);
    if (norm == 0.0F) return;
    const float radius = iteration == 1 ? box_length / 1000.0F : box_length / 5.0F;
    const float scale = fminf(norm * cooling * 0.05F, radius) / norm;
    x[node] += scale * fx;
    y[node] += scale * fy;
}
__global__ void fmmm_quantize_continuous_positions(DeviceGraphView graph, const float* x, const float* y) {
    if (blockIdx.x || threadIdx.x || graph.node_count == 0) return;
    float min_x=x[0], max_x=x[0], min_y=y[0], max_y=y[0];
    for (int i=1;i<graph.node_count;++i) { min_x=fminf(min_x,x[i]); max_x=fmaxf(max_x,x[i]); min_y=fminf(min_y,y[i]); max_y=fmaxf(max_y,y[i]); }
    for (int i=0;i<graph.node_count;++i) {
        graph.x[i] = max_x == min_x ? (graph.max_width-1)/2 : static_cast<std::int64_t>(floorf(1.0F+(x[i]-min_x)*static_cast<float>(graph.max_width-2)/(max_x-min_x)+0.5F));
        graph.y[i] = max_y == min_y ? (graph.max_height-1)/2 : static_cast<std::int64_t>(floorf(1.0F+(y[i]-min_y)*static_cast<float>(graph.max_height-2)/(max_y-min_y)+0.5F));
    }
}

// The hierarchy witness deliberately uses a single device thread. This is a
// correctness reference for OGDF's mutable Set, not a parallel coarsener. It
// owns all selection, membership, parent, and coarse-edge state on the GPU;
// host code only reads its bounded diagnostic result after completion.
constexpr int kFmmmSolarSystemMaxNodes = 64;
constexpr int kFmmmSolarSystemMaxEdges = 256;
constexpr int kFmmmSolarSystemOk = 0;
constexpr int kFmmmSolarSystemCapacity = 1;
constexpr int kFmmmSolarSystemNoMoonNeighbour = 2;

__global__ void fmmm_build_solar_system_reference(DeviceGraphView graph, unsigned long long seed,
                                                   int* status, int* parent, int* coarse_node_count,
                                                   int* coarse_edge_count, int* coarse_source,
                                                   int* coarse_target, float* coarse_length,
                                                   int* output_type, int* output_sun,
                                                   float* output_sun_distance) {
    if (blockIdx.x || threadIdx.x) return;
    if (graph.node_count < 0 || graph.node_count > kFmmmSolarSystemMaxNodes ||
        graph.edge_count < 0) {
        *status = kFmmmSolarSystemCapacity;
        return;
    }

    const int node_count = graph.node_count;
    const int edge_count = graph.edge_count;
    int selectable[kFmmmSolarSystemMaxNodes];
    int selectable_position[kFmmmSolarSystemMaxNodes];
    int star_mass[kFmmmSolarSystemMaxNodes];
    int type[kFmmmSolarSystemMaxNodes];
    int dedicated_sun[kFmmmSolarSystemMaxNodes];
    int sun_to_parent[kFmmmSolarSystemMaxNodes];
    float sun_distance[kFmmmSolarSystemMaxNodes];

    for (int node = 0; node < node_count; ++node) {
        selectable[node] = node;
        selectable_position[node] = node;
        star_mass[node] = 1;
        type[node] = 0;
        dedicated_sun[node] = -1;
        sun_to_parent[node] = -1;
        sun_distance[node] = 0.0F;
        parent[node] = -1;
    }
    // At level zero every node mass is one, so OGDF's star mass is one plus
    // the degree in stable normalized edge order.
    for (int edge = 0; edge < edge_count; ++edge) {
        ++star_mass[graph.source[edge]];
        ++star_mass[graph.target[edge]];
    }

    FmmmOgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(seed));
    int last_selectable = node_count - 1;
    int next_parent = 0;
    while (last_selectable >= 0) {
        int last_trial = last_selectable;
        int selected_index = -1;
        int selected_mass = 0;
        for (int trial = 0; trial < 20 && last_trial >= 0; ++trial) {
            const int random_index = static_cast<int>(fmmm_ogdf_random_inclusive(
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
        const int sun = selectable[selected_index];
        const int tail = selectable[last_selectable];
        selectable[last_selectable] = sun;
        selectable[selected_index] = tail;
        selectable_position[sun] = last_selectable;
        selectable_position[tail] = selected_index;
        --last_selectable;

        type[sun] = 1;
        dedicated_sun[sun] = sun;
        sun_distance[sun] = 0.0F;
        sun_to_parent[sun] = next_parent;
        parent[sun] = next_parent++;

        // OGDF iterates a node's adjacency entries. With this profile graph
        // construction appends entries in normalized edge order, which this
        // scan reproduces without constructing a host CSR representation.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = graph.source[edge];
            const int target = graph.target[edge];
            if (source != sun && target != sun) continue;
            const int planet = source == sun ? target : source;
            type[planet] = 2;
            dedicated_sun[planet] = sun;
            sun_distance[planet] = 20.0F;
            parent[planet] = sun_to_parent[sun];
        }
        // Deletions in OGDF's Set are swap removals. The membership test is
        // the inverse position against the live tail, not the node type.
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = graph.source[edge];
            const int target = graph.target[edge];
            if (source != sun && target != sun) continue;
            const int planet = source == sun ? target : source;
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
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = graph.source[edge];
            const int target = graph.target[edge];
            if (source != sun && target != sun) continue;
            const int planet = source == sun ? target : source;
            for (int moon_edge = 0; moon_edge < edge_count; ++moon_edge) {
                const int moon_source = graph.source[moon_edge];
                const int moon_target = graph.target[moon_edge];
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

    // This is OGDF's second pass: ascending node order and strict comparison
    // preserve the first adjacency entry on equal length ties.
    for (int moon = 0; moon < node_count; ++moon) {
        if (type[moon] != 0) continue;
        int nearest = -1;
        float nearest_distance = 0.0F;
        for (int edge = 0; edge < edge_count; ++edge) {
            const int source = graph.source[edge];
            const int target = graph.target[edge];
            if (source != moon && target != moon) continue;
            const int neighbour = source == moon ? target : source;
            if (type[neighbour] != 2 && type[neighbour] != 3) continue;
            constexpr float edge_length = 20.0F;
            if (nearest == -1 || nearest_distance > edge_length) {
                nearest = neighbour;
                nearest_distance = edge_length;
            }
        }
        if (nearest == -1) {
            *status = kFmmmSolarSystemNoMoonNeighbour;
            return;
        }
        type[moon] = 4;
        dedicated_sun[moon] = dedicated_sun[nearest];
        sun_distance[moon] = nearest_distance + sun_distance[nearest];
        parent[moon] = sun_to_parent[dedicated_sun[moon]];
        type[nearest] = 3;
    }

    for (int node = 0; node < node_count; ++node) {
        output_type[node] = type[node];
        output_sun[node] = dedicated_sun[node];
        output_sun_distance[node] = sun_distance[node];
    }

    int output_edges = 0;
    for (int edge = 0; edge < edge_count; ++edge) {
        const int first_parent = parent[graph.source[edge]];
        const int second_parent = parent[graph.target[edge]];
        if (first_parent == second_parent) continue;
        const int source = first_parent < second_parent ? first_parent : second_parent;
        const int target = first_parent < second_parent ? second_parent : first_parent;
        const float length = sun_distance[graph.source[edge]] + 20.0F + sun_distance[graph.target[edge]];
        int prior = -1;
        for (int prior_edge = 0; prior_edge < output_edges; ++prior_edge) {
            if (coarse_source[prior_edge] == source && coarse_target[prior_edge] == target) {
                prior = prior_edge;
                break;
            }
        }
        if (prior == -1) {
            coarse_source[output_edges] = source;
            coarse_target[output_edges] = target;
            coarse_length[output_edges] = length;
            ++output_edges;
        }
    }
    // Although Multilevel::delete_parallel_edges_and_update_edgelength()
    // contains an averaging loop, it calls makeSimpleUndirected() first.
    // That routine already removes all but the first parallel edge, so the
    // observable pinned-OGDF behavior is first-created-edge retention.
    // The witness is canonical rather than dependent on OGDF's internal edge
    // container order. Keep the device result canonical too.
    for (int i = 1; i < output_edges; ++i) {
        const int source = coarse_source[i];
        const int target = coarse_target[i];
        const float length = coarse_length[i];
        int j = i - 1;
        while (j >= 0 && (coarse_source[j] > source ||
                          (coarse_source[j] == source && coarse_target[j] > target))) {
            coarse_source[j + 1] = coarse_source[j];
            coarse_target[j + 1] = coarse_target[j];
            coarse_length[j + 1] = coarse_length[j];
            --j;
        }
        coarse_source[j + 1] = source;
        coarse_target[j + 1] = target;
        coarse_length[j + 1] = length;
    }
    *coarse_node_count = next_parent;
    *coarse_edge_count = output_edges;
    *status = kFmmmSolarSystemOk;
}

// FMMM resets the same PRNG to randSeed() after hierarchy construction. The
// coarsest placement therefore has its own reproducible draw stream rather
// than continuing the draws used for solar-system sun selection.
__global__ void fmmm_coarsest_random_placement_reference(int node_count, unsigned long long seed,
                                                          double* box_length, double* x, double* y) {
    if (blockIdx.x || threadIdx.x) return;
    const double box = ceil(static_cast<double>(node_count * 10) * 1.1);
    *box_length = box;
    FmmmOgdfMt19937 random;
    random.seed(static_cast<std::uint32_t>(seed));
    constexpr double kBillion = 1000000000.0;
    for (int node = 0; node < node_count; ++node) {
        x[node] = static_cast<double>(fmmm_ogdf_random_billion(random)) / kBillion * (box - 2.0) + 1.0;
        y[node] = static_cast<double>(fmmm_ogdf_random_billion(random)) / kBillion * (box - 2.0) + 1.0;
    }
}
// Pair coarsening makes the multilevel boundary explicit.  No host
// coordinates intervene between the initial seed and exact GPU admission.
__global__ void fmmm_pair_coarsen(DeviceGraphView graph, float* x, float* y, int coarse_count) { const int coarse=blockIdx.x*blockDim.x+threadIdx.x;if(coarse>=coarse_count)return;const int a=2*coarse,b=a+1;const float bx=b<graph.node_count?static_cast<float>(graph.x[b]):static_cast<float>(graph.x[a]);const float by=b<graph.node_count?static_cast<float>(graph.y[b]):static_cast<float>(graph.y[a]);x[coarse]=(static_cast<float>(graph.x[a])+bx)*0.5F;y[coarse]=(static_cast<float>(graph.y[a])+by)*0.5F; }
__global__ void fmmm_apply_coarse_force(DeviceGraphView graph,const float* fx,const float* fy) { if(blockIdx.x||threadIdx.x)return;for(int node=0;node<graph.node_count;++node){const int coarse=node/2;const auto nx=graph.x[node]+static_cast<std::int64_t>((fx[coarse]>0.0F)-(fx[coarse]<0.0F));const auto ny=graph.y[node]+static_cast<std::int64_t>((fy[coarse]>0.0F)-(fy[coarse]<0.0F));if(fmmm_candidate_is_valid(graph,node,nx,ny)){graph.x[node]=nx;graph.y[node]=ny;}} }

void fmmm_bh_force(float* position_x, float* position_y, float* force_x, float* force_y, int node_count, const std::int32_t* edge_source=nullptr, const std::int32_t* edge_target=nullptr, int edge_count=0, float* attraction_x=nullptr, float* attraction_y=nullptr) {
    const int capacity=node_count*4+1;
    DeviceBuffer<float> mass(capacity),center_x(capacity),center_y(capacity),width(capacity);
    DeviceBuffer<std::int32_t> child(static_cast<size_t>(capacity)*4),ready(capacity),next_node(1);
    lcn::release::FmmmDeviceStateView state{position_x,position_y,force_x,force_y,attraction_x,attraction_y,edge_source,edge_target,child.get(),mass.get(),center_x.get(),center_y.get(),width.get(),ready.get(),next_node.get(),node_count,edge_count,capacity};
    lcn::release::fmmm_bh_initialize(state);
    lcn::release::fmmm_bh_build_tree(state);
    lcn::release::fmmm_bh_compute_repulsion(state,{1.0F,1.0F,64,1});
    cuda_check(cudaGetLastError(),"CUDA Barnes-Hut force");
}

struct FmmmSolarSystemEdge {
    int source = 0;
    int target = 0;
    float length = 0.0F;
};

struct FmmmSolarSystemDiagnostic {
    int status = kFmmmSolarSystemCapacity;
    int coarse_node_count = 0;
    std::vector<int> parent;
    std::vector<int> node_type;
    std::vector<int> dedicated_sun;
    std::vector<float> sun_distance;
    double coarsest_box_length = 0.0;
    std::vector<std::pair<double, double>> coarsest_initial_placement;
    std::vector<FmmmSolarSystemEdge> edges;
};

FmmmSolarSystemDiagnostic fmmm_solar_system_diagnostic(DeviceGraph& graph, std::uint64_t seed) {
    FmmmSolarSystemDiagnostic result;
    const int node_count = graph.node_count();
    DeviceBuffer<int> status(1), parent(static_cast<size_t>(node_count)), coarse_node_count(1),
        coarse_edge_count(1), coarse_source(static_cast<size_t>(graph.edge_count())),
        coarse_target(static_cast<size_t>(graph.edge_count())), node_type(static_cast<size_t>(node_count)),
        dedicated_sun(static_cast<size_t>(node_count));
    DeviceBuffer<float> coarse_length(static_cast<size_t>(graph.edge_count())), sun_distance(static_cast<size_t>(node_count));
    fmmm_build_solar_system_reference<<<1, 1>>>(
        graph.view(), seed, status.get(), parent.get(), coarse_node_count.get(), coarse_edge_count.get(),
        coarse_source.get(), coarse_target.get(), coarse_length.get(), node_type.get(), dedicated_sun.get(),
        sun_distance.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM solar-system hierarchy");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM solar-system hierarchy");
    status.download(&result.status, 1);
    if (result.status != kFmmmSolarSystemOk) return result;

    int edge_count = 0;
    coarse_node_count.download(&result.coarse_node_count, 1);
    coarse_edge_count.download(&edge_count, 1);
    result.parent.resize(static_cast<size_t>(node_count));
    result.node_type.resize(static_cast<size_t>(node_count));
    result.dedicated_sun.resize(static_cast<size_t>(node_count));
    result.sun_distance.resize(static_cast<size_t>(node_count));
    parent.download(result.parent.data(), result.parent.size());
    node_type.download(result.node_type.data(), result.node_type.size());
    dedicated_sun.download(result.dedicated_sun.data(), result.dedicated_sun.size());
    sun_distance.download(result.sun_distance.data(), result.sun_distance.size());
    std::vector<int> source(static_cast<size_t>(edge_count)), target(static_cast<size_t>(edge_count));
    std::vector<float> length(static_cast<size_t>(edge_count));
    if (edge_count > 0) {
        cuda_check(cudaMemcpy(source.data(), coarse_source.get(), source.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM solar-system sources");
        cuda_check(cudaMemcpy(target.data(), coarse_target.get(), target.size() * sizeof(int), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM solar-system targets");
        cuda_check(cudaMemcpy(length.data(), coarse_length.get(), length.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM solar-system lengths");
    }
    for (int edge = 0; edge < edge_count; ++edge) {
        result.edges.push_back({source[edge], target[edge], length[edge]});
    }
    DeviceBuffer<double> box_length(1), initial_x(static_cast<size_t>(result.coarse_node_count)),
        initial_y(static_cast<size_t>(result.coarse_node_count));
    fmmm_coarsest_random_placement_reference<<<1, 1>>>(result.coarse_node_count, seed, box_length.get(),
                                                        initial_x.get(), initial_y.get());
    cuda_check(cudaGetLastError(), "CUDA FMMM coarsest initial placement");
    cuda_check(cudaDeviceSynchronize(), "CUDA FMMM coarsest initial placement");
    box_length.download(&result.coarsest_box_length, 1);
    std::vector<double> placement_x(static_cast<size_t>(result.coarse_node_count)),
        placement_y(static_cast<size_t>(result.coarse_node_count));
    initial_x.download(placement_x.data(), placement_x.size());
    initial_y.download(placement_y.data(), placement_y.size());
    for (int node = 0; node < result.coarse_node_count; ++node) {
        result.coarsest_initial_placement.emplace_back(placement_x[node], placement_y[node]);
    }
    return result;
}

py::dict fmmm_solar_system_dict(const FmmmSolarSystemDiagnostic& result) {
    py::dict out;
    out["provider"] = "CUDA_FMMM_SOLAR_SYSTEM_DIAGNOSTIC";
    out["generation_device"] = "CUDA";
    out["timed"] = false;
    out["bounded"] = true;
    out["status"] = result.status == kFmmmSolarSystemOk ? "OK" :
                    (result.status == kFmmmSolarSystemCapacity ? "UNSUPPORTED_CAPACITY" :
                                                                "NO_MOON_NEIGHBOUR");
    out["parent"] = result.parent;
    out["node_type"] = result.node_type;
    out["dedicated_sun"] = result.dedicated_sun;
    out["sun_distance"] = result.sun_distance;
    out["coarsest_box_length"] = result.coarsest_box_length;
    out["coarsest_initial_placement"] = result.coarsest_initial_placement;
    out["coarse_node_count"] = result.coarse_node_count;
    py::list edges;
    for (const auto& edge : result.edges) edges.append(py::make_tuple(edge.source, edge.target, edge.length));
    out["coarse_edges"] = edges;
    return out;
}

constexpr int kFmmmMinGraphSize = 50;

struct FmmmTraceStage {
    std::string name;
    std::string force_domain;
    std::vector<std::pair<std::int64_t,std::int64_t>> coordinates;
    std::vector<std::pair<float,float>> forces;
    std::vector<std::int32_t> parent;
};

struct FmmmDiagnosticTrace {
    std::vector<FmmmTraceStage> stages;
    int normal_force_iterations = 0;
    int postprocess_iterations = 0;
    int fine_tuning_iterations = 0;
};

struct FmmmForceSchedule { int normal; int postprocess; int fine_tuning; };

FmmmForceSchedule fmmm_force_schedule(int requested_iterations, int node_count) {
    if (requested_iterations <= 0 || node_count <= 1) return {0, 0, 0};
    // Pinned OGDF: LinearlyDecreasing with factor 10; its small-graph guard
    // raises a normal-force count below 100. Postprocessing is finest-only.
    const int normal = node_count <= 500
        ? std::max(100, 10 * requested_iterations)
        : 10 * requested_iterations;
    return {normal, 10, 20};
}

std::vector<std::pair<std::int64_t,std::int64_t>> download_fmmm_coordinates(DeviceGraph& graph);

void fmmm_trace_snapshot(FmmmDiagnosticTrace* trace, const char* name, DeviceGraph& graph,
                         const float* force_x = nullptr, const float* force_y = nullptr,
                         int force_count = 0, const char* force_domain = "") {
    if (trace == nullptr) return;
    FmmmTraceStage stage;
    stage.name = name;
    stage.force_domain = force_domain;
    stage.coordinates = download_fmmm_coordinates(graph);
    if (force_x != nullptr && force_y != nullptr && force_count > 0) {
        std::vector<float> x(force_count), y(force_count);
        cuda_check(cudaMemcpy(x.data(), force_x, x.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM diagnostic force x");
        cuda_check(cudaMemcpy(y.data(), force_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "download CUDA FMMM diagnostic force y");
        for (int i = 0; i < force_count; ++i) stage.forces.emplace_back(x[i], y[i]);
    }
    trace->stages.push_back(std::move(stage));
}

void cuda_fmmm_generate(DeviceGraph& graph, std::uint64_t seed, int iterations, bool layers_enabled,
                        FmmmDiagnosticTrace* trace = nullptr) {
    // This records the schedule actually implemented by this CUDA reference.
    // It is deliberately separate from the frozen OGDF schedule: Phase 1
    // diagnostics must expose a scheduling mismatch, not hide it behind the
    // common ``iterations`` argument.
    const FmmmForceSchedule schedule = fmmm_force_schedule(iterations, graph.node_count());
    if (trace != nullptr) {
        trace->normal_force_iterations = schedule.normal;
        trace->postprocess_iterations = schedule.postprocess;
        trace->fine_tuning_iterations = schedule.fine_tuning;
    }
    cuda_fmmm_initial_placement<<<1,1>>>(graph.view(), seed); cuda_check(cudaGetLastError(),"CUDA FMMM initial placement"); cuda_check(cudaDeviceSynchronize(),"CUDA FMMM initial placement");
    fmmm_trace_snapshot(trace, "initial_placement", graph);
    if(iterations>0&&graph.node_count()>1){
        const auto view=graph.view();
        if (graph.node_count() > kFmmmMinGraphSize) {
            const int coarse_count=(graph.node_count()+1)/2;
            DeviceBuffer<float> coarse_x(coarse_count),coarse_y(coarse_count),coarse_fx(coarse_count),coarse_fy(coarse_count);
            fmmm_pair_coarsen<<<(coarse_count+127)/128,128>>>(view,coarse_x.get(),coarse_y.get(),coarse_count);cuda_check(cudaGetLastError(),"coarsen CUDA FMMM seed");
            fmmm_trace_snapshot(trace, "coarsen_index_pair", graph);
            if (trace != nullptr) {
                trace->stages.back().parent.resize(graph.node_count());
                for (int node = 0; node < graph.node_count(); ++node) trace->stages.back().parent[node] = node / 2;
            }
            fmmm_bh_force(coarse_x.get(),coarse_y.get(),coarse_fx.get(),coarse_fy.get(),coarse_count);
            fmmm_trace_snapshot(trace, "coarse_repulsion", graph, coarse_fx.get(), coarse_fy.get(), coarse_count, "coarse");
            fmmm_apply_coarse_force<<<1,1>>>(view,coarse_fx.get(),coarse_fy.get());cuda_check(cudaGetLastError(),"uncoarsen CUDA FMMM seed");
            fmmm_trace_snapshot(trace, "uncoarsen_index_pair", graph);
        } else {
            fmmm_trace_snapshot(trace, "coarsen_skipped_min_graph_size", graph);
        }
        DeviceBuffer<float> position_x(graph.node_count()),position_y(graph.node_count()),force_x(graph.node_count()),force_y(graph.node_count()),attr_x(graph.node_count()),attr_y(graph.node_count());
        fmmm_copy_positions<<<(graph.node_count()+127)/128,128>>>(view,position_x.get(),position_y.get());cuda_check(cudaGetLastError(),"initialize continuous CUDA FMMM positions");
        const int total_force_passes = schedule.normal + schedule.postprocess + schedule.fine_tuning;
        for(int pass=0;pass<total_force_passes;++pass){const int phase=pass<schedule.normal?0:(pass<schedule.normal+schedule.postprocess?1:2);const int local_iter=phase==0?pass+1:(phase==1?pass-schedule.normal+1:pass-schedule.normal-schedule.postprocess+1);fmmm_bh_force(position_x.get(),position_y.get(),force_x.get(),force_y.get(),graph.node_count(),view.source,view.target,graph.edge_count(),attr_x.get(),attr_y.get());if (pass < 3) fmmm_trace_snapshot(trace, ("fine_force_" + std::to_string(pass)).c_str(), graph, force_x.get(), force_y.get(), graph.node_count(), "fine");fmmm_apply_continuous_force<<<(graph.node_count()+127)/128,128>>>(position_x.get(),position_y.get(),force_x.get(),force_y.get(),attr_x.get(),attr_y.get(),graph.node_count(),phase,local_iter,11.0F*static_cast<float>(graph.node_count()));cuda_check(cudaGetLastError(),"apply continuous CUDA FMMM force");if (pass < 3) fmmm_trace_snapshot(trace, ("fine_move_" + std::to_string(pass)).c_str(), graph);}
        fmmm_quantize_continuous_positions<<<1,1>>>(view,position_x.get(),position_y.get());cuda_check(cudaGetLastError(),"quantize continuous CUDA FMMM positions");
        lcn::release::fmmm_full_square_project(view, layers_enabled, iterations, seed);
        cuda_check(cudaDeviceSynchronize(),"CUDA multilevel Barnes-Hut seed force");
        fmmm_trace_snapshot(trace, "exact_projection", graph);
    }
}

std::vector<std::pair<std::int64_t,std::int64_t>> download_fmmm_coordinates(DeviceGraph& graph) {
    std::vector<std::int64_t> x(graph.node_count()),y(graph.node_count()); const auto view=graph.view();
    if(!x.empty()){cuda_check(cudaMemcpy(x.data(),view.x,x.size()*sizeof(std::int64_t),cudaMemcpyDeviceToHost),"CUDA FMMM coordinate download");cuda_check(cudaMemcpy(y.data(),view.y,y.size()*sizeof(std::int64_t),cudaMemcpyDeviceToHost),"CUDA FMMM coordinate download");}
    std::vector<std::pair<std::int64_t,std::int64_t>> out;for(int i=0;i<graph.node_count();++i)out.emplace_back(x[i],y[i]);return out;
}

py::dict fmmm_trace_dict(const FmmmDiagnosticTrace& trace) {
    py::dict out;
    out["schema_version"] = "lcn.fmmm-diagnostic-trace.v1";
    out["timed"] = false;
    out["bounded"] = true;
    py::dict force_schedule;
    force_schedule["normal_force_iterations"] = trace.normal_force_iterations;
    force_schedule["postprocess_iterations"] = trace.postprocess_iterations;
    force_schedule["fine_tuning_iterations"] = trace.fine_tuning_iterations;
    out["force_schedule"] = force_schedule;
    py::list stages;
    for (const auto& stage : trace.stages) {
        py::dict record;
        record["stage"] = stage.name;
        record["coordinates"] = stage.coordinates;
        record["force_domain"] = stage.force_domain.empty() ? py::none() : py::cast(stage.force_domain);
        record["forces"] = stage.forces;
        record["parent"] = stage.parent;
        stages.append(record);
    }
    out["stages"] = stages;
    return out;
}

// Phase-1 result construction is intentionally kept beside the device
// launcher.  In particular, there is no CPU layout iteration or CPU scorer:
// the only final-state readback is after the device reference pipeline and
// FULL_SQUARE_GPU have both completed.  Host-side exceptions are converted to
// a bounded device-backed publication by the pybind boundary below.
constexpr const char* kFmmmReferenceProfile = "ogdf-foxglove-202510-profile-1";
constexpr const char* kFmmmProfileHash = "3a224acde16c030643747e42747f81130cdd170498ba9d8ef6f647a804beaac5";

struct ResidentSessionGuard {
    lcn::release::DeviceFmmmSession* value = nullptr;
    ~ResidentSessionGuard() {
        if (value != nullptr) {
            lcn::release::fmmm_session_destroy(*value);
        }
    }
};

// Owns a D2D snapshot of session position_x/y so FMMM→search handoff can
// keep device-resident coords after the session is destroyed.
struct FmmmDeviceCoordKeepalive {
    double* x = nullptr;
    double* y = nullptr;
    std::int32_t node_count = 0;
};

void free_fmmm_device_coord_keepalive(void* raw) {
    auto* keep = static_cast<FmmmDeviceCoordKeepalive*>(raw);
    if (keep == nullptr) return;
    if (keep->x != nullptr) cudaFree(keep->x);
    if (keep->y != nullptr) cudaFree(keep->y);
    delete keep;
}

void attach_retained_device_coords(py::dict& out, lcn::release::DeviceFmmmSession& session) {
    out["retain_device"] = false;
    out["device_x_ptr"] = static_cast<std::uintptr_t>(0);
    out["device_y_ptr"] = static_cast<std::uintptr_t>(0);
    out["device_node_count"] = 0;
    out["device_dtype"] = "float64";
    out["device_keepalive"] = py::none();
    if (!session.position_x.d_ptr || !session.position_y.d_ptr) return;
    const auto n = static_cast<std::int32_t>(session.position_x.valid_elements);
    if (n <= 0 || static_cast<std::size_t>(n) != session.position_y.valid_elements) return;

    double* dx = nullptr;
    double* dy = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    if (cudaMalloc(reinterpret_cast<void**>(&dx), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dy), bytes) != cudaSuccess ||
        cudaMemcpy(dx, session.position_x.d_ptr, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess ||
        cudaMemcpy(dy, session.position_y.d_ptr, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        if (dx != nullptr) cudaFree(dx);
        if (dy != nullptr) cudaFree(dy);
        return;
    }
    auto* keep = new FmmmDeviceCoordKeepalive{dx, dy, n};
    out["retain_device"] = true;
    out["device_x_ptr"] = reinterpret_cast<std::uintptr_t>(dx);
    out["device_y_ptr"] = reinterpret_cast<std::uintptr_t>(dy);
    out["device_node_count"] = n;
    out["device_dtype"] = "float64";
    out["device_keepalive"] = py::capsule(keep, "lcn.fmmm_device_coords.v1",
                                          &free_fmmm_device_coord_keepalive);
}

// Error publications are deliberately fixed-size.  No layout, score, or
// host-owned partial state is copied into this record.  The record is written
// into the session publication backing allocation and is consumed by the same
// final bounded D2H path as a successful publication.
__global__ void resident_error_publication_kernel(
    std::uint8_t* backing,
    lcn::release::DeviceFmmmPublicationRecord* header_out,
    int status,
    int reason_code,
    std::uint64_t graph_epoch,
    std::uint64_t tree_epoch,
    std::uint64_t publication_epoch,
    std::uint64_t allocation_generation) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    auto* record = reinterpret_cast<lcn::release::DeviceFmmmPublicationRecord*>(backing);
    *record = lcn::release::DeviceFmmmPublicationRecord{};
    record->status = status;
    record->reason_code = reason_code;
    record->admitted = 0;
    record->fallback_required = 1;
    record->node_count = 0;
    record->edge_count = 0;
    record->graph_epoch = graph_epoch;
    record->tree_epoch = tree_epoch;
    record->publication_epoch = publication_epoch;
    record->allocation_generation = allocation_generation;
    *header_out = *record;
}

int resident_error_reason_code(const std::string& reason) {
    // Stable bounded codes; the human/machine-readable string remains in the
    // Python publication envelope while the device record stays fixed-size.
    if (reason == "upload") return 10;
    if (reason == "normalize") return 11;
    if (reason == "hierarchy") return 12;
    if (reason == "build_nmm") return 13;
    if (reason == "run_schedule") return 14;
    if (reason == "score_and_admit") return 15;
    if (reason == "publish_device") return 16;
    if (reason == "publication_not_ready") return 17;
    if (reason == "allocation_limit") return 18;
    if (reason == "invalid_endpoint") return 19;
    if (reason == "self_loop") return 20;
    return 1;
}

bool resident_enqueue_error_publication(lcn::release::DeviceFmmmSession& session,
                                        lcn::release::FmmmDeviceStatus status,
                                        const std::string& reason) {
    if (session.publication_backing.d_ptr == nullptr ||
        session.publication_header.d_ptr == nullptr ||
        session.publication_backing.capacity_elements <
            sizeof(lcn::release::DeviceFmmmPublicationRecord)) {
        return false;
    }
    ++session.publication_epoch;
    if (cudaMemsetAsync(session.publication_backing.d_ptr, 0,
                        session.publication_backing.capacity_elements, nullptr) != cudaSuccess) {
        return false;
    }
    resident_error_publication_kernel<<<1, 1>>>(
        session.publication_backing.d_ptr,
        session.publication_header.d_ptr,
        static_cast<int>(status),
        resident_error_reason_code(reason),
        session.graph_epoch,
        session.tree_epoch,
        session.publication_epoch,
        session.allocation_generation);
    if (cudaGetLastError() != cudaSuccess) return false;
    session.publication_ready = true;
    return true;
}

py::dict resident_publication_telemetry(const lcn::release::DeviceFmmmPublication& publication) {
    if (publication.telemetry_json.empty()) {
        throw std::runtime_error("device publication did not provide telemetry JSON");
    }
    py::dict telemetry = py::cast<py::dict>(
        py::module_::import("json").attr("loads")(publication.telemetry_json));
    telemetry["graph_epoch"] = publication.graph_epoch;
    telemetry["tree_epoch"] = publication.tree_epoch;
    telemetry["publication_epoch"] = publication.publication_epoch;
    telemetry["allocation_generation"] = publication.allocation_generation;

    // The publication must carry the session's complete audit record. Do not
    // turn missing records into an apparently valid aggregate-only ledger.
    if (!telemetry.contains("transfer_records") ||
        !py::isinstance<py::list>(telemetry["transfer_records"])) {
        throw std::runtime_error("device publication transfer records are missing");
    }
    const py::list source_records = py::reinterpret_borrow<py::list>(telemetry["transfer_records"]);
    if (source_records.empty()) {
        throw std::runtime_error("device publication transfer record list is empty");
    }
    const char* required_record_fields[] = {
        "sequence", "phase_id", "operation", "direction", "bytes",
        "source_kind", "destination_kind", "callsite_id", "stream_id",
        "allocation_generation"
    };
    for (const py::handle item : source_records) {
        if (!py::isinstance<py::dict>(item)) {
            throw std::runtime_error("device publication transfer record is not an object");
        }
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        for (const char* field : required_record_fields) {
            if (!record.contains(field)) {
                throw std::runtime_error(std::string("device publication transfer record missing ") + field);
            }
        }
    }
    if (telemetry.contains("transfer_record_count") &&
        py::cast<std::size_t>(telemetry["transfer_record_count"]) != source_records.size()) {
        throw std::runtime_error("device publication transfer record count mismatch");
    }

    // The current session ABI exposes the device-owned record stream through
    // the bounded publication telemetry JSON. The ABI may add a display alias
    // (`kind`) but never invents a record or a transfer event.
    py::dict transfer_ledger;
    for (const char* key : {
             "input_upload_groups", "intermediate_h2d_groups",
             "intermediate_d2h_groups", "device_to_device_groups",
             "final_download_groups", "host_iteration_poll_groups",
             "phase_allocation_groups", "phase_free_groups"}) {
        if (telemetry.contains(key)) transfer_ledger[key] = telemetry[key];
    }
    {
        py::list records;
        for (const py::handle item : source_records) {
            py::dict record = py::reinterpret_borrow<py::dict>(item);
            if (!record.contains("kind")) {
                record["kind"] = record.contains("operation") ? record["operation"] : py::str("unknown");
            }
            records.append(record);
        }
        transfer_ledger["records"] = records;
    }
    transfer_ledger["source"] = "session_transfer_counters";
    telemetry["transfer_ledger"] = transfer_ledger;
    return telemetry;
}

py::dict resident_device_error_result(
    const lcn::release::DeviceFmmmPublication& publication,
    const char* status,
    const std::string& reason,
    std::uint64_t seed,
    int iterations,
    const std::string& config) {
    py::dict out;
    out["backend"] = "fmmm-gpu";
    out["provider"] = "CUDA_FMMM_SEED";
    out["status"] = status;
    out["reason"] = reason;
    // These values originate from the bounded device publication.  They are
    // intentionally empty/None for every failure; no host fallback result is
    // synthesized from input vectors or an intermediate device buffer.
    out["coordinates"] = publication.coordinates;
    out["score"] = py::none();
    out["admitted"] = false;
    out["fallback_required"] = true;
    out["fallback_reason"] = reason;
    out["publication_bounded"] = true;
    out["seed"] = seed;
    out["iterations"] = iterations;
    out["config"] = config;
    out["config_hash"] = kFmmmProfileHash;
    out["reference_profile"] = kFmmmReferenceProfile;
    out["generation_device"] = "CUDA";
    out["admission_authority"] = "CONTINUOUS_CROSSINGS_V1";
    out["algorithm"] = "cuda_fmmm_device_resident_error_publication";
    out["raw_trace"] = py::none();
    out["route"] = "device_resident_session";
    out["graph_epoch"] = publication.graph_epoch;
    out["tree_epoch"] = publication.tree_epoch;
    out["publication_epoch"] = publication.publication_epoch;
    out["allocation_generation"] = publication.allocation_generation;
    out["telemetry"] = resident_publication_telemetry(publication);
    out["telemetry_json"] = publication.telemetry_json;
    return out;
}

py::dict resident_device_failure(
    lcn::release::DeviceFmmmSession* existing_session,
    const char* status,
    const std::string& reason,
    std::uint64_t seed,
    int iterations,
    const std::string& config,
    std::int64_t width,
    std::int64_t height) {
    ResidentSessionGuard temporary;
    lcn::release::DeviceFmmmSession* session = existing_session;
    if (session == nullptr) {
        // A create/parsing failure has no session yet.  Use an empty, valid
        // session only as the bounded publication owner: the rejected input is
        // never uploaded and no input-derived state enters the device.
        lcn::release::RawFmmmInput empty_input;
        empty_input.max_width = std::max<std::int64_t>(3, width);
        empty_input.max_height = std::max<std::int64_t>(3, height);
        lcn::release::FmmmConfig empty_config;
        empty_config.seed = seed;
        empty_config.iterations = std::clamp(iterations, 0, 64);
        empty_config.layers_enabled = false;
        empty_config.config_string = config;
        temporary.value = lcn::release::fmmm_session_create(empty_input, empty_config);
        session = temporary.value;
    }
    if (session == nullptr) {
        throw std::runtime_error("unable to allocate device-backed failure publication");
    }
    const auto device_status = std::string(status) == "INVALID_INPUT"
        ? lcn::release::FmmmDeviceStatus::invalid_input
        : lcn::release::FmmmDeviceStatus::backend_error;
    if (!resident_enqueue_error_publication(*session, device_status, reason)) {
        throw std::runtime_error("unable to enqueue device-backed failure publication");
    }
    // This is the only host transfer for the failure result.  It copies the
    // fixed publication backing allocation, not positions, score, or status
    // from any intermediate phase.
    const auto publication = lcn::release::fmmm_session_download_publication(*session);
    if (publication.telemetry_json.empty()) {
        throw std::runtime_error("device-backed failure publication download failed");
    }
    return resident_device_error_result(publication, status, reason, seed, iterations, config);
}

py::dict legacy_reference_rejection(const char* status, const std::string& reason,
                                    std::uint64_t seed, int iterations,
                                    const std::string& config) {
    py::dict out;
    out["backend"] = "fmmm-gpu";
    out["provider"] = "CUDA_FMMM_SEED";
    out["status"] = status;
    out["reason"] = reason;
    out["coordinates"] = py::none();
    out["score"] = py::none();
    out["admitted"] = false;
    out["seed"] = seed;
    out["iterations"] = iterations;
    out["config"] = config;
    out["config_hash"] = kFmmmProfileHash;
    out["reference_profile"] = kFmmmReferenceProfile;
    out["generation_device"] = "CUDA";
    out["admission_authority"] = "FULL_SQUARE_GPU";
    out["raw_trace"] = py::none();
    out["route"] = "reference_oracle";
    return out;
}

lcn::release::RawFmmmInput make_resident_probe_input(int node_count, int edge_count,
                                                      std::int64_t width,
                                                      std::int64_t height) {
    if (node_count < 2 || edge_count < 0 || width < 3 || height < 3) {
        throw py::value_error("invalid resident probe dimensions");
    }
    const int columns = std::max(1, static_cast<int>(width - 2));
    const int rows = std::max(1, static_cast<int>(height - 2));
    if (static_cast<std::int64_t>(node_count) > static_cast<std::int64_t>(columns) * rows) {
        throw py::value_error("resident probe node count exceeds strict-interior capacity");
    }
    lcn::release::RawFmmmInput input;
    input.max_width = width;
    input.max_height = height;
    input.nodes_x.reserve(node_count);
    input.nodes_y.reserve(node_count);
    for (int i = 0; i < node_count; ++i) {
        input.nodes_x.push_back(1 + (i % columns));
        input.nodes_y.push_back(1 + (i / columns));
    }
    input.edges_source.reserve(edge_count);
    input.edges_target.reserve(edge_count);
    input.edges_layer.reserve(edge_count);
    for (int i = 0; i < edge_count; ++i) {
        const int source = i % node_count;
        int target = (i + 1) % node_count;
        if (target == source) target = (target + 1) % node_count;
        input.edges_source.push_back(source);
        input.edges_target.push_back(target);
        input.edges_layer.push_back(0);
    }
    return input;
}

lcn::release::FmmmConfig make_resident_probe_config(std::uint64_t seed, int iterations) {
    lcn::release::FmmmConfig config;
    config.seed = seed;
    config.iterations = iterations;
    config.layers_enabled = false;
    config.config_string = kFmmmReferenceProfile;
    return config;
}

lcn::release::FmmmDeviceStatus run_resident_session_to_nmm(lcn::release::DeviceFmmmSession& session) {
    auto status = lcn::release::fmmm_session_upload(session);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_normalize(session);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_build_hierarchy(session);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    return lcn::release::fmmm_session_build_nmm(session);
}

py::dict resident_probe_error(const char* reason) {
    py::dict out;
    out["status"] = "BACKEND_ERROR";
    out["reason"] = reason;
    out["coordinates"] = py::none();
    out["score"] = py::none();
    out["admitted"] = false;
    out["fallback_required"] = true;
    return out;
}

py::dict fmmm_ok(const GraphSpec& spec, std::uint64_t seed, const std::string& config,
                 int iterations, bool layers_enabled) {
    DeviceGraph graph(spec);
    graph.upload(false);
    cuda_fmmm_generate(graph, seed, iterations, layers_enabled);
    const ScoreResult score = evaluate_full_square(graph, layers_enabled);
    if (!score.valid) {
        return legacy_reference_rejection(
            "REJECTED_BY_ADMISSION", "FULL_SQUARE_GPU rejected projected coordinates",
            seed, iterations, config);
    }
    const auto coordinates = download_fmmm_coordinates(graph);
    py::dict out;
    out["backend"] = "fmmm-gpu";
    out["provider"] = "CUDA_FMMM_SEED";
    out["status"] = "OK";
    out["reason"] = "none";
    out["coordinates"] = coordinates;
    out["score"] = score_dict(score);
    out["admitted"] = true;
    out["seed"] = seed;
    out["config"] = config;
    out["config_hash"] = kFmmmProfileHash;
    out["reference_profile"] = kFmmmReferenceProfile;
    out["generation_device"] = "CUDA";
    out["algorithm"] = "cuda_fmmm_phase1_ordered_reference";
    out["iterations"] = iterations;
    out["admission_authority"] = "FULL_SQUARE_GPU";
    // This remains false until the independent OGDF integration corpus has
    // passed.  A profile name is provenance, not evidence of equality.
    out["coordinate_parity_claim"] = false;
    // The bounded trace ABI is deliberately opt-in only; timed calls never
    // retrieve raw floating point state.
    out["raw_trace"] = py::none();
    return out;
}

py::dict fmmm_run(const py::iterable& nodes, const py::iterable& edges, std::int64_t width,
                  std::int64_t height, std::uint64_t seed, const std::string& config,
                  int iterations, bool layers_enabled, bool retain_device = false,
                  bool time_exclude_score = false,
                  const std::string& coordinate_type = "double") {
    if (width < 3 || height < 3) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", "bounds_too_small", seed, iterations, config, width, height);
    }
    if (iterations < 0 || iterations > 64) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", "iterations_out_of_range", seed, iterations, config, width, height);
    }
    try {
        lcn::release::FmmmConfig conf;
        conf.seed = seed;
        conf.iterations = iterations;
        conf.layers_enabled = layers_enabled;
        conf.config_string = config;
        if (coordinate_type == "double") {
            conf.coordinate_type = lcn::release::FmmmConfig::CoordinateType::double_precision;
        } else if (coordinate_type == "fp32") {
            conf.coordinate_type = lcn::release::FmmmConfig::CoordinateType::fp32;
        } else {
            throw py::value_error("coordinate_type must be 'double' or 'fp32'");
        }

        lcn::release::RawFmmmInput raw_input;
        for (const py::handle item : nodes) {
            const py::sequence point = py::reinterpret_borrow<py::sequence>(item);
            if (point.size() != 2) {
                throw py::value_error("each node must be an (x, y) pair");
            }
            if (!py::isinstance<py::float_>(point[0]) && !py::isinstance<py::int_>(point[0]))
                throw py::value_error("node coordinates must be finite real numbers");
            if (!py::isinstance<py::float_>(point[1]) && !py::isinstance<py::int_>(point[1]))
                throw py::value_error("node coordinates must be finite real numbers");
            raw_input.nodes_x.push_back(py::cast<double>(point[0]));
            raw_input.nodes_y.push_back(py::cast<double>(point[1]));
        }
        for (const py::handle item : edges) {
            const py::sequence edge = py::reinterpret_borrow<py::sequence>(item);
            if (edge.size() != 3) {
                throw py::value_error("each edge must be a (source, target, layer) triple");
            }
            if (!py::isinstance<py::int_>(edge[0]) || py::isinstance<py::bool_>(edge[0]) ||
                !py::isinstance<py::int_>(edge[1]) || py::isinstance<py::bool_>(edge[1]) ||
                !py::isinstance<py::int_>(edge[2]) || py::isinstance<py::bool_>(edge[2])) {
                throw py::value_error("edge values must be signed integers");
            }
            raw_input.edges_source.push_back(py::cast<int>(edge[0]));
            raw_input.edges_target.push_back(py::cast<int>(edge[1]));
            raw_input.edges_layer.push_back(py::cast<int>(edge[2]));
        }
        raw_input.max_width = width;
        raw_input.max_height = height;

        ResidentSessionGuard guard;
        guard.value = lcn::release::fmmm_session_create(raw_input, conf);
        lcn::release::DeviceFmmmSession* session = guard.value;
        if (!session) {
            const std::string reason = lcn::release::fmmm_session_last_create_error();
            const bool invalid = reason == "bounds_too_small" ||
                reason == "iterations_out_of_range" || reason == "node_tuple_arity" ||
                reason == "edge_tuple_arity" || reason == "input_count_out_of_range" ||
                reason == "invalid_endpoint" || reason == "self_loop";
            return resident_device_failure(
                nullptr,
                invalid ? "INVALID_INPUT" : "BACKEND_ERROR",
                reason.empty() ? "session_creation_failed" : reason,
                seed, iterations, config, width, height);
        }

        auto run_phase = [&](lcn::release::FmmmDeviceStatus status, const char* phase) -> py::dict {
            if (status == lcn::release::FmmmDeviceStatus::ok) return py::dict();
            const char* public_status = status == lcn::release::FmmmDeviceStatus::invalid_input
                ? "INVALID_INPUT" : "BACKEND_ERROR";
            return resident_device_failure(
                session, public_status, phase, seed, iterations, config, width, height);
        };

        using clock = std::chrono::steady_clock;
        const auto layout_t0 = clock::now();
        const auto upload_status = lcn::release::fmmm_session_upload(*session);
        if (upload_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(upload_status, "upload");
        const auto normalize_status = lcn::release::fmmm_session_normalize(*session);
        if (normalize_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(normalize_status, "normalize");
        const auto hierarchy_status = lcn::release::fmmm_session_build_hierarchy(*session);
        if (hierarchy_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(hierarchy_status, "hierarchy");
        const auto nmm_status = lcn::release::fmmm_session_build_nmm(*session);
        if (nmm_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(nmm_status, "build_nmm");
        const auto schedule_status = lcn::release::fmmm_session_run_schedule(*session);
        if (schedule_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(schedule_status, "run_schedule");

        double layout_ms = -1.0;
        double score_admit_publish_ms = -1.0;
        if (time_exclude_score) {
            // Close CUDA device_ms after schedule; sync so host layout_ms covers
            // upload→schedule. Score/admit/publish/download run AFTER this.
            const auto close_status =
                lcn::release::fmmm_session_close_layout_device_timing(*session);
            if (close_status != lcn::release::FmmmDeviceStatus::ok)
                return run_phase(close_status, "close_layout_timing");
            layout_ms = std::chrono::duration<double, std::milli>(clock::now() - layout_t0).count();
        }

        const auto score_t0 = clock::now();
        const auto score_status = lcn::release::fmmm_session_score_and_admit(*session);
        if (score_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(score_status, "score_and_admit");
        const auto publish_status = lcn::release::fmmm_session_publish_device(*session);
        if (publish_status != lcn::release::FmmmDeviceStatus::ok)
            return run_phase(publish_status, "publish_device");

        const lcn::release::DeviceFmmmPublication publication =
            lcn::release::fmmm_session_download_publication(*session);
        if (time_exclude_score) {
            score_admit_publish_ms =
                std::chrono::duration<double, std::milli>(clock::now() - score_t0).count();
        }
        if (publication.status != "OK" || !publication.admitted) {
            const char* publication_status = publication.status.empty()
                ? "BACKEND_ERROR" : publication.status.c_str();
            // A non-admitted publication is already the single bounded device
            // result.  Do not synthesize a second host-side failure state or
            // attempt a CPU fallback after the final download.
            py::dict out = resident_device_error_result(
                publication,
                "BACKEND_ERROR",
                publication.reason.empty() ? "publication_not_admitted" : publication.reason,
                seed, iterations, config);
            out["status"] = publication_status;
            out["time_exclude_score"] = time_exclude_score;
            if (time_exclude_score) {
                out["layout_ms"] = layout_ms;
                out["score_admit_publish_ms"] = score_admit_publish_ms;
            }
            return out;
        }

        ScoreResult score;
        score.status = ProposalStatus::accepted;
        score.valid = true;
        score.exact_valid = true;
        score.edge_crossings = publication.edge_crossings;
        score.K = publication.K;
        score.nK = publication.nK;
        score.Phi = publication.Phi;
        score.C = publication.C;
        score.fallback_required = false;
        score.fallback_reason = "none";

        py::dict out;
        out["backend"] = "fmmm-gpu";
        out["provider"] = "CUDA_FMMM_SEED";
        out["status"] = publication.status;
        out["reason"] = publication.reason.empty() ? "none" : publication.reason;
        out["coordinates"] = publication.coordinates;
        out["coordinate_type"] = coordinate_type;
        out["input_coordinate_type"] = "double";
        out["output_coordinate_type"] = coordinate_type;
        out["score"] = score_dict(score);
        out["admitted"] = publication.admitted;
        out["publication_bounded"] = true;
        out["seed"] = seed;
        out["config"] = config;
        out["config_hash"] = kFmmmProfileHash;
        out["reference_profile"] = kFmmmReferenceProfile;
        out["generation_device"] = "CUDA";
        out["algorithm"] = "cuda_fmmm_device_resident";
        out["iterations"] = iterations;
        out["admission_authority"] = "CONTINUOUS_CROSSINGS_V1";
        out["coordinate_parity_claim"] = false;
        out["raw_trace"] = py::none();
        out["route"] = "device_resident_session";
        out["graph_epoch"] = publication.graph_epoch;
        out["tree_epoch"] = publication.tree_epoch;
        out["publication_epoch"] = publication.publication_epoch;
        out["allocation_generation"] = publication.allocation_generation;
        py::dict telemetry = resident_publication_telemetry(publication);
        out["time_exclude_score"] = time_exclude_score;
        if (time_exclude_score) {
            out["layout_ms"] = layout_ms;
            out["score_admit_publish_ms"] = score_admit_publish_ms;
            telemetry["layout_ms"] = layout_ms;
            telemetry["score_admit_publish_ms"] = score_admit_publish_ms;
            telemetry["time_exclude_score"] = true;
            telemetry["layout_timing_boundary"] = "upload_through_run_schedule";
            telemetry["score_timing_boundary"] = "score_admit_publish_download_after_layout";
        } else {
            out["layout_ms"] = py::none();
            out["score_admit_publish_ms"] = py::none();
        }
        out["telemetry"] = telemetry;
        out["telemetry_json"] = publication.telemetry_json;
        if (retain_device) {
            attach_retained_device_coords(out, *session);
        } else {
            out["retain_device"] = false;
            out["device_x_ptr"] = static_cast<std::uintptr_t>(0);
            out["device_y_ptr"] = static_cast<std::uintptr_t>(0);
            out["device_node_count"] = 0;
            out["device_dtype"] = "float64";
            out["device_keepalive"] = py::none();
        }
        return out;
    } catch (const py::value_error& error) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", error.what(), seed, iterations, config, width, height);
    } catch (const py::type_error& error) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", error.what(), seed, iterations, config, width, height);
    } catch (const py::cast_error& error) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", error.what(), seed, iterations, config, width, height);
    } catch (const std::invalid_argument& error) {
        return resident_device_failure(
            nullptr, "INVALID_INPUT", error.what(), seed, iterations, config, width, height);
    } catch (const std::exception& error) {
        return resident_device_failure(
            nullptr, "BACKEND_ERROR", error.what(), seed, iterations, config, width, height);
    }
}

lcn::release::RawFmmmInput parse_raw_fmmm_nodes(
    const py::iterable& nodes, const py::iterable& edges,
    std::int64_t width, std::int64_t height) {
    lcn::release::RawFmmmInput raw_input;
    for (const py::handle item : nodes) {
        const py::sequence point = py::reinterpret_borrow<py::sequence>(item);
        if (point.size() != 2) {
            throw py::value_error("each node must be an (x, y) pair");
        }
        if ((!py::isinstance<py::float_>(point[0]) && !py::isinstance<py::int_>(point[0])) ||
            (!py::isinstance<py::float_>(point[1]) && !py::isinstance<py::int_>(point[1]))) {
            throw py::value_error("node coordinates must be finite real numbers");
        }
        raw_input.nodes_x.push_back(py::cast<double>(point[0]));
        raw_input.nodes_y.push_back(py::cast<double>(point[1]));
    }
    for (const py::handle item : edges) {
        const py::sequence edge = py::reinterpret_borrow<py::sequence>(item);
        if (edge.size() != 3) {
            throw py::value_error("each edge must be a (source, target, layer) triple");
        }
        if (!py::isinstance<py::int_>(edge[0]) || py::isinstance<py::bool_>(edge[0]) ||
            !py::isinstance<py::int_>(edge[1]) || py::isinstance<py::bool_>(edge[1]) ||
            !py::isinstance<py::int_>(edge[2]) || py::isinstance<py::bool_>(edge[2])) {
            throw py::value_error("edge values must be signed integers");
        }
        raw_input.edges_source.push_back(py::cast<int>(edge[0]));
        raw_input.edges_target.push_back(py::cast<int>(edge[1]));
        raw_input.edges_layer.push_back(py::cast<int>(edge[2]));
    }
    raw_input.max_width = width;
    raw_input.max_height = height;
    return raw_input;
}

py::dict publication_to_seed_result(
    const lcn::release::DeviceFmmmPublication& publication,
    std::uint64_t seed, int iterations, const std::string& config,
    const char* batch_path) {
    if (publication.status != "OK" || !publication.admitted) {
        const char* publication_status = publication.status.empty()
            ? "BACKEND_ERROR" : publication.status.c_str();
        py::dict out = resident_device_error_result(
            publication,
            "BACKEND_ERROR",
            publication.reason.empty() ? "publication_not_admitted" : publication.reason,
            seed, iterations, config);
        out["status"] = publication_status;
        out["batch_path"] = batch_path;
        return out;
    }

    ScoreResult score;
    score.status = ProposalStatus::accepted;
    score.valid = true;
    score.exact_valid = true;
    score.edge_crossings = publication.edge_crossings;
    score.K = publication.K;
    score.nK = publication.nK;
    score.Phi = publication.Phi;
    score.C = publication.C;
    score.fallback_required = false;
    score.fallback_reason = "none";

    py::dict out;
    out["backend"] = "fmmm-gpu";
    out["provider"] = "CUDA_FMMM_SEED";
    out["status"] = publication.status;
    out["reason"] = publication.reason.empty() ? "none" : publication.reason;
    out["coordinates"] = publication.coordinates;
    out["score"] = score_dict(score);
    out["admitted"] = publication.admitted;
    out["publication_bounded"] = true;
    out["seed"] = seed;
    out["config"] = config;
    out["config_hash"] = kFmmmProfileHash;
    out["reference_profile"] = kFmmmReferenceProfile;
    out["generation_device"] = "CUDA";
    out["algorithm"] = "cuda_fmmm_device_resident";
    out["iterations"] = iterations;
    out["admission_authority"] = "CONTINUOUS_CROSSINGS_V1";
    out["coordinate_parity_claim"] = false;
    out["raw_trace"] = py::none();
    out["route"] = "device_resident_session";
    out["graph_epoch"] = publication.graph_epoch;
    out["tree_epoch"] = publication.tree_epoch;
    out["publication_epoch"] = publication.publication_epoch;
    out["allocation_generation"] = publication.allocation_generation;
    out["telemetry"] = resident_publication_telemetry(publication);
    out["telemetry_json"] = publication.telemetry_json;
    out["retain_device"] = false;
    out["device_x_ptr"] = static_cast<std::uintptr_t>(0);
    out["device_y_ptr"] = static_cast<std::uintptr_t>(0);
    out["device_node_count"] = 0;
    out["device_dtype"] = "float64";
    out["device_keepalive"] = py::none();
    out["batch_path"] = batch_path;
    return out;
}

lcn::release::FmmmDeviceStatus enqueue_session_pipeline(
    lcn::release::DeviceFmmmSession& session, cudaStream_t stream) {
    auto status = lcn::release::fmmm_session_upload(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_normalize(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_build_hierarchy(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_build_nmm(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_run_schedule(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_score_and_admit(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    return lcn::release::fmmm_session_publish_device(session, stream);
}

lcn::release::FmmmDeviceStatus enqueue_session_pipeline_from_hierarchy(
    lcn::release::DeviceFmmmSession& session, cudaStream_t stream) {
    auto status = lcn::release::fmmm_session_build_hierarchy(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_build_nmm(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_run_schedule(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    status = lcn::release::fmmm_session_score_and_admit(session, stream);
    if (status != lcn::release::FmmmDeviceStatus::ok) return status;
    return lcn::release::fmmm_session_publish_device(session, stream);
}

// Single-invocation multi-stream batch: B independent device sessions on B
// CUDA streams. Hierarchy is not shared (position/seed dependent); force and
// score phases overlap on the device. Matches per-seed fmmm_seed semantics.
py::list fmmm_run_batch_multistream(
    const std::vector<lcn::release::RawFmmmInput>& inputs,
    std::uint64_t seed, const std::string& config, int iterations,
    bool layers_enabled) {
    const std::size_t batch = inputs.size();
    py::list out;
    if (batch == 0) return out;

    struct Lane {
        lcn::release::DeviceFmmmSession* session = nullptr;
        cudaStream_t stream = nullptr;
        std::uint64_t seed = 0;
        bool enqueue_ok = false;
        const char* fail_phase = nullptr;
        lcn::release::FmmmDeviceStatus fail_status = lcn::release::FmmmDeviceStatus::ok;
    };
    std::vector<Lane> lanes(batch);

    auto cleanup = [&]() {
        for (Lane& lane : lanes) {
            if (lane.session != nullptr) {
                lcn::release::fmmm_session_destroy(*lane.session);
                lane.session = nullptr;
            }
            if (lane.stream != nullptr) {
                cudaStreamDestroy(lane.stream);
                lane.stream = nullptr;
            }
        }
    };

    try {
        for (std::size_t i = 0; i < batch; ++i) {
            lanes[i].seed = seed + static_cast<std::uint64_t>(i);
            lcn::release::FmmmConfig conf;
            conf.seed = lanes[i].seed;
            conf.iterations = iterations;
            conf.layers_enabled = layers_enabled;
            conf.config_string = config;
            lanes[i].session = lcn::release::fmmm_session_create(inputs[i], conf);
            if (!lanes[i].session) {
                cleanup();
                for (std::size_t j = 0; j < batch; ++j) {
                    out.append(resident_device_failure(
                        nullptr, "BACKEND_ERROR",
                        lcn::release::fmmm_session_last_create_error(),
                        seed + static_cast<std::uint64_t>(j), iterations, config,
                        inputs[j].max_width, inputs[j].max_height));
                }
                return out;
            }
            if (cudaStreamCreateWithFlags(&lanes[i].stream, cudaStreamNonBlocking) !=
                cudaSuccess) {
                cleanup();
                throw std::runtime_error("cudaStreamCreateWithFlags failed");
            }
        }

        for (std::size_t i = 0; i < batch; ++i) {
            const auto status = enqueue_session_pipeline(*lanes[i].session, lanes[i].stream);
            if (status != lcn::release::FmmmDeviceStatus::ok) {
                lanes[i].enqueue_ok = false;
                lanes[i].fail_status = status;
                lanes[i].fail_phase = "device_batch_enqueue";
            } else {
                lanes[i].enqueue_ok = true;
            }
        }

        for (std::size_t i = 0; i < batch; ++i) {
            Lane& lane = lanes[i];
            if (!lane.enqueue_ok) {
                const char* public_status =
                    lane.fail_status == lcn::release::FmmmDeviceStatus::invalid_input
                        ? "INVALID_INPUT" : "BACKEND_ERROR";
                out.append(resident_device_failure(
                    lane.session, public_status,
                    lane.fail_phase ? lane.fail_phase : "device_batch_enqueue",
                    lane.seed, iterations, config,
                    inputs[i].max_width, inputs[i].max_height));
                continue;
            }
            const auto publication =
                lcn::release::fmmm_session_download_publication(*lane.session);
            py::dict item = publication_to_seed_result(
                publication, lane.seed, iterations, config, "device_multistream");
            out.append(item);
        }
        cleanup();
        return out;
    } catch (...) {
        cleanup();
        throw;
    }
}

// Shared-session MVP: one session, upload+normalize once, then rearm seed and
// rerun hierarchy→publish for each layout. Hierarchy is rebuilt per seed
// (positions are seed-dependent) but graph alloc/upload/normalize are amortized.
py::list fmmm_run_batch_shared_session(
    const lcn::release::RawFmmmInput& graph_input,
    std::size_t batch, std::uint64_t seed, const std::string& config,
    int iterations, bool layers_enabled) {
    py::list out;
    if (batch == 0) return out;

    lcn::release::FmmmConfig conf;
    conf.seed = seed;
    conf.iterations = iterations;
    conf.layers_enabled = layers_enabled;
    conf.config_string = config;

    ResidentSessionGuard guard;
    guard.value = lcn::release::fmmm_session_create(graph_input, conf);
    if (!guard.value) {
        for (std::size_t i = 0; i < batch; ++i) {
            out.append(resident_device_failure(
                nullptr, "BACKEND_ERROR",
                lcn::release::fmmm_session_last_create_error(),
                seed + static_cast<std::uint64_t>(i), iterations, config,
                graph_input.max_width, graph_input.max_height));
        }
        return out;
    }
    lcn::release::fmmm_session_enable_reuse(*guard.value);

    auto status = lcn::release::fmmm_session_upload(*guard.value);
    if (status != lcn::release::FmmmDeviceStatus::ok) {
        for (std::size_t i = 0; i < batch; ++i) {
            out.append(resident_device_failure(
                guard.value, "BACKEND_ERROR", "upload",
                seed + static_cast<std::uint64_t>(i), iterations, config,
                graph_input.max_width, graph_input.max_height));
        }
        return out;
    }
    status = lcn::release::fmmm_session_normalize(*guard.value);
    if (status != lcn::release::FmmmDeviceStatus::ok) {
        for (std::size_t i = 0; i < batch; ++i) {
            out.append(resident_device_failure(
                guard.value, "BACKEND_ERROR", "normalize",
                seed + static_cast<std::uint64_t>(i), iterations, config,
                graph_input.max_width, graph_input.max_height));
        }
        return out;
    }

    for (std::size_t i = 0; i < batch; ++i) {
        const std::uint64_t item_seed = seed + static_cast<std::uint64_t>(i);
        if (i > 0) {
            if (!lcn::release::fmmm_session_rearm_for_seed(*guard.value, item_seed)) {
                out.append(resident_device_failure(
                    guard.value, "BACKEND_ERROR", "rearm_for_seed",
                    item_seed, iterations, config,
                    graph_input.max_width, graph_input.max_height));
                continue;
            }
        } else {
            guard.value->config.seed = item_seed;
        }
        status = enqueue_session_pipeline_from_hierarchy(*guard.value, nullptr);
        if (status != lcn::release::FmmmDeviceStatus::ok) {
            const char* public_status =
                status == lcn::release::FmmmDeviceStatus::invalid_input
                    ? "INVALID_INPUT" : "BACKEND_ERROR";
            out.append(resident_device_failure(
                guard.value, public_status, "shared_session_pipeline",
                item_seed, iterations, config,
                graph_input.max_width, graph_input.max_height));
            // Fail-closed for later items: attempt rearm recovery only if
            // publication completed; otherwise stop contaminating the session.
            continue;
        }
        const auto publication =
            lcn::release::fmmm_session_download_publication(*guard.value);
        out.append(publication_to_seed_result(
            publication, item_seed, iterations, config, "device_shared_session"));
    }
    return out;
}

py::list fmmm_run_batch(
    const py::iterable& layouts, const py::iterable& edges, std::int64_t width,
    std::int64_t height, std::uint64_t seed, const std::string& config,
    int iterations, bool layers_enabled) {
    if (width < 3 || height < 3) {
        py::list out;
        out.append(resident_device_failure(
            nullptr, "INVALID_INPUT", "bounds_too_small", seed, iterations, config,
            width, height));
        return out;
    }
    if (iterations < 0 || iterations > 64) {
        py::list out;
        out.append(resident_device_failure(
            nullptr, "INVALID_INPUT", "iterations_out_of_range", seed, iterations,
            config, width, height));
        return out;
    }

    std::vector<lcn::release::RawFmmmInput> inputs;
    for (const py::handle nodes : layouts) {
        inputs.push_back(parse_raw_fmmm_nodes(
            py::reinterpret_borrow<py::iterable>(nodes), edges, width, height));
    }
    if (inputs.empty()) return py::list();

    // Prefer multi-stream concurrent device batch. Fall back to shared-session
    // amortization when LCN_FMMM_BATCH_MODE=shared_session (debug/compare).
    const char* mode = std::getenv("LCN_FMMM_BATCH_MODE");
    const bool shared =
        mode != nullptr && (std::strcmp(mode, "shared_session") == 0 ||
                            std::strcmp(mode, "shared") == 0);
    if (shared) {
        return fmmm_run_batch_shared_session(
            inputs.front(), inputs.size(), seed, config, iterations, layers_enabled);
    }
    return fmmm_run_batch_multistream(inputs, seed, config, iterations, layers_enabled);
}

} // namespace

// Dedicated FMMM release API registration.
// All legacy simulated annealing / PTS / PDS search engines have been cleanly excised,
// leaving an uncluttered registration boundary specifically for multilevel layout.
void register_resident_api(py::module_& module) {
    module.def("fmmm_seed", [](const py::iterable& nodes, const py::iterable& edges, std::int64_t width, std::int64_t height,
                                 std::uint64_t seed, const std::string& config, int iterations, bool layers_enabled,
                                 bool retain_device, bool time_exclude_score,
                                 const std::string& coordinate_type) {
        return fmmm_run(nodes, edges, width, height, seed, config, iterations, layers_enabled, retain_device,
                        time_exclude_score, coordinate_type);
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(), py::arg("max_width"), py::arg("max_height"), py::arg("seed"),
       py::arg("config") = kFmmmReferenceProfile, py::arg("iterations") = 0, py::arg("layers_enabled") = true,
       py::arg("retain_device") = false, py::arg("time_exclude_score") = false,
       py::arg("coordinate_type") = "double");

    // Paper Table-1 timing seam: same pipeline as fmmm_seed, but layout_ms /
    // device_ms close after run_schedule; score_and_admit + publish + download
    // run afterward (untimed for layout_ms). Product CLI should keep fmmm_seed.
    module.def("fmmm_seed_layout_only", [](const py::iterable& nodes, const py::iterable& edges,
                                            std::int64_t width, std::int64_t height, std::uint64_t seed,
                                            const std::string& config, int iterations, bool layers_enabled,
                                            bool retain_device) {
        return fmmm_run(nodes, edges, width, height, seed, config, iterations, layers_enabled, retain_device,
                        /*time_exclude_score=*/true);
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(), py::arg("max_width"), py::arg("max_height"), py::arg("seed"),
       py::arg("config") = kFmmmReferenceProfile, py::arg("iterations") = 0, py::arg("layers_enabled") = true,
       py::arg("retain_device") = false);

    module.def("fmmm_solar_system_diagnostic", [](const py::iterable& nodes, const py::iterable& edges,
                                                     std::int64_t width, std::int64_t height,
                                                     std::uint64_t seed) {
        GraphSpec spec = parse_graph(nodes, edges, width, height);
        if (spec.nodes.size() <= static_cast<size_t>(kFmmmMinGraphSize) ||
            spec.nodes.size() > static_cast<size_t>(kFmmmSolarSystemMaxNodes)) {
            throw py::value_error("FMMM solar-system diagnostic is bounded to 51..64 nodes and at most 256 edges");
        }
        DeviceGraph graph(std::move(spec));
        graph.upload(false);
        return fmmm_solar_system_dict(fmmm_solar_system_diagnostic(graph, seed));
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(), py::arg("max_width") = 64,
       py::arg("max_height") = 64, py::arg("seed") = 100);

    module.def("fmmm_full_path_coarsest_placement", [](int node_count, double node_width,
                                                           double node_height, std::uint64_t seed) {
        const FmmmFullPathPlacementDiagnostic result = fmmm_full_path_coarsest_placement(
            node_count, node_width, node_height, seed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_LEVEL_ENGINE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : "UNSUPPORTED_INPUT";
        if (result.status == 0) {
            out["initial_box_length"] = result.initial_box_length;
            out["dynamic_box_length"] = result.dynamic_box_length;
            out["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
            out["positions"] = result.positions;
        } else {
            out["initial_box_length"] = py::none();
            out["dynamic_box_length"] = py::none();
            out["down_left_corner"] = py::none();
            out["positions"] = py::list();
        }
        return out;
    }, py::arg("node_count"), py::kw_only(), py::arg("node_width") = 20.0,
       py::arg("node_height") = 20.0, py::arg("seed") = 100);

    module.def("fmmm_full_path_solar_hierarchy", [](int node_count, const py::iterable& edges,
                                                        std::uint64_t seed) {
        std::vector<int> source, target;
        std::vector<double> length;
        for (const py::handle value : edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each hierarchy edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0]));
            target.push_back(py::cast<int>(edge[1]));
            length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmFullPathHierarchyDiagnostic result = fmmm_full_path_solar_hierarchy(
            node_count, source, target, length, seed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_LEVEL_ENGINE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" :
            (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        out["coarse_node_count"] = result.coarse_node_count;
        out["parent"] = result.parent;
        out["node_type"] = result.node_type;
        out["dedicated_sun"] = result.dedicated_sun;
        out["sun_distance"] = result.sun_distance;
        py::list coarse_edges;
        for (const auto& edge : result.coarse_edges) {
            coarse_edges.append(py::make_tuple(edge.source, edge.target, edge.ideal_length));
        }
        out["coarse_edges"] = coarse_edges;
        return out;
    }, py::arg("node_count"), py::arg("edges"), py::kw_only(), py::arg("seed") = 100);

    module.def("fmmm_full_path_advanced_plan", [](int node_count, const py::iterable& edges,
                                                     std::uint64_t seed) {
        std::vector<int> source, target; std::vector<double> length;
        for (const py::handle value : edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each hierarchy edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0])); target.push_back(py::cast<int>(edge[1])); length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmAdvancedPlanDiagnostic result = fmmm_full_path_advanced_plan(node_count, source, target, length, seed);
        py::dict out; out["provider"] = "CUDA_FMMM_FULL_PATH_HIERARCHY_PLAN"; out["generation_device"] = "CUDA"; out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        if (result.status != 0) return out;
        out["lower_node_count"] = result.lower_node_count; out["lower_edge_ordinal_span"] = result.lower_edge_ordinal_span;
        out["high_node_count"] = result.high_node_count; out["high_edge_ordinal_span"] = result.high_edge_ordinal_span;
        out["lower_type"] = result.lower_type; out["lower_dedicated_sun"] = result.lower_dedicated_sun; out["lower_dedicated_sun_distance"] = result.lower_dedicated_sun_distance;
        out["high_lower_node"] = result.high_lower_node; out["lower_edge_source"] = result.lower_edge_source; out["lower_edge_target"] = result.lower_edge_target; out["lower_edge_length"] = result.lower_edge_length; out["lower_edge_is_moon"] = result.lower_edge_is_moon;
        out["lower_adjacency_begin"] = result.lower_adjacency_begin; out["lower_adjacency_edge"] = result.lower_adjacency_edge; out["lambda_begin"] = result.lambda_begin; out["lambda_values"] = result.lambda_values; out["neighbour_sun_begin"] = result.neighbour_sun_begin; out["neighbour_sun_nodes"] = result.neighbour_sun_nodes; out["moon_begin"] = result.moon_begin; out["moon_nodes"] = result.moon_nodes;
        out["high_edge_source"] = result.high_edge_source; out["high_edge_target"] = result.high_edge_target; out["high_edge_is_extra"] = result.high_edge_is_extra; out["high_adjacency_begin"] = result.high_adjacency_begin; out["high_adjacency_edge"] = result.high_adjacency_edge;
        return out;
    }, py::arg("node_count"), py::arg("edges"), py::kw_only(), py::arg("seed") = 100);

    // End-to-end diagnostic seam: the host supplies only the normalized fine
    // level and the frozen coarse-placement oracle input.  CUDA constructs
    // the Advanced plan, runs the coarse force loop, and uncoarsens without
    // a plan or coordinate round trip between those three stages.
    module.def("fmmm_full_path_native_hierarchy_force_uncoarsen", [](int node_count,
                                                                        const py::iterable& fine_edges,
                                                                        const py::dict& coarse_force,
                                                                        std::uint64_t seed,
                                                                        int placement_draws_consumed) {
        std::vector<int> source, target;
        std::vector<double> length, coarse_x, coarse_y;
        for (const py::handle value : fine_edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each fine edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0]));
            target.push_back(py::cast<int>(edge[1]));
            length.push_back(py::cast<double>(edge[2]));
        }
        auto doubles = [&coarse_force](const char* name) {
            const auto values = py::reinterpret_borrow<py::sequence>(coarse_force[py::str(name)]);
            std::vector<double> out; out.reserve(static_cast<size_t>(values.size()));
            for (const py::handle value : values) out.push_back(py::cast<double>(value));
            return out;
        };
        coarse_x = doubles("position_x");
        coarse_y = doubles("position_y");
        if (coarse_x.size() != coarse_y.size()) throw py::value_error("coarse_force positions must be paired");
        std::vector<std::pair<double, double>> initial;
        initial.reserve(coarse_x.size());
        for (size_t node = 0; node < coarse_x.size(); ++node) initial.emplace_back(coarse_x[node], coarse_y[node]);
        const FmmmNativeUncoarsenDiagnostic result = fmmm_full_path_native_hierarchy_force_uncoarsen(
            node_count, source, target, length, initial,
            py::cast<double>(coarse_force["box_length"]), py::cast<double>(coarse_force["down_left_x"]),
            py::cast<double>(coarse_force["down_left_y"]), py::cast<double>(coarse_force["average_ideal_length"]),
            py::cast<int>(coarse_force["iterations"]), seed, placement_draws_consumed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_NATIVE_COORDINATOR";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        if (result.status != 0) return out;
        py::list positions;
        for (const auto& point : result.positions) positions.append(py::make_tuple(point.first, point.second));
        py::dict box;
        box["length"] = result.box_length;
        box["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
        out["position"] = positions;
        out["box"] = box;
        return out;
    }, py::arg("node_count"), py::arg("fine_edges"), py::arg("coarse_force"), py::kw_only(),
       py::arg("seed") = 100, py::arg("placement_draws_consumed") = 0);

    module.def("fmmm_full_path_native_profile_candidate", [](int node_count, const py::iterable& fine_edges,
                                                                 double average_ideal_length, int force_iterations,
                                                                 std::uint64_t seed) {
        std::vector<int> source, target;
        std::vector<double> length;
        for (const py::handle value : fine_edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each fine edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0]));
            target.push_back(py::cast<int>(edge[1]));
            length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmNativeUncoarsenDiagnostic result = fmmm_full_path_native_profile_candidate(
            node_count, source, target, length, average_ideal_length, force_iterations, seed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_NATIVE_PROFILE_CANDIDATE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        if (result.status != 0) return out;
        py::list positions;
        for (const auto& point : result.positions) positions.append(py::make_tuple(point.first, point.second));
        py::dict box;
        box["length"] = result.box_length;
        box["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
        out["position"] = positions;
        out["box"] = box;
        return out;
    }, py::arg("node_count"), py::arg("fine_edges"), py::kw_only(), py::arg("average_ideal_length"),
       py::arg("force_iterations") = 100, py::arg("seed") = 100);

    module.def("fmmm_full_path_native_profile_normal_candidate", [](int node_count, const py::iterable& fine_edges,
                                                                        double average_ideal_length, int force_iterations,
                                                                        std::uint64_t seed) {
        std::vector<int> source, target;
        std::vector<double> length;
        for (const py::handle value : fine_edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each fine edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0]));
            target.push_back(py::cast<int>(edge[1]));
            length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmNativeUncoarsenDiagnostic result = fmmm_full_path_native_profile_normal_candidate(
            node_count, source, target, length, average_ideal_length, force_iterations, seed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_NATIVE_PROFILE_NORMAL_CANDIDATE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        if (result.status != 0) return out;
        py::list positions;
        for (const auto& point : result.positions) positions.append(py::make_tuple(point.first, point.second));
        py::dict box;
        box["length"] = result.box_length;
        box["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
        out["position"] = positions;
        out["box"] = box;
        return out;
    }, py::arg("node_count"), py::arg("fine_edges"), py::kw_only(), py::arg("average_ideal_length"),
       py::arg("force_iterations") = 100, py::arg("seed") = 100);

    module.def("fmmm_full_path_native_profile_postprocessed_candidate", [](int node_count, const py::iterable& fine_edges,
                                                                               double average_ideal_length, int force_iterations,
                                                                               std::uint64_t seed) {
        std::vector<int> source, target;
        std::vector<double> length;
        for (const py::handle value : fine_edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each fine edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0])); target.push_back(py::cast<int>(edge[1])); length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmNativeUncoarsenDiagnostic result = fmmm_full_path_native_profile_postprocessed_candidate(
            node_count, source, target, length, average_ideal_length, force_iterations, seed);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_NATIVE_PROFILE_POSTPROCESSED_CANDIDATE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT");
        if (result.status != 0) return out;
        py::list positions;
        for (const auto& point : result.positions) positions.append(py::make_tuple(point.first, point.second));
        py::dict box;
        box["length"] = result.box_length; box["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
        out["position"] = positions; out["box"] = box;
        return out;
    }, py::arg("node_count"), py::arg("fine_edges"), py::kw_only(), py::arg("average_ideal_length"),
       py::arg("force_iterations") = 100, py::arg("seed") = 100);

    module.def("fmmm_full_path_native_profile_packed_candidate", [](int node_count, const py::iterable& fine_edges,
                                                                        double average_ideal_length, int force_iterations,
                                                                        std::uint64_t seed) {
        std::vector<int> source, target; std::vector<double> length;
        for (const py::handle value : fine_edges) { const auto edge = py::reinterpret_borrow<py::sequence>(value); if (edge.size() != 3) throw py::value_error("each fine edge must be a (source, target, ideal_length) triple"); source.push_back(py::cast<int>(edge[0])); target.push_back(py::cast<int>(edge[1])); length.push_back(py::cast<double>(edge[2])); }
        const FmmmNativeUncoarsenDiagnostic result = fmmm_full_path_native_profile_packed_candidate(node_count, source, target, length, average_ideal_length, force_iterations, seed);
        py::dict out; out["provider"] = "CUDA_FMMM_FULL_PATH_NATIVE_PROFILE_PACKED_CANDIDATE"; out["generation_device"] = "CUDA"; out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : (result.status == 2 ? "NO_MOON_NEIGHBOUR" : "UNSUPPORTED_INPUT"); if (result.status != 0) return out;
        py::list positions; for (const auto& point : result.positions) positions.append(py::make_tuple(point.first, point.second));
        out["position"] = positions; return out;
    }, py::arg("node_count"), py::arg("fine_edges"), py::kw_only(), py::arg("average_ideal_length"), py::arg("force_iterations") = 100, py::arg("seed") = 100);

    // A deliberately thin upload/launch/download shell around the device
    // owner.  The caller supplies the already-normalized v3 oracle arrays;
    // this binding does not derive hierarchy data or run an alternate CPU
    // placement path.  Keeping the sparse edge-ordinal spans explicit is
    // necessary because OGDF retains ordinal holes after coarsening.
    module.def("fmmm_advanced_uncoarsen_reference", [](const py::dict& plan) {
        auto ints = [&plan](const char* name) {
            const auto values = py::reinterpret_borrow<py::sequence>(plan[py::str(name)]);
            std::vector<int> out; out.reserve(static_cast<size_t>(values.size()));
            for (const py::handle value : values) out.push_back(py::cast<int>(value));
            return out;
        };
        auto doubles = [&plan](const char* name) {
            const auto values = py::reinterpret_borrow<py::sequence>(plan[py::str(name)]);
            std::vector<double> out; out.reserve(static_cast<size_t>(values.size()));
            for (const py::handle value : values) out.push_back(py::cast<double>(value));
            return out;
        };
        auto bytes = [&plan](const char* name) {
            const auto values = py::reinterpret_borrow<py::sequence>(plan[py::str(name)]);
            std::vector<std::uint8_t> out; out.reserve(static_cast<size_t>(values.size()));
            for (const py::handle value : values) out.push_back(py::cast<bool>(value) ? 1U : 0U);
            return out;
        };
        const int lower_count = py::cast<int>(plan["lower_node_count"]);
        const int lower_span = py::cast<int>(plan["lower_edge_ordinal_span"]);
        const int high_count = py::cast<int>(plan["high_node_count"]);
        const int high_span = py::cast<int>(plan["high_edge_ordinal_span"]);
        if (lower_count <= 0 || high_count <= 0) throw py::value_error("advanced uncoarsen node counts must be positive");

        const auto lower_type_h = ints("lower_type"); const auto lower_sun_h = ints("lower_dedicated_sun");
        const auto lower_distance_h = doubles("lower_dedicated_sun_distance");
        const auto high_lower_h = ints("high_lower_node"); const auto high_x_h = doubles("high_position_x"); const auto high_y_h = doubles("high_position_y");
        const auto low_source_h = ints("lower_edge_source"); const auto low_target_h = ints("lower_edge_target");
        const auto low_length_h = doubles("lower_edge_length"); const auto low_moon_h = bytes("lower_edge_is_moon");
        const auto low_adj_begin_h = ints("lower_adjacency_begin"); const auto low_adj_edge_h = ints("lower_adjacency_edge");
        const auto high_source_h = ints("high_edge_source"); const auto high_target_h = ints("high_edge_target");
        const auto high_extra_h = bytes("high_edge_is_extra");
        const auto high_adj_begin_h = ints("high_adjacency_begin"); const auto high_adj_edge_h = ints("high_adjacency_edge");
        const auto lambda_begin_h = ints("lambda_begin"); const auto lambda_h = doubles("lambda_values");
        const auto neighbour_begin_h = ints("neighbour_sun_begin"); const auto neighbour_h = ints("neighbour_sun_nodes");
        const auto moon_begin_h = ints("moon_begin"); const auto moon_h = ints("moon_nodes");
        if (lower_type_h.size() != static_cast<size_t>(lower_count) || lower_sun_h.size() != static_cast<size_t>(lower_count)
                || lower_distance_h.size() != static_cast<size_t>(lower_count) || high_lower_h.size() != static_cast<size_t>(high_count)
                || high_x_h.size() != static_cast<size_t>(high_count) || high_y_h.size() != static_cast<size_t>(high_count)
                || low_source_h.size() != static_cast<size_t>(lower_span) || low_target_h.size() != static_cast<size_t>(lower_span)
                || low_length_h.size() != static_cast<size_t>(lower_span) || low_moon_h.size() != static_cast<size_t>(lower_span)
                || high_source_h.size() != static_cast<size_t>(high_span) || high_target_h.size() != static_cast<size_t>(high_span)
                || high_extra_h.size() != static_cast<size_t>(high_span) || low_adj_begin_h.size() != static_cast<size_t>(lower_count + 1)
                || high_adj_begin_h.size() != static_cast<size_t>(high_count + 1) || lambda_begin_h.size() != static_cast<size_t>(lower_count + 1)
                || neighbour_begin_h.size() != static_cast<size_t>(lower_count + 1) || moon_begin_h.size() != static_cast<size_t>(lower_count + 1)) {
            throw py::value_error("advanced uncoarsen plan has inconsistent SoA lengths");
        }
        auto upload = [](auto& device, const auto& host) { device.upload(host.data(), host.size()); };
        DeviceBuffer<int> lower_type(lower_type_h.size()), lower_sun(lower_sun_h.size()), high_lower(high_lower_h.size());
        DeviceBuffer<double> lower_distance(lower_distance_h.size()), high_x(high_x_h.size()), high_y(high_y_h.size());
        DeviceBuffer<int> low_source(low_source_h.size()), low_target(low_target_h.size()), low_adj_begin(low_adj_begin_h.size()), low_adj_edge(low_adj_edge_h.size());
        DeviceBuffer<double> low_length(low_length_h.size()); DeviceBuffer<std::uint8_t> low_moon(low_moon_h.size());
        DeviceBuffer<int> high_source(high_source_h.size()), high_target(high_target_h.size()), high_adj_begin(high_adj_begin_h.size()), high_adj_edge(high_adj_edge_h.size());
        DeviceBuffer<std::uint8_t> high_extra(high_extra_h.size());
        DeviceBuffer<int> lambda_begin(lambda_begin_h.size()), neighbour_begin(neighbour_begin_h.size()), neighbour(neighbour_h.size()), moon_begin(moon_begin_h.size()), moon(moon_h.size());
        DeviceBuffer<double> lambda(lambda_h.size());
        upload(lower_type, lower_type_h); upload(lower_sun, lower_sun_h); upload(lower_distance, lower_distance_h); upload(high_lower, high_lower_h); upload(high_x, high_x_h); upload(high_y, high_y_h);
        upload(low_source, low_source_h); upload(low_target, low_target_h); upload(low_length, low_length_h); upload(low_moon, low_moon_h); upload(low_adj_begin, low_adj_begin_h); upload(low_adj_edge, low_adj_edge_h);
        upload(high_source, high_source_h); upload(high_target, high_target_h); upload(high_extra, high_extra_h); upload(high_adj_begin, high_adj_begin_h); upload(high_adj_edge, high_adj_edge_h);
        upload(lambda_begin, lambda_begin_h); upload(lambda, lambda_h); upload(neighbour_begin, neighbour_begin_h); upload(neighbour, neighbour_h); upload(moon_begin, moon_begin_h); upload(moon, moon_h);
        DeviceBuffer<double> position_x(static_cast<size_t>(lower_count)), position_y(static_cast<size_t>(lower_count)), angle_1(static_cast<size_t>(lower_count)), angle_2(static_cast<size_t>(lower_count)), box_length(1), down_left_x(1), down_left_y(1);
        DeviceBuffer<std::uint8_t> placed(static_cast<size_t>(lower_count)); DeviceBuffer<int> status(1);
        FmmmAdvancedUncoarsenPlanView view{lower_count, lower_span, high_count, high_span, lower_type.get(), lower_sun.get(), lower_distance.get(), high_lower.get(), high_x.get(), high_y.get(), low_source.get(), low_target.get(), low_length.get(), low_moon.get(), low_adj_begin.get(), low_adj_edge.get(), high_source.get(), high_target.get(), high_extra.get(), high_adj_begin.get(), high_adj_edge.get(), lambda_begin.get(), lambda.get(), neighbour_begin.get(), neighbour.get(), moon_begin.get(), moon.get()};
        FmmmAdvancedUncoarsenStateView output{position_x.get(), position_y.get(), placed.get(), angle_1.get(), angle_2.get(), box_length.get(), down_left_x.get(), down_left_y.get(), status.get()};
        // The optional force input creates the first real full-path device
        // seam: the coarsest loop overwrites high_x/high_y in-place and that
        // very allocation becomes Advanced uncoarsening's high-level input.
        // No coarse coordinates cross the host boundary between these stages.
        if (plan.contains("coarse_force")) {
            const py::dict force = py::reinterpret_borrow<py::dict>(plan["coarse_force"]);
            auto force_ints = [&force](const char* name) {
                const auto values = py::reinterpret_borrow<py::sequence>(force[py::str(name)]);
                std::vector<int> out; out.reserve(static_cast<size_t>(values.size()));
                for (const py::handle value : values) out.push_back(py::cast<int>(value));
                return out;
            };
            auto force_doubles = [&force](const char* name) {
                const auto values = py::reinterpret_borrow<py::sequence>(force[py::str(name)]);
                std::vector<double> out; out.reserve(static_cast<size_t>(values.size()));
                for (const py::handle value : values) out.push_back(py::cast<double>(value));
                return out;
            };
            const auto force_x_h = force_doubles("position_x"), force_y_h = force_doubles("position_y");
            const auto force_source_h = force_ints("edge_source"), force_target_h = force_ints("edge_target");
            const auto force_length_h = force_doubles("edge_length");
            if (force_x_h.size() != static_cast<size_t>(high_count) || force_y_h.size() != static_cast<size_t>(high_count)
                    || force_source_h.size() != force_target_h.size() || force_source_h.size() != force_length_h.size()) {
                throw py::value_error("coarse_force does not describe the supplied high level");
            }
            upload(high_x, force_x_h); upload(high_y, force_y_h);
            DeviceBuffer<int> force_source(force_source_h.size()), force_target(force_target_h.size()), force_status(1);
            DeviceBuffer<double> force_length(force_length_h.size()), force_box(1), force_left_x(1), force_left_y(1);
            upload(force_source, force_source_h); upload(force_target, force_target_h); upload(force_length, force_length_h);
            fmmm_full_path_exact_force_loop_device(high_count, static_cast<int>(force_source_h.size()),
                force_source.get(), force_target.get(), force_length.get(), high_x.get(), high_y.get(),
                py::cast<double>(force["box_length"]), py::cast<double>(force["down_left_x"]),
                py::cast<double>(force["down_left_y"]), py::cast<double>(force["average_ideal_length"]),
                py::cast<int>(force["iterations"]), force_status.get(), force_box.get(), force_left_x.get(), force_left_y.get());
            cuda_check(cudaDeviceSynchronize(), "CUDA FMMM coarse force to Advanced seam");
            int force_status_h = 1; force_status.download(&force_status_h, 1);
            if (force_status_h != 0) {
                py::dict failed; failed["provider"] = "CUDA_FMMM_ADVANCED_UNCOARSEN_REFERENCE";
                failed["generation_device"] = "CUDA"; failed["timed"] = false; failed["status"] = "UNSUPPORTED_INPUT";
                return failed;
            }
        }
        fmmm_advanced_uncoarsen_reference(view, output, {py::cast<std::uint64_t>(plan["seed"]), py::cast<int>(plan["placement_draws_consumed"])});
        cuda_check(cudaDeviceSynchronize(), "CUDA FMMM advanced uncoarsen completion");
        int status_h = static_cast<int>(FmmmAdvancedUncoarsenStatus::InvalidPlan); status.download(&status_h, 1);
        py::dict out; out["provider"] = "CUDA_FMMM_ADVANCED_UNCOARSEN_REFERENCE"; out["generation_device"] = "CUDA"; out["timed"] = false;
        out["status"] = status_h == static_cast<int>(FmmmAdvancedUncoarsenStatus::Ok) ? "OK" : (status_h == static_cast<int>(FmmmAdvancedUncoarsenStatus::Unsupported) ? "UNSUPPORTED_INPUT" : "INVALID_INPUT");
        if (status_h != static_cast<int>(FmmmAdvancedUncoarsenStatus::Ok)) return out;
        std::vector<double> x(static_cast<size_t>(lower_count)), y(static_cast<size_t>(lower_count)), a1(static_cast<size_t>(lower_count)), a2(static_cast<size_t>(lower_count)), box(1), left_x(1), left_y(1); std::vector<std::uint8_t> placed_h(static_cast<size_t>(lower_count));
        position_x.download(x.data(), x.size()); position_y.download(y.data(), y.size()); angle_1.download(a1.data(), a1.size()); angle_2.download(a2.data(), a2.size()); placed.download(placed_h.data(), placed_h.size()); box_length.download(box.data(), 1); down_left_x.download(left_x.data(), 1); down_left_y.download(left_y.data(), 1);
        py::list positions, angles, placed_out; for (int i = 0; i < lower_count; ++i) { positions.append(py::make_tuple(x[static_cast<size_t>(i)], y[static_cast<size_t>(i)])); angles.append(py::make_tuple(a1[static_cast<size_t>(i)], a2[static_cast<size_t>(i)])); placed_out.append(placed_h[static_cast<size_t>(i)] != 0); }
        py::dict final_box; final_box["length"] = box[0]; final_box["down_left_corner"] = py::make_tuple(left_x[0], left_y[0]);
        out["position"] = positions; out["angles"] = angles; out["placed"] = placed_out; out["box"] = final_box; return out;
    }, py::arg("plan"));

    module.def("fmmm_full_path_exact_force_step", [](const py::iterable& positions,
                                                         const py::iterable& edges, double box_length,
                                                         const py::sequence& down_left_corner,
                                                         double average_ideal_length, int iteration) {
        if (down_left_corner.size() != 2) throw py::value_error("down_left_corner must be an (x, y) pair");
        std::vector<std::pair<double, double>> host_positions;
        std::vector<int> source, target;
        std::vector<double> length;
        for (const py::handle value : positions) {
            const auto point = py::reinterpret_borrow<py::sequence>(value);
            if (point.size() != 2) throw py::value_error("each position must be an (x, y) pair");
            host_positions.emplace_back(py::cast<double>(point[0]), py::cast<double>(point[1]));
        }
        for (const py::handle value : edges) {
            const auto edge = py::reinterpret_borrow<py::sequence>(value);
            if (edge.size() != 3) throw py::value_error("each force edge must be a (source, target, ideal_length) triple");
            source.push_back(py::cast<int>(edge[0])); target.push_back(py::cast<int>(edge[1])); length.push_back(py::cast<double>(edge[2]));
        }
        const FmmmExactForceStepDiagnostic result = fmmm_full_path_exact_force_step(host_positions, source, target, length,
            box_length, py::cast<double>(down_left_corner[0]), py::cast<double>(down_left_corner[1]), average_ideal_length, iteration);
        py::dict out;
        out["provider"] = "CUDA_FMMM_FULL_PATH_LEVEL_ENGINE";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["status"] = result.status == 0 ? "OK" : "UNSUPPORTED_INPUT";
        if (result.status == 0) {
            out["adjusted_position"] = result.adjusted_positions;
            out["attraction"] = result.attraction;
            out["repulsion"] = result.repulsion;
            out["combined_and_capped_force"] = result.capped_force;
            out["moved_position"] = result.moved_positions;
            py::dict box_after;
            box_after["length"] = result.box_length;
            box_after["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y);
            out["box_after"] = box_after;
        } else {
            out["adjusted_position"] = py::list(); out["attraction"] = py::list(); out["repulsion"] = py::list();
            out["combined_and_capped_force"] = py::list(); out["moved_position"] = py::list(); out["box_after"] = py::none();
        }
        return out;
    }, py::arg("positions"), py::arg("edges"), py::kw_only(), py::arg("box_length"),
       py::arg("down_left_corner"), py::arg("average_ideal_length"), py::arg("iteration") = 1);

    module.def("fmmm_full_path_exact_force_loop", [](const py::iterable& positions,
                                                         const py::iterable& edges, double box_length,
                                                         const py::sequence& down_left_corner,
                                                         double average_ideal_length, int iterations) {
        if (down_left_corner.size() != 2) throw py::value_error("down_left_corner must be an (x, y) pair");
        std::vector<std::pair<double, double>> host_positions; std::vector<int> source, target; std::vector<double> length;
        for (const py::handle value : positions) { const auto point = py::reinterpret_borrow<py::sequence>(value); if (point.size() != 2) throw py::value_error("each position must be an (x, y) pair"); host_positions.emplace_back(py::cast<double>(point[0]), py::cast<double>(point[1])); }
        for (const py::handle value : edges) { const auto edge = py::reinterpret_borrow<py::sequence>(value); if (edge.size() != 3) throw py::value_error("each force edge must be a (source, target, ideal_length) triple"); source.push_back(py::cast<int>(edge[0])); target.push_back(py::cast<int>(edge[1])); length.push_back(py::cast<double>(edge[2])); }
        const FmmmExactForceLoopDiagnostic result = fmmm_full_path_exact_force_loop(host_positions, source, target, length, box_length, py::cast<double>(down_left_corner[0]), py::cast<double>(down_left_corner[1]), average_ideal_length, iterations);
        py::dict out; out["provider"] = "CUDA_FMMM_FULL_PATH_LEVEL_ENGINE"; out["generation_device"] = "CUDA"; out["timed"] = false; out["status"] = result.status == 0 ? "OK" : "UNSUPPORTED_INPUT";
        if (result.status == 0) { out["final_position"] = result.final_positions; py::dict box; box["length"] = result.box_length; box["down_left_corner"] = py::make_tuple(result.down_left_x, result.down_left_y); out["box_after"] = box; }
        else { out["final_position"] = py::list(); out["box_after"] = py::none(); }
        return out;
    }, py::arg("positions"), py::arg("edges"), py::kw_only(), py::arg("box_length"),
       py::arg("down_left_corner"), py::arg("average_ideal_length"), py::arg("iterations") = 100);

    module.def("fmmm_diagnostic_trace", [](const py::iterable& nodes, const py::iterable& edges,
                                               std::int64_t width, std::int64_t height,
                                               std::uint64_t seed, int iterations, bool layers_enabled) {
        GraphSpec spec = parse_graph(nodes, edges, width, height);
        if (width < 3 || height < 3 || iterations < 0 || iterations > 3 ||
            spec.nodes.size() > 32 || spec.edges.size() > 256) {
            throw py::value_error("FMMM diagnostic trace is bounded to interior bounds, 0..3 passes, 32 nodes, and 256 edges");
        }
        DeviceGraph graph(std::move(spec));
        graph.upload(false);
        FmmmDiagnosticTrace trace;
        cuda_fmmm_generate(graph, seed, iterations, layers_enabled, &trace);
        const ScoreResult score = evaluate_full_square(graph, layers_enabled);
        py::dict out;
        out["provider"] = "CUDA_FMMM_DIAGNOSTIC";
        out["generation_device"] = "CUDA";
        out["timed"] = false;
        out["reference_profile"] = kFmmmReferenceProfile;
        out["config_hash"] = kFmmmProfileHash;
        out["trace"] = fmmm_trace_dict(trace);
        out["status"] = score.valid ? "OK" : "REJECTED_BY_ADMISSION";
        if (score.valid) {
            out["score"] = score_dict(score);
            out["coordinates"] = download_fmmm_coordinates(graph);
        } else {
            out["score"] = py::none();
            out["coordinates"] = py::list();
        }
        return out;
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(), py::arg("max_width"), py::arg("max_height"),
       py::arg("seed"), py::arg("iterations") = 3, py::arg("layers_enabled") = true);

    module.def("fmmm_seed_batch", [](const py::iterable& layouts, const py::iterable& edges, std::int64_t width, std::int64_t height,
                                       std::uint64_t seed, const std::string& config, int iterations, bool layers_enabled) {
        // Single Python invocation. Default path is device multi-stream
        // concurrent sessions (seed+offset per lane). Set
        // LCN_FMMM_BATCH_MODE=shared_session for upload/normalize amortization
        // on one reused session. Seed derivation remains index-based.
        return fmmm_run_batch(layouts, edges, width, height, seed, config, iterations,
                              layers_enabled);
    }, py::arg("layouts"), py::arg("edges"), py::kw_only(), py::arg("max_width"), py::arg("max_height"), py::arg("seed"), py::arg("config") = kFmmmReferenceProfile, py::arg("iterations") = 0, py::arg("layers_enabled") = true);

    module.def("fmmm_finalize_ogdf_profile", [](const py::iterable& nodes, const py::iterable& edges,
                                                const py::iterable& raw_x_it, const py::iterable& raw_y_it,
                                                std::int64_t width, std::int64_t height, bool layers_enabled) {
        GraphSpec spec = parse_graph(nodes, edges, width, height);
        DeviceGraph graph(std::move(spec));
        graph.upload(false);
        
        std::vector<double> h_raw_x, h_raw_y;
        for (const py::handle value : raw_x_it) h_raw_x.push_back(py::cast<double>(value));
        for (const py::handle value : raw_y_it) h_raw_y.push_back(py::cast<double>(value));
        
        if (h_raw_x.size() != static_cast<std::size_t>(graph.node_count()) || h_raw_y.size() != h_raw_x.size()) {
            throw py::value_error("raw_x and raw_y size must match node count");
        }
        
        DeviceBuffer<double> raw_x_buf(h_raw_x.size());
        DeviceBuffer<double> raw_y_buf(h_raw_y.size());
        if (!h_raw_x.empty()) {
            raw_x_buf.upload(h_raw_x.data(), h_raw_x.size());
            raw_y_buf.upload(h_raw_y.data(), h_raw_y.size());
        }
        
        FmmmFinalResult result = fmmm_finalize_ogdf_profile(graph, raw_x_buf, raw_y_buf, width, height, layers_enabled);
        
        py::dict out;
        if (result.status == FmmmFinalStatus::ok) {
            out["status"] = "OK";
            out["reason"] = result.reason;
            out["coordinates"] = result.coordinates;
            out["score"] = score_dict(result.score);
            out["stage"] = "COMMIT";
            out["first_bad_node"] = -1;
            out["admission_authority"] = "CONTINUOUS_CROSSINGS_V1";
        } else if (result.status == FmmmFinalStatus::invalid_input) {
            out["status"] = "INVALID_INPUT";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
            out["stage"] = "AFFINE_AND_VALIDATE";
            out["first_bad_node"] = result.first_bad_node;
        } else if (result.status == FmmmFinalStatus::rejected_by_admission) {
            out["status"] = "REJECTED_BY_ADMISSION";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
            out["stage"] = "SCORE";
            out["first_bad_node"] = -1;
        } else {
            out["status"] = "BACKEND_ERROR";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
            out["stage"] = "ALLOCATION";
            out["first_bad_node"] = -1;
        }
        return out;
    }, py::arg("nodes"), py::arg("edges"), py::arg("raw_x"), py::arg("raw_y"),
       py::arg("width"), py::arg("height"), py::arg("layers_enabled"));

    module.def("fmmm_finalize_ogdf_profile_integer", [](const py::iterable& nodes, const py::iterable& edges,
                                                        const py::iterable& raw_x_it, const py::iterable& raw_y_it,
                                                        std::int64_t width, std::int64_t height, bool layers_enabled) {
        GraphSpec spec = parse_graph(nodes, edges, width, height);
        DeviceGraph graph(std::move(spec));
        graph.upload(false);
        std::vector<double> h_raw_x, h_raw_y;
        for (const py::handle value : raw_x_it) h_raw_x.push_back(py::cast<double>(value));
        for (const py::handle value : raw_y_it) h_raw_y.push_back(py::cast<double>(value));
        if (h_raw_x.size() != static_cast<std::size_t>(graph.node_count()) || h_raw_y.size() != h_raw_x.size()) {
            throw py::value_error("raw_x and raw_y size must match node count");
        }
        DeviceBuffer<double> raw_x_buf(h_raw_x.size());
        DeviceBuffer<double> raw_y_buf(h_raw_y.size());
        if (!h_raw_x.empty()) {
            raw_x_buf.upload(h_raw_x.data(), h_raw_x.size());
            raw_y_buf.upload(h_raw_y.data(), h_raw_y.size());
        }
        FmmmFinalResult result =
            fmmm_finalize_ogdf_profile_integer(graph, raw_x_buf, raw_y_buf, width, height, layers_enabled);
        py::dict out;
        if (result.status == FmmmFinalStatus::ok) {
            out["status"] = "OK";
            out["reason"] = result.reason;
            out["coordinates"] = result.coordinates;
            out["score"] = score_dict(result.score);
            out["admission_authority"] = "FULL_SQUARE_GPU";
        } else if (result.status == FmmmFinalStatus::invalid_input) {
            out["status"] = "INVALID_INPUT";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
            out["first_bad_node"] = result.first_bad_node;
        } else if (result.status == FmmmFinalStatus::rejected_by_admission) {
            out["status"] = "REJECTED_BY_ADMISSION";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
        } else {
            out["status"] = "BACKEND_ERROR";
            out["reason"] = result.reason;
            out["coordinates"] = py::none();
            out["score"] = py::none();
        }
        return out;
    }, py::arg("nodes"), py::arg("edges"), py::arg("raw_x"), py::arg("raw_y"),
       py::arg("width"), py::arg("height"), py::arg("layers_enabled"));

    module.def("fmmm_workspace_lifecycle_probe", [](std::size_t node_count, std::size_t edge_count, std::size_t level_count, std::size_t scale) {
        lcn::release::FmmmWorkspaceShape shape;
        shape.node_count = node_count;
        shape.edge_count = edge_count;
        shape.level_count = level_count;
        shape.cell_capacity = scale;
        shape.max_bytes = 0;
        shape.max_level_nodes = scale;
        shape.max_level_edges = scale;
        shape.max_aux_entries = scale;
        shape.low_adj_count = scale;
        shape.high_adj_count = scale;
        shape.lambda_count = scale;
        shape.neighbour_count = scale;
        shape.moon_count = scale;
        
        lcn::release::FmmmWorkspace workspace;
        auto create_status = lcn::release::FmmmWorkspace::create(shape, workspace);
        
        int create_status_int = static_cast<int>(create_status);
        bool view_valid_after_create = (workspace.view().position_x != nullptr);
        
        auto reset_status = workspace.reset();
        int reset_status_int = static_cast<int>(reset_status);
        
        workspace.release();
        bool view_null_after_release = (workspace.view().position_x == nullptr);
        
        py::dict out;
        out["create_status"] = create_status_int;
        out["reset_status"] = reset_status_int;
        out["view_valid_after_create"] = view_valid_after_create;
        out["view_null_after_release"] = view_null_after_release;
        return out;
    }, py::arg("node_count"), py::arg("edge_count"), py::arg("level_count"), py::arg("scale"));

    module.def("fmmm_is_single_level_route", [](int node_count, int min_graph_size) {
        return lcn::release::fmmm_is_single_level_route(node_count, min_graph_size);
    }, py::arg("node_count"), py::arg("min_graph_size") = lcn::release::kFmmmDefaultMinGraphSize);

    module.def("fmmm_hierarchy_reject_if_single_level", [](int node_count, int min_graph_size) {
        lcn::release::FmmmStageResult gate =
            lcn::release::fmmm_hierarchy_reject_if_single_level(node_count, min_graph_size);
        py::dict out;
        out["status"] = static_cast<int>(gate.status);
        out["completed_stage"] = static_cast<int>(gate.completed_stage);
        out["reason"] = gate.reason;
        return out;
    }, py::arg("node_count"), py::arg("min_graph_size") = lcn::release::kFmmmDefaultMinGraphSize);

    module.def("fmmm_run_single_level_diagnostic", [](int node_count,
                                                      const py::iterable& edge_source_it,
                                                      const py::iterable& edge_target_it,
                                                      const py::iterable& edge_length_it,
                                                      const py::iterable& initial_x_it,
                                                      const py::iterable& initial_y_it,
                                                      const py::dict& profile_dict,
                                                      const py::dict& state_dict,
                                                      bool provide_staging,
                                                      bool provide_force_scratch) {
        const bool measure_cuda_events = profile_dict.contains("_cuda_event_timing")
                && py::cast<bool>(profile_dict["_cuda_event_timing"]);
        cudaEvent_t event_e2e_start = nullptr;
        cudaEvent_t event_device_start = nullptr;
        cudaEvent_t event_device_end = nullptr;
        cudaEvent_t event_e2e_end = nullptr;
        if (measure_cuda_events) {
            // The diagnostic path launches on the legacy/default stream.  Keep
            // both event pairs on that same stream so the measurements are
            // ordered with the existing G1 work; no production stream or
            // algorithm semantics are changed.
            cuda_check(cudaEventCreate(&event_e2e_start), "create G1 e2e start event");
            cuda_check(cudaEventCreate(&event_device_start), "create G1 device start event");
            cuda_check(cudaEventCreate(&event_device_end), "create G1 device end event");
            cuda_check(cudaEventCreate(&event_e2e_end), "create G1 e2e end event");
            cuda_check(cudaEventRecord(event_e2e_start, nullptr), "record G1 e2e start event");
        }
        std::vector<int> h_source, h_target;
        std::vector<double> h_length, h_x, h_y;
        for (const py::handle value : edge_source_it) h_source.push_back(py::cast<int>(value));
        for (const py::handle value : edge_target_it) h_target.push_back(py::cast<int>(value));
        for (const py::handle value : edge_length_it) h_length.push_back(py::cast<double>(value));
        for (const py::handle value : initial_x_it) h_x.push_back(py::cast<double>(value));
        for (const py::handle value : initial_y_it) h_y.push_back(py::cast<double>(value));

        if (h_source.size() != h_target.size() || h_source.size() != h_length.size()) throw py::value_error("Edge arrays size mismatch");
        if (node_count > 0 && (h_x.size() != static_cast<std::size_t>(node_count) || h_y.size() != h_x.size())) {
            throw py::value_error("Node coordinate size mismatch");
        }

        DeviceBuffer<int> d_source(h_source.size());
        DeviceBuffer<int> d_target(h_target.size());
        DeviceBuffer<double> d_length(h_length.size());
        if (!h_source.empty()) {
            d_source.upload(h_source.data(), h_source.size());
            d_target.upload(h_target.data(), h_target.size());
            d_length.upload(h_length.data(), h_length.size());
        }

        const int alloc_nodes = std::max(node_count, 0);
        std::vector<int> h_original_id(static_cast<std::size_t>(alloc_nodes));
        for (int i = 0; i < alloc_nodes; ++i) h_original_id[static_cast<std::size_t>(i)] = i;
        DeviceBuffer<int> d_original_id(static_cast<std::size_t>(alloc_nodes));
        if (alloc_nodes > 0) d_original_id.upload(h_original_id.data(), static_cast<std::size_t>(alloc_nodes));

        lcn::release::FmmmGraphView view;
        view.original_node_id = d_original_id.get();
        view.edge_source = d_source.get();
        view.edge_target = d_target.get();
        view.edge_length = d_length.get();
        view.node_count = node_count;
        view.edge_count = static_cast<int>(h_source.size());

        lcn::release::FmmmProfile profile;
        profile.normal_iterations = profile_dict.contains("normal_iterations") ? py::cast<int>(profile_dict["normal_iterations"]) : 0;
        profile.postprocess_iterations = profile_dict.contains("postprocess_iterations") ? py::cast<int>(profile_dict["postprocess_iterations"]) : 10;
        profile.fine_tuning_iterations = profile_dict.contains("fine_tuning_iterations") ? py::cast<int>(profile_dict["fine_tuning_iterations"]) : 20;
        profile.act_level = profile_dict.contains("act_level") ? py::cast<int>(profile_dict["act_level"]) : 0;
        profile.max_level = profile_dict.contains("max_level") ? py::cast<int>(profile_dict["max_level"]) : 0;
        profile.average_ideal_length = profile_dict.contains("average_ideal_length") ? py::cast<double>(profile_dict["average_ideal_length"]) : 1.0;
        profile.initial_box_length = profile_dict.contains("initial_box_length") ? py::cast<double>(profile_dict["initial_box_length"]) : 100.0;
        profile.initial_down_left_x = profile_dict.contains("initial_down_left_x") ? py::cast<double>(profile_dict["initial_down_left_x"]) : 0.0;
        profile.initial_down_left_y = profile_dict.contains("initial_down_left_y") ? py::cast<double>(profile_dict["initial_down_left_y"]) : 0.0;
        profile.normal_spring_strength = profile_dict.contains("normal_spring_strength") ? py::cast<double>(profile_dict["normal_spring_strength"]) : 1.0;
        profile.normal_repulsion_strength = profile_dict.contains("normal_repulsion_strength") ? py::cast<double>(profile_dict["normal_repulsion_strength"]) : 1.0;
        profile.post_spring_strength = profile_dict.contains("post_spring_strength") ? py::cast<double>(profile_dict["post_spring_strength"]) : 1.0;
        profile.post_repulsion_strength = profile_dict.contains("post_repulsion_strength") ? py::cast<double>(profile_dict["post_repulsion_strength"]) : 1.0;
        profile.fine_spring_strength = profile_dict.contains("fine_spring_strength") ? py::cast<double>(profile_dict["fine_spring_strength"]) : 2.0;
        profile.fine_repulsion_strength = profile_dict.contains("fine_repulsion_strength") ? py::cast<double>(profile_dict["fine_repulsion_strength"]) : 0.2;
        profile.cool_value = profile_dict.contains("cool_value") ? py::cast<double>(profile_dict["cool_value"]) : 0.99;
        profile.fine_tune_scalar = profile_dict.contains("fine_tune_scalar") ? py::cast<double>(profile_dict["fine_tune_scalar"]) : 0.2;
        profile.cool_temperature = profile_dict.contains("cool_temperature") ? py::cast<bool>(profile_dict["cool_temperature"]) : false;
        profile.resize_drawing = profile_dict.contains("resize_drawing") ? py::cast<bool>(profile_dict["resize_drawing"]) : true;
        profile.apply_final_resize = profile_dict.contains("apply_final_resize") ? py::cast<bool>(profile_dict["apply_final_resize"]) : true;
        profile.apply_single_component_pack =
            profile_dict.contains("apply_single_component_pack")
                ? py::cast<bool>(profile_dict["apply_single_component_pack"])
                : true;
        profile.seed = profile_dict.contains("seed") ? py::cast<std::uint64_t>(profile_dict["seed"]) : 0;

        DeviceBuffer<double> d_x(static_cast<std::size_t>(alloc_nodes));
        DeviceBuffer<double> d_y(static_cast<std::size_t>(alloc_nodes));
        DeviceBuffer<double> d_force_x(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_force_y(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_attraction_x(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_attraction_y(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_repulsion_x(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_repulsion_y(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_last_movement_x(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<double> d_last_movement_y(provide_force_scratch ? static_cast<std::size_t>(alloc_nodes) : 0);
        DeviceBuffer<int> d_stage_status(provide_staging ? 1 : 0);
        DeviceBuffer<double> d_stage_box(provide_staging ? 1 : 0);
        DeviceBuffer<double> d_stage_left_x(provide_staging ? 1 : 0);
        DeviceBuffer<double> d_stage_left_y(provide_staging ? 1 : 0);

        if (alloc_nodes > 0) {
            d_x.upload(h_x.data(), static_cast<std::size_t>(alloc_nodes));
            d_y.upload(h_y.data(), static_cast<std::size_t>(alloc_nodes));
            if (provide_force_scratch) {
                cuda_check(cudaMemset(d_force_x.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset force_x");
                cuda_check(cudaMemset(d_force_y.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset force_y");
                cuda_check(cudaMemset(d_attraction_x.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset attraction_x");
                cuda_check(cudaMemset(d_attraction_y.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset attraction_y");
                cuda_check(cudaMemset(d_repulsion_x.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset repulsion_x");
                cuda_check(cudaMemset(d_repulsion_y.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset repulsion_y");
                cuda_check(cudaMemset(d_last_movement_x.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset last_movement_x");
                cuda_check(cudaMemset(d_last_movement_y.get(), 0, static_cast<std::size_t>(alloc_nodes) * sizeof(double)), "memset last_movement_y");
            }
        }

        const double initial_box = state_dict.contains("box_length") ? py::cast<double>(state_dict["box_length"]) : 100.0;
        const double initial_left_x = state_dict.contains("down_left_x") ? py::cast<double>(state_dict["down_left_x"]) : 0.0;
        const double initial_left_y = state_dict.contains("down_left_y") ? py::cast<double>(state_dict["down_left_y"]) : 0.0;
        const double initial_resize = state_dict.contains("resize_scale") ? py::cast<double>(state_dict["resize_scale"]) : 1.0;
        const double initial_cooling = state_dict.contains("cooling") ? py::cast<double>(state_dict["cooling"]) : 1.0;
        const int initial_iteration = state_dict.contains("iteration") ? py::cast<int>(state_dict["iteration"]) : 0;
        const int initial_phase = state_dict.contains("phase") ? py::cast<int>(state_dict["phase"]) : 0;

        lcn::release::FmmmLevelState state;
        state.position_x = d_x.get();
        state.position_y = d_y.get();
        state.force_x = provide_force_scratch ? d_force_x.get() : nullptr;
        state.force_y = provide_force_scratch ? d_force_y.get() : nullptr;
        state.attraction_x = provide_force_scratch ? d_attraction_x.get() : nullptr;
        state.attraction_y = provide_force_scratch ? d_attraction_y.get() : nullptr;
        state.repulsion_x = provide_force_scratch ? d_repulsion_x.get() : nullptr;
        state.repulsion_y = provide_force_scratch ? d_repulsion_y.get() : nullptr;
        state.last_movement_x = provide_force_scratch ? d_last_movement_x.get() : nullptr;
        state.last_movement_y = provide_force_scratch ? d_last_movement_y.get() : nullptr;
        state.stage_status = provide_staging ? d_stage_status.get() : nullptr;
        state.stage_box = provide_staging ? d_stage_box.get() : nullptr;
        state.stage_left_x = provide_staging ? d_stage_left_x.get() : nullptr;
        state.stage_left_y = provide_staging ? d_stage_left_y.get() : nullptr;
        state.box_length = initial_box;
        state.down_left_x = initial_left_x;
        state.down_left_y = initial_left_y;
        state.resize_scale = initial_resize;
        state.cooling = initial_cooling;
        state.iteration = initial_iteration;
        state.phase = initial_phase;
        state.node_count = node_count;

        if (measure_cuda_events) {
            cuda_check(cudaEventRecord(event_device_start, nullptr), "record G1 device start event");
        }
        lcn::release::FmmmStageResult result = lcn::release::fmmm_run_single_level(state, view, profile);
        if (measure_cuda_events) {
            cuda_check(cudaEventRecord(event_device_end, nullptr), "record G1 device end event");
            cuda_check(cudaEventSynchronize(event_device_end), "synchronize G1 device end event");
        }

        if (alloc_nodes > 0) {
            d_x.download(h_x.data(), static_cast<std::size_t>(alloc_nodes));
            d_y.download(h_y.data(), static_cast<std::size_t>(alloc_nodes));
        }

        py::dict out;
        out["status"] = static_cast<int>(result.status);
        out["completed_stage"] = static_cast<int>(result.completed_stage);
        out["first_bad_node"] = result.first_bad_node;
        out["first_bad_level"] = result.first_bad_level;
        out["reason"] = result.reason;
        out["single_level_route"] = lcn::release::fmmm_is_single_level_route(node_count);

        py::dict out_state;
        out_state["box_length"] = state.box_length;
        out_state["down_left_x"] = state.down_left_x;
        out_state["down_left_y"] = state.down_left_y;
        out_state["resize_scale"] = state.resize_scale;
        out_state["cooling"] = state.cooling;
        out_state["iteration"] = state.iteration;
        out_state["phase"] = state.phase;
        out["state"] = out_state;

        // On invalid_input, boundary fields must remain unpublished (caller seed values).
        out["boundary_published"] = result.status == lcn::release::FmmmPipelineStatus::ok;
        out["seed_box_length"] = initial_box;
        out["seed_down_left_x"] = initial_left_x;
        out["seed_down_left_y"] = initial_left_y;

        py::list py_x, py_y;
        for (double val : h_x) py_x.append(val);
        for (double val : h_y) py_y.append(val);
        out["position_x"] = py_x;
        out["position_y"] = py_y;

        if (measure_cuda_events) {
            // The final download above is synchronous.  Recording the end
            // event after it gives a device-ordered diagnostic boundary that
            // includes the complete native call's CUDA work.  It is distinct
            // from the force/device interval, which starts immediately before
            // fmmm_run_single_level.
            cuda_check(cudaEventRecord(event_e2e_end, nullptr), "record G1 e2e end event");
            cuda_check(cudaEventSynchronize(event_e2e_end), "synchronize G1 e2e end event");
            float device_ms = 0.0F;
            float e2e_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&device_ms, event_device_start, event_device_end),
                       "measure G1 device event interval");
            cuda_check(cudaEventElapsedTime(&e2e_ms, event_e2e_start, event_e2e_end),
                       "measure G1 e2e event interval");
            out["cuda_event_device_ms"] = device_ms;
            out["cuda_event_end_to_end_ms"] = e2e_ms;
            out["cuda_event_device"] = gpu_identity();
            cudaEventDestroy(event_e2e_start);
            cudaEventDestroy(event_device_start);
            cudaEventDestroy(event_device_end);
            cudaEventDestroy(event_e2e_end);
        }

        return out;
    }, py::arg("node_count"), py::arg("edge_source"), py::arg("edge_target"), py::arg("edge_length"),
       py::arg("initial_x"), py::arg("initial_y"), py::arg("profile"), py::arg("state"),
       py::arg("provide_staging") = true, py::arg("provide_force_scratch") = true);

    module.def("fmmm_hierarchy_diagnostic", [](int node_count,
                                            const py::iterable& edge_source_it,
                                            const py::iterable& edge_target_it,
                                            const py::iterable& edge_length_it,
                                            const py::dict& profile_dict) {
        std::vector<int> h_source, h_target;
        std::vector<double> h_length;
        for (const auto& item : edge_source_it) h_source.push_back(py::cast<int>(item));
        for (const auto& item : edge_target_it) h_target.push_back(py::cast<int>(item));
        for (const auto& item : edge_length_it) h_length.push_back(py::cast<double>(item));

        // Frozen 20x20 BoundingCircle fine length used by Automatic-2 / hierarchy AB.
        // Callers may pass already-expanded ideal lengths; values of 1.0 are treated
        // as "uninitialized unit" and expanded to the frozen profile length.
        constexpr double kFrozenBoundingCircleLength = 48.284271247461902;
        for (double& length : h_length) {
            if (length == 1.0) length = kFrozenBoundingCircleLength;
        }

        lcn::release::FmmmGraphView view;
        view.node_count = node_count;
        view.edge_count = static_cast<int>(h_source.size());

        DeviceBuffer<int> d_source(view.edge_count), d_target(view.edge_count);
        DeviceBuffer<double> d_length(view.edge_count);
        if (view.edge_count > 0) {
            d_source.upload(h_source.data(), view.edge_count);
            d_target.upload(h_target.data(), view.edge_count);
            d_length.upload(h_length.data(), view.edge_count);
        }
        view.edge_source = d_source.get();
        view.edge_target = d_target.get();
        view.edge_length = d_length.get();

        lcn::release::FmmmProfile profile;
        profile.act_level = profile_dict.contains("act_level") ? py::cast<int>(profile_dict["act_level"]) : 0;
        profile.max_level = profile_dict.contains("max_level") ? py::cast<int>(profile_dict["max_level"]) : 0;
        profile.seed = profile_dict.contains("seed") ? py::cast<std::uint64_t>(profile_dict["seed"]) : 0;
        profile.normal_iterations =
            profile_dict.contains("normal_iterations") ? py::cast<int>(profile_dict["normal_iterations"]) : 3;
        profile.max_iter_factor =
            profile_dict.contains("max_iter_factor") ? py::cast<int>(profile_dict["max_iter_factor"]) : 10;
        profile.postprocess_iterations =
            profile_dict.contains("postprocess_iterations") ? py::cast<int>(profile_dict["postprocess_iterations"]) : 10;
        profile.fine_tuning_iterations =
            profile_dict.contains("fine_tuning_iterations") ? py::cast<int>(profile_dict["fine_tuning_iterations"]) : 20;
        profile.resize_drawing =
            profile_dict.contains("resize_drawing") ? py::cast<bool>(profile_dict["resize_drawing"]) : true;

        lcn::release::FmmmHierarchyState state;
        state.capacity = 32;
        std::vector<lcn::release::FmmmHierarchyLevel> h_levels(static_cast<std::size_t>(state.capacity));
        state.levels = h_levels.data();

        auto release_guard = [&state]() { lcn::release::fmmm_release_hierarchy(state); };

        lcn::release::FmmmStageResult build_res = lcn::release::fmmm_build_hierarchy(state, view, profile);
        if (build_res.status != lcn::release::FmmmPipelineStatus::ok) {
            release_guard();
            py::dict err;
            err["status"] = static_cast<int>(build_res.status);
            err["completed_stage"] = static_cast<int>(build_res.completed_stage);
            err["reason"] = build_res.reason;
            err["error"] = build_res.reason;
            err["hierarchy_refused"] =
                (build_res.reason != nullptr &&
                 std::string(build_res.reason).find("single-level") != std::string::npos);
            return err;
        }

        lcn::release::FmmmStageResult exec_res = lcn::release::fmmm_execute_hierarchy(state);
        if (exec_res.status != lcn::release::FmmmPipelineStatus::ok) {
            const int failed_levels = state.level_count;
            release_guard();
            py::dict err;
            err["status"] = static_cast<int>(exec_res.status);
            err["completed_stage"] = static_cast<int>(exec_res.completed_stage);
            err["reason"] = exec_res.reason;
            err["error"] = exec_res.reason;
            err["level_count"] = failed_levels;
            return err;
        }

        auto download_xy = [](const double* dx, const double* dy, int n) {
            py::list xs, ys;
            if (n <= 0 || dx == nullptr || dy == nullptr) return std::make_pair(xs, ys);
            std::vector<double> hx(static_cast<std::size_t>(n)), hy(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(hx.data(), dx, static_cast<std::size_t>(n) * sizeof(double),
                                  cudaMemcpyDeviceToHost), "hierarchy diag px");
            cuda_check(cudaMemcpy(hy.data(), dy, static_cast<std::size_t>(n) * sizeof(double),
                                  cudaMemcpyDeviceToHost), "hierarchy diag py");
            for (double v : hx) xs.append(v);
            for (double v : hy) ys.append(v);
            return std::make_pair(xs, ys);
        };
        auto download_scalar = [](const double* p) {
            double v = 0.0;
            if (p) cuda_check(cudaMemcpy(&v, p, sizeof(double), cudaMemcpyDeviceToHost), "hierarchy diag scalar");
            return v;
        };
        auto download_ints = [](const int* p, int n) {
            py::list out;
            if (n <= 0 || p == nullptr) return out;
            std::vector<int> h(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(h.data(), p, static_cast<std::size_t>(n) * sizeof(int),
                                  cudaMemcpyDeviceToHost), "hierarchy diag ints");
            for (int v : h) out.append(v);
            return out;
        };
        auto download_int_scalar = [](const int* p) {
            int v = 0;
            if (p) cuda_check(cudaMemcpy(&v, p, sizeof(int), cudaMemcpyDeviceToHost), "hierarchy diag int");
            return v;
        };

        const int coarsest_n = state.levels[state.level_count - 1].plan.lower_node_count;
        const int placement_draws = 2 * coarsest_n;

        py::list levels_out;
        for (int level = 0; level < state.level_count; ++level) {
            const auto& lvl = state.levels[level];
            const int n = lvl.plan.lower_node_count;
            const int hn = lvl.plan.high_node_count;
            auto xy = download_xy(lvl.state.position_x, lvl.state.position_y, n);
            py::dict row;
            row["level"] = lvl.level;
            row["lower_node_count"] = lvl.plan.lower_node_count;
            row["lower_edge_ordinal_span"] = lvl.plan.lower_edge_ordinal_span;
            row["high_node_count"] = lvl.plan.high_node_count;
            row["high_edge_ordinal_span"] = lvl.plan.high_edge_ordinal_span;
            row["prng_before"] = lvl.prng_before;
            row["prng_after"] = lvl.prng_after;
            row["solar_draws_consumed"] = static_cast<int>(lvl.prng_after);
            row["placement_draws_consumed"] =
                (level < state.level_count - 1) ? placement_draws : 0;
            if (lvl.state.consumed_draws != nullptr) {
                row["consumed_draws"] = download_int_scalar(lvl.state.consumed_draws);
            }
            row["position_x"] = xy.first;
            row["position_y"] = xy.second;
            row["box_length"] = download_scalar(lvl.state.box_length);
            row["down_left_x"] = download_scalar(lvl.state.down_left_x);
            row["down_left_y"] = download_scalar(lvl.state.down_left_y);
            // Membership / parent map (retained through execute release-after-consume).
            if (level < state.level_count - 1 && lvl.plan.lower_type != nullptr) {
                row["lower_type"] = download_ints(lvl.plan.lower_type, n);
                row["lower_dedicated_sun"] = download_ints(lvl.plan.lower_dedicated_sun, n);
                row["high_lower_node"] = download_ints(lvl.plan.high_lower_node, hn);
                if (lvl.plan.lambda_begin != nullptr) {
                    row["lambda_begin"] = download_ints(lvl.plan.lambda_begin, n + 1);
                    const int lambda_count = download_int_scalar(lvl.plan.lambda_begin + n);
                    if (lambda_count > 0 && lvl.plan.lambda_values != nullptr
                            && lvl.plan.neighbour_sun_nodes != nullptr) {
                        std::vector<double> hlam(static_cast<std::size_t>(lambda_count));
                        cuda_check(cudaMemcpy(hlam.data(), lvl.plan.lambda_values,
                                              static_cast<std::size_t>(lambda_count) * sizeof(double),
                                              cudaMemcpyDeviceToHost), "hierarchy diag lambda");
                        py::list lv;
                        for (double v : hlam) lv.append(v);
                        row["lambda_values"] = lv;
                        row["neighbour_sun_nodes"] =
                            download_ints(lvl.plan.neighbour_sun_nodes, lambda_count);
                    }
                }
            }
            if (level < state.level_count - 1) {
                // high_position holds Advanced-placement endpoint after execute.
                auto un = download_xy(lvl.plan.high_position_x, lvl.plan.high_position_y, n);
                row["uncoarsen_position_x"] = un.first;
                row["uncoarsen_position_y"] = un.second;
            }
            levels_out.append(row);
        }

        // Finest level (0) is the recursive hierarchy final before G1/G5.
        const auto& fine = state.levels[0];
        auto fine_xy = download_xy(fine.state.position_x, fine.state.position_y, fine.plan.lower_node_count);

        py::dict out;
        out["status"] = 0;
        out["completed_stage"] = static_cast<int>(exec_res.completed_stage);
        out["level_count"] = state.level_count;
        out["levels"] = levels_out;
        out["position_x"] = fine_xy.first;
        out["position_y"] = fine_xy.second;
        out["box_length"] = download_scalar(fine.state.box_length);
        out["down_left_x"] = download_scalar(fine.state.down_left_x);
        out["down_left_y"] = download_scalar(fine.state.down_left_y);
        out["placement_draws_consumed"] = placement_draws;

        if (state.level_count > 1) {
            const auto& coarse = state.levels[state.level_count - 1];
            auto coarse_xy = download_xy(coarse.state.position_x, coarse.state.position_y,
                                         coarse.plan.lower_node_count);
            out["coarsest_position_x"] = coarse_xy.first;
            out["coarsest_position_y"] = coarse_xy.second;
            out["coarsest_node_count"] = coarse.plan.lower_node_count;
            out["coarsest_box_length"] = download_scalar(coarse.state.box_length);
        }
        release_guard();
        return out;
    });

    module.def("fmmm_nmm_reuse_probe", [](int node_count, int iterations, std::uint64_t seed) {
        if (node_count < 175 || iterations <= 0) {
            throw std::invalid_argument("reuse probe requires node_count >= 175 and iterations > 0");
        }
        std::vector<double> hx(node_count), hy(node_count);
        for (int i = 0; i < node_count; ++i) {
            hx[i] = static_cast<double>(i % 17);
            hy[i] = static_cast<double>(i / 17);
        }
        auto alloc_d = [](std::size_t n) {
            double* p = nullptr;
            if (cudaMalloc(&p, n * sizeof(double)) != cudaSuccess) throw std::runtime_error("reuse probe cudaMalloc double");
            return p;
        };
        auto alloc_i = [](std::size_t n) {
            int* p = nullptr;
            if (cudaMalloc(&p, n * sizeof(int)) != cudaSuccess) throw std::runtime_error("reuse probe cudaMalloc int");
            return p;
        };
        double *x = alloc_d(node_count), *y = alloc_d(node_count);
        double *last_x = alloc_d(node_count), *last_y = alloc_d(node_count);
        double *attr_x = alloc_d(node_count), *attr_y = alloc_d(node_count);
        double *rep_x = alloc_d(node_count), *rep_y = alloc_d(node_count);
        double *out_box = alloc_d(1), *out_left_x = alloc_d(1), *out_left_y = alloc_d(1);
        int* status = alloc_i(1);
        cudaMemcpy(x, hx.data(), node_count * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(y, hy.data(), node_count * sizeof(double), cudaMemcpyHostToDevice);
        void* workspace = lcn::release::fmmm_nmm_workspace_create();
        if (!workspace) throw std::runtime_error("reuse probe workspace allocation");
        auto cleanup = [&]() {
            lcn::release::fmmm_nmm_workspace_release(workspace);
            cudaFree(x); cudaFree(y); cudaFree(last_x); cudaFree(last_y);
            cudaFree(attr_x); cudaFree(attr_y); cudaFree(rep_x); cudaFree(rep_y);
            cudaFree(out_box); cudaFree(out_left_x); cudaFree(out_left_y); cudaFree(status);
        };
        auto run = [&](int run_iterations, std::uint64_t run_seed) {
            return lcn::release::fmmm_run_nmm_force_loop(
                node_count, 0, nullptr, nullptr, nullptr, x, y, 10.0, 0.0, 0.0, 1.0,
                run_iterations, status, out_box, out_left_x, out_left_y, 0,
                last_x, last_y, attr_x, attr_y, rep_x, rep_y, run_seed, workspace);
        };
        auto first = run(iterations, seed);
        const int builds_after_first = lcn::release::fmmm_nmm_workspace_build_count(workspace);
        const int maps_after_first = lcn::release::fmmm_nmm_workspace_map_build_count(workspace);
        const int launches_after_first = lcn::release::fmmm_nmm_workspace_map_launch_count(workspace);
        auto second = run(iterations, seed);
        const int builds_after_same = lcn::release::fmmm_nmm_workspace_build_count(workspace);
        const int maps_after_same = lcn::release::fmmm_nmm_workspace_map_build_count(workspace);
        const int launches_after_same = lcn::release::fmmm_nmm_workspace_map_launch_count(workspace);
        auto invalidated = run(iterations, seed + 1);
        const int builds_after_epoch = lcn::release::fmmm_nmm_workspace_build_count(workspace);
        const int maps_after_epoch = lcn::release::fmmm_nmm_workspace_map_build_count(workspace);
        const int launches_after_epoch = lcn::release::fmmm_nmm_workspace_map_launch_count(workspace);
        py::dict out;
        out["first_status"] = static_cast<int>(first.status);
        out["same_status"] = static_cast<int>(second.status);
        out["epoch_status"] = static_cast<int>(invalidated.status);
        out["builds_after_first"] = builds_after_first;
        out["map_builds_after_first"] = maps_after_first;
        out["map_launches_after_first"] = launches_after_first;
        out["builds_after_same"] = builds_after_same;
        out["map_builds_after_same"] = maps_after_same;
        out["map_launches_after_same"] = launches_after_same;
        out["builds_after_epoch"] = builds_after_epoch;
        out["map_builds_after_epoch"] = maps_after_epoch;
        out["map_launches_after_epoch"] = launches_after_epoch;
        cleanup();
        return out;
    });

    module.def("fmmm_nmm_force", [](py::dict args) {
        lcn::release::FmmmNmmState state;
        lcn::release::FmmmNmmOptions options;

        state.node_count = args.contains("node_count") ? py::cast<int>(args["node_count"]) : 0;
        state.cell_count = args.contains("cell_count") ? py::cast<int>(args["cell_count"]) : 0;
        state.precision = args.contains("precision") ? py::cast<int>(args["precision"]) : 4;
        state.expansion_stride = args.contains("expansion_stride")
                ? py::cast<int>(args["expansion_stride"])
                : (state.precision + 1);
        options.using_nmm = args.contains("using_nmm") ? py::cast<bool>(args["using_nmm"]) : true;
        options.box_length = args.contains("box_length") ? py::cast<double>(args["box_length"]) : 0.0;
        options.down_left_x = args.contains("down_left_x") ? py::cast<double>(args["down_left_x"]) : 0.0;
        options.down_left_y = args.contains("down_left_y") ? py::cast<double>(args["down_left_y"]) : 0.0;
        options.precision = state.precision;
        options.particles_in_leaf =
                args.contains("particles_in_leaf") ? py::cast<int>(args["particles_in_leaf"]) : 25;
        options.seed = args.contains("seed") ? py::cast<std::uint64_t>(args["seed"]) : 0;
        options.tree_construction = args.contains("tree_construction")
                ? static_cast<lcn::release::FmmmNmmTreeConstruction>(py::cast<int>(args["tree_construction"]))
                : lcn::release::FmmmNmmTreeConstruction::subtree_by_subtree;
        options.small_cell = args.contains("small_cell")
                ? static_cast<lcn::release::FmmmNmmSmallCell>(py::cast<int>(args["small_cell"]))
                : lcn::release::FmmmNmmSmallCell::iteratively;
        const bool build_tree = args.contains("build_tree") ? py::cast<bool>(args["build_tree"])
                                                            : (state.cell_count <= 0);

        if (state.node_count == 0) return py::dict();

        auto up_double = [&](const char* key, int n) -> double* {
            if (!args.contains(key) || n <= 0) return nullptr;
            std::vector<double> h = py::cast<std::vector<double>>(args[key]);
            double* d = nullptr; cudaMalloc(&d, h.size() * sizeof(double));
            cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            return d;
        };
        auto up_int = [&](const char* key, int n) -> int* {
            if (!args.contains(key) || n <= 0) return nullptr;
            std::vector<int> h = py::cast<std::vector<int>>(args[key]);
            int* d = nullptr; cudaMalloc(&d, h.size() * sizeof(int));
            cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
            return d;
        };
        auto alloc_double = [&](int n) -> double* {
            if (n <= 0) return nullptr;
            double* d = nullptr; cudaMalloc(&d, n * sizeof(double)); return d;
        };
        auto alloc_int = [&](int n) -> int* {
            if (n <= 0) return nullptr;
            int* d = nullptr; cudaMalloc(&d, n * sizeof(int)); return d;
        };

        state.position_x = up_double("position_x", state.node_count);
        state.position_y = up_double("position_y", state.node_count);
        state.force_x = args.contains("force_x") ? up_double("force_x", state.node_count)
                                                 : alloc_double(state.node_count);
        state.force_y = args.contains("force_y") ? up_double("force_y", state.node_count)
                                                 : alloc_double(state.node_count);
        if (state.force_x && !args.contains("force_x")) cudaMemset(state.force_x, 0, state.node_count * sizeof(double));
        if (state.force_y && !args.contains("force_y")) cudaMemset(state.force_y, 0, state.node_count * sizeof(double));

        if (!build_tree) {
            state.cell_center_x = up_double("cell_center_x", state.cell_count);
            state.cell_center_y = up_double("cell_center_y", state.cell_count);
            state.cell_particle_count = up_int("cell_particle_count", state.cell_count);
            state.cell_downleft_x = up_double("cell_downleft_x", state.cell_count);
            state.cell_downleft_y = up_double("cell_downleft_y", state.cell_count);
            state.cell_box_length = up_double("cell_box_length", state.cell_count);
            state.cell_level = up_int("cell_level", state.cell_count);
            state.cell_parent = up_int("cell_parent", state.cell_count);
            state.cell_child4 = up_int("cell_child4", state.cell_count * 4);

            int ic = args.contains("I_count") ? py::cast<int>(args["I_count"]) : 0;
            state.I_count = ic;
            state.I_begin = up_int("I_begin", state.cell_count + 1);
            state.I_nodes = up_int("I_nodes", ic);

            int d1c = args.contains("D1_count") ? py::cast<int>(args["D1_count"]) : 0;
            state.D1_count = d1c;
            state.D1_begin = up_int("D1_begin", state.cell_count + 1);
            state.D1_nodes = up_int("D1_nodes", d1c);

            int d2c = args.contains("D2_count") ? py::cast<int>(args["D2_count"]) : 0;
            state.D2_count = d2c;
            state.D2_begin = up_int("D2_begin", state.cell_count + 1);
            state.D2_nodes = up_int("D2_nodes", d2c);

            int mc = args.contains("M_count") ? py::cast<int>(args["M_count"]) : 0;
            state.M_count = mc;
            state.M_begin = up_int("M_begin", state.cell_count + 1);
            state.M_nodes = up_int("M_nodes", mc);

            int cc = args.contains("contained_node_count") ? py::cast<int>(args["contained_node_count"]) : 0;
            state.contained_node_count = cc;
            state.contained_begin = up_int("contained_begin", state.cell_count + 1);
            state.contained_nodes = up_int("contained_nodes", cc);

            int esc = state.cell_count * state.expansion_stride;
            state.multipole_re = up_double("multipole_re", esc);
            state.multipole_im = up_double("multipole_im", esc);
            state.local_re = up_double("local_re", esc);
            state.local_im = up_double("local_im", esc);
        } else {
            // Force package-owned tree construction.
            state.cell_count = 0;
        }

        // Snapshot pointers before call so we can free package-built arrays.
        const bool package_built = build_tree;
        lcn::release::FmmmStageResult res = lcn::release::fmmm_nmm_force(state, options);

        py::dict out;
        out["status"] = static_cast<int>(res.status);
        out["reason"] = res.reason;
        out["completed_stage"] = static_cast<int>(res.completed_stage);
        out["cell_count"] = state.cell_count;

        if (res.status == lcn::release::FmmmPipelineStatus::ok && state.force_x) {
            std::vector<double> hx(state.node_count), hy(state.node_count);
            cudaMemcpy(hx.data(), state.force_x, state.node_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hy.data(), state.force_y, state.node_count * sizeof(double), cudaMemcpyDeviceToHost);
            out["force_x"] = hx; out["force_y"] = hy;
        }

        auto download_d = [&](double* d, int n) {
            std::vector<double> h(n, 0.0);
            if (d && n > 0) cudaMemcpy(h.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost);
            return h;
        };
        auto download_i = [&](int* d, int n) {
            std::vector<int> h(n, 0);
            if (d && n > 0) cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
            return h;
        };

        if (res.status == lcn::release::FmmmPipelineStatus::ok && state.cell_count > 0) {
            out["cell_downleft_x"] = download_d(state.cell_downleft_x, state.cell_count);
            out["cell_downleft_y"] = download_d(state.cell_downleft_y, state.cell_count);
            out["cell_box_length"] = download_d(state.cell_box_length, state.cell_count);
            out["cell_center_x"] = download_d(state.cell_center_x, state.cell_count);
            out["cell_center_y"] = download_d(state.cell_center_y, state.cell_count);
            out["cell_level"] = download_i(state.cell_level, state.cell_count);
            out["cell_parent"] = download_i(state.cell_parent, state.cell_count);
            out["cell_child4"] = download_i(state.cell_child4, state.cell_count * 4);
            out["cell_particle_count"] = download_i(state.cell_particle_count, state.cell_count);
            out["contained_begin"] = download_i(state.contained_begin, state.cell_count + 1);
            out["contained_nodes"] = download_i(state.contained_nodes, state.contained_node_count);
            out["I_begin"] = download_i(state.I_begin, state.cell_count + 1);
            out["I_nodes"] = download_i(state.I_nodes, state.I_count);
            out["D1_begin"] = download_i(state.D1_begin, state.cell_count + 1);
            out["D1_nodes"] = download_i(state.D1_nodes, state.D1_count);
            out["D2_begin"] = download_i(state.D2_begin, state.cell_count + 1);
            out["D2_nodes"] = download_i(state.D2_nodes, state.D2_count);
            out["M_begin"] = download_i(state.M_begin, state.cell_count + 1);
            out["M_nodes"] = download_i(state.M_nodes, state.M_count);
            const int esc = state.cell_count * state.expansion_stride;
            out["multipole_re"] = download_d(state.multipole_re, esc);
            out["multipole_im"] = download_d(state.multipole_im, esc);
            out["local_re"] = download_d(state.local_re, esc);
            out["local_im"] = download_d(state.local_im, esc);
            out["package_built_tree"] = package_built;
        }

        if(state.position_x) cudaFree(state.position_x);
        if(state.position_y) cudaFree(state.position_y);
        if(state.force_x) cudaFree(state.force_x);
        if(state.force_y) cudaFree(state.force_y);
        if(state.cell_center_x) cudaFree(state.cell_center_x);
        if(state.cell_center_y) cudaFree(state.cell_center_y);
        if(state.cell_particle_count) cudaFree(state.cell_particle_count);
        if(state.cell_downleft_x) cudaFree(state.cell_downleft_x);
        if(state.cell_downleft_y) cudaFree(state.cell_downleft_y);
        if(state.cell_box_length) cudaFree(state.cell_box_length);
        if(state.cell_level) cudaFree(state.cell_level);
        if(state.cell_parent) cudaFree(state.cell_parent);
        if(state.cell_child4) cudaFree(state.cell_child4);
        if(state.I_begin) cudaFree(state.I_begin);
        if(state.I_nodes) cudaFree(state.I_nodes);
        if(state.D1_begin) cudaFree(state.D1_begin);
        if(state.D1_nodes) cudaFree(state.D1_nodes);
        if(state.D2_begin) cudaFree(state.D2_begin);
        if(state.D2_nodes) cudaFree(state.D2_nodes);
        if(state.M_begin) cudaFree(state.M_begin);
        if(state.M_nodes) cudaFree(state.M_nodes);
        if(state.contained_begin) cudaFree(state.contained_begin);
        if(state.contained_nodes) cudaFree(state.contained_nodes);
        if(state.multipole_re) cudaFree(state.multipole_re);
        if(state.multipole_im) cudaFree(state.multipole_im);
        if(state.local_re) cudaFree(state.local_re);
        if(state.local_im) cudaFree(state.local_im);
        if(state.particle_to_leaf) cudaFree(state.particle_to_leaf);

        return out;
    });

    module.def("fmmm_pack_components", [](py::dict args) {
        lcn::release::FmmmPackingState state;
        lcn::release::FmmmPackingOptions options;

        state.node_count = args.contains("node_count") ? py::cast<int>(args["node_count"]) : 0;
        state.component_count = args.contains("component_count") ? py::cast<int>(args["component_count"]) : 0;
        state.component_node_count = args.contains("component_node_count") ? py::cast<int>(args["component_node_count"]) : 0;

        options.page_ratio = args.contains("page_ratio") ? py::cast<double>(args["page_ratio"]) : 1.0;
        options.presort_components = args.contains("presort_components") ? static_cast<lcn::release::FmmmPackPreSort>(py::cast<int>(args["presort_components"])) : lcn::release::FmmmPackPreSort::decreasing_height;
        options.tip_over_components = args.contains("tip_over_components") ? static_cast<lcn::release::FmmmPackTipOver>(py::cast<int>(args["tip_over_components"])) : lcn::release::FmmmPackTipOver::no_growing_row;
        options.min_dist_cc = args.contains("min_dist_cc") ? py::cast<double>(args["min_dist_cc"]) : 30.0;

        if (state.node_count == 0 || state.component_count == 0) return py::dict();

        auto up_double = [&](const char* key, int n) -> double* {
            if (!args.contains(key) || n <= 0) return nullptr;
            std::vector<double> h = py::cast<std::vector<double>>(args[key]);
            double* d = nullptr; cudaMalloc(&d, h.size() * sizeof(double));
            cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            return d;
        };
        auto alloc_double = [&](int n) -> double* {
            double* d = nullptr; cudaMalloc(&d, n * sizeof(double)); return d;
        };
        auto up_int = [&](const char* key, int n) -> int* {
            if (!args.contains(key) || n <= 0) return nullptr;
            std::vector<int> h = py::cast<std::vector<int>>(args[key]);
            int* d = nullptr; cudaMalloc(&d, h.size() * sizeof(int));
            cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
            return d;
        };
        auto alloc_int = [&](int n) -> int* {
            int* d = nullptr; cudaMalloc(&d, n * sizeof(int)); return d;
        };

        state.position_x = up_double("position_x", state.node_count);
        state.position_y = up_double("position_y", state.node_count);
        state.node_width = up_double("node_width", state.node_count);
        state.node_height = up_double("node_height", state.node_count);

        state.component_begin = up_int("component_begin", state.component_count + 1);
        state.component_nodes = up_int("component_nodes", state.component_node_count);

        state.rect_left = alloc_double(state.component_count);
        state.rect_bottom = alloc_double(state.component_count);
        state.rect_right = alloc_double(state.component_count);
        state.rect_top = alloc_double(state.component_count);
        state.old_dlc_x = alloc_double(state.component_count);
        state.old_dlc_y = alloc_double(state.component_count);
        state.new_dlc_x = alloc_double(state.component_count);
        state.new_dlc_y = alloc_double(state.component_count);
        state.tipped = alloc_int(state.component_count);
        state.row_index = alloc_int(state.component_count);
        state.translate_x = alloc_double(state.component_count);
        state.translate_y = alloc_double(state.component_count);
        state.chosen_candidate = alloc_int(state.component_count);
        state.chosen_angle = alloc_double(state.component_count);

        lcn::release::FmmmStageResult res = lcn::release::fmmm_pack_components(state, options);

        py::dict out;
        out["status"] = static_cast<int>(res.status);
        out["reason"] = res.reason;
        out["completed_stage"] = static_cast<int>(res.completed_stage);

        if (res.status == lcn::release::FmmmPipelineStatus::ok) {
            std::vector<double> hx(state.node_count), hy(state.node_count);
            cudaMemcpy(hx.data(), state.position_x, state.node_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hy.data(), state.position_y, state.node_count * sizeof(double), cudaMemcpyDeviceToHost);
            out["position_x"] = hx; out["position_y"] = hy;

            std::vector<double> rleft(state.component_count), rbottom(state.component_count), rright(state.component_count), rtop(state.component_count);
            std::vector<double> oldx(state.component_count), oldy(state.component_count), newx(state.component_count), newy(state.component_count);
            std::vector<double> tx(state.component_count), ty(state.component_count), angles(state.component_count);
            std::vector<int> tip(state.component_count), row(state.component_count), cand(state.component_count);
            cudaMemcpy(rleft.data(), state.rect_left, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(rbottom.data(), state.rect_bottom, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(rright.data(), state.rect_right, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(rtop.data(), state.rect_top, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(oldx.data(), state.old_dlc_x, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(oldy.data(), state.old_dlc_y, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(newx.data(), state.new_dlc_x, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(newy.data(), state.new_dlc_y, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(tip.data(), state.tipped, state.component_count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(row.data(), state.row_index, state.component_count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(tx.data(), state.translate_x, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(ty.data(), state.translate_y, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(cand.data(), state.chosen_candidate, state.component_count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(angles.data(), state.chosen_angle, state.component_count * sizeof(double), cudaMemcpyDeviceToHost);

            out["rect_left"] = rleft; out["rect_bottom"] = rbottom;
            out["rect_right"] = rright; out["rect_top"] = rtop;
            out["old_dlc_x"] = oldx; out["old_dlc_y"] = oldy;
            out["new_dlc_x"] = newx; out["new_dlc_y"] = newy;
            out["tipped"] = tip; out["row_index"] = row;
            out["translate_x"] = tx; out["translate_y"] = ty;
            out["chosen_candidate"] = cand; out["chosen_angle"] = angles;
        }

        cudaFree(state.position_x); cudaFree(state.position_y);
        cudaFree(const_cast<double*>(state.node_width)); cudaFree(const_cast<double*>(state.node_height));
        cudaFree(const_cast<int*>(state.component_begin)); cudaFree(const_cast<int*>(state.component_nodes));
        cudaFree(state.rect_left); cudaFree(state.rect_bottom); cudaFree(state.rect_right); cudaFree(state.rect_top);
        cudaFree(state.old_dlc_x); cudaFree(state.old_dlc_y); cudaFree(state.new_dlc_x); cudaFree(state.new_dlc_y);
        cudaFree(state.tipped); cudaFree(state.row_index); cudaFree(state.translate_x); cudaFree(state.translate_y);
        cudaFree(state.chosen_candidate); cudaFree(state.chosen_angle);

        return out;
    });
    module.def("fmmm_session_create_binding", [](const py::iterable& nodes, const py::iterable& edges, std::int64_t width, std::int64_t height, std::uint64_t seed, const std::string& config, int iterations, bool layers_enabled) {
        lcn::release::RawFmmmInput input;
        for (const py::handle item : nodes) {
            const py::sequence point = py::reinterpret_borrow<py::sequence>(item);
            input.nodes_x.push_back(py::cast<double>(point[0]));
            input.nodes_y.push_back(py::cast<double>(point[1]));
        }
        for (const py::handle item : edges) {
            const py::sequence edge = py::reinterpret_borrow<py::sequence>(item);
            input.edges_source.push_back(py::cast<int>(edge[0]));
            input.edges_target.push_back(py::cast<int>(edge[1]));
            input.edges_layer.push_back(edge.size() > 2 ? py::cast<int>(edge[2]) : 0);
        }
        input.max_width = width;
        input.max_height = height;
        lcn::release::FmmmConfig conf;
        conf.seed = seed;
        conf.iterations = iterations;
        conf.layers_enabled = layers_enabled;
        conf.config_string = config;
        auto session = lcn::release::fmmm_session_create(input, conf);
        return reinterpret_cast<std::uintptr_t>(session);
    });
    module.def("fmmm_session_destroy_binding", [](std::uintptr_t session_ptr) {
        auto session = reinterpret_cast<lcn::release::DeviceFmmmSession*>(session_ptr);
        if (session) lcn::release::fmmm_session_destroy(*session);
    });

    module.def("fmmm_transfer_ledger_probe", []() {
        const auto input = make_resident_probe_input(2, 1, 32, 32);
        const auto config = make_resident_probe_config(7, 3);
        ResidentSessionGuard guard;
        guard.value = lcn::release::fmmm_session_create(input, config);
        if (!guard.value) {
            return resident_probe_error(lcn::release::fmmm_session_last_create_error());
        }
        if (run_resident_session_to_nmm(*guard.value) != lcn::release::FmmmDeviceStatus::ok ||
            lcn::release::fmmm_session_run_schedule(*guard.value) != lcn::release::FmmmDeviceStatus::ok ||
            lcn::release::fmmm_session_score_and_admit(*guard.value) != lcn::release::FmmmDeviceStatus::ok ||
            lcn::release::fmmm_session_publish_device(*guard.value) != lcn::release::FmmmDeviceStatus::ok) {
            return resident_probe_error("resident_transfer_probe_phase_failed");
        }
        const auto publication = lcn::release::fmmm_session_download_publication(*guard.value);
        py::dict out = resident_publication_telemetry(publication);
        out["status"] = publication.status;
        out["reason"] = publication.reason;
        out["phase_allocation_groups"] = std::max(0, guard.value->phase_allocation_groups - 1);
        out["phase_free_groups"] = guard.value->phase_free_groups;
        out["host_iteration_poll_groups"] = guard.value->host_iteration_poll_groups;
        return out;
    });

    module.def("fmmm_session_workspace_reuse_probe", [](int node_count, int edge_count,
                                                          int iterations, std::int64_t width,
                                                          std::int64_t height, std::uint64_t seed) {
        const auto input = make_resident_probe_input(node_count, edge_count, width, height);
        const auto config = make_resident_probe_config(seed, iterations);
        ResidentSessionGuard guard;
        guard.value = lcn::release::fmmm_session_create(input, config);
        if (!guard.value) return resident_probe_error(lcn::release::fmmm_session_last_create_error());
        if (run_resident_session_to_nmm(*guard.value) != lcn::release::FmmmDeviceStatus::ok)
            return resident_probe_error("resident_workspace_probe_setup_failed");

        const auto before_view = guard.value->workspace.view();
        const auto before_position_x = guard.value->position_x.d_ptr;
        const auto before_position_y = guard.value->position_y.d_ptr;
        const auto generation_before = guard.value->allocation_generation;
        const auto schedule_status = lcn::release::fmmm_session_run_schedule(*guard.value);
        const auto after_view = guard.value->workspace.view();
        const auto generation_after = guard.value->allocation_generation;

        py::dict out;
        out["status"] = schedule_status == lcn::release::FmmmDeviceStatus::ok ? "OK" : "BACKEND_ERROR";
        out["reason"] = schedule_status == lcn::release::FmmmDeviceStatus::ok
            ? "none" : "resident_workspace_probe_schedule_failed";
        // Session creation is the one allowed owner allocation. This reports
        // phase churn after that baseline, using the session's real counter.
        out["phase_allocation_groups"] = std::max(0, guard.value->phase_allocation_groups - 1);
        out["phase_free_groups"] = guard.value->phase_free_groups;
        out["allocation_generation_before_schedule"] = generation_before;
        out["allocation_generation_after_schedule"] = generation_after;
        out["workspace_ids_stable"] =
            schedule_status == lcn::release::FmmmDeviceStatus::ok &&
            generation_before == generation_after &&
            before_view.position_x == after_view.position_x &&
            before_view.position_y == after_view.position_y &&
            before_view.force_x == after_view.force_x &&
            before_view.force_y == after_view.force_y &&
            guard.value->position_x.d_ptr == before_position_x &&
            guard.value->position_y.d_ptr == before_position_y;
        return out;
    });

    module.def("fmmm_session_epoch_rebuild_probe", [](int node_count, int edge_count,
                                                        int iterations, std::int64_t width,
                                                        std::int64_t height, std::uint64_t seed) {
        // The NMM implementation selects its persistent device tree at 175
        // particles. Use at least that many for this probe so the epoch
        // invalidation path is exercised rather than short-circuited.
        const auto input = make_resident_probe_input(std::max(175, node_count), edge_count,
                                                     width, height);
        const auto config = make_resident_probe_config(seed, iterations);
        ResidentSessionGuard guard;
        guard.value = lcn::release::fmmm_session_create(input, config);
        if (!guard.value) return resident_probe_error(lcn::release::fmmm_session_last_create_error());
        if (run_resident_session_to_nmm(*guard.value) != lcn::release::FmmmDeviceStatus::ok)
            return resident_probe_error("resident_epoch_probe_setup_failed");

        lcn::release::FmmmNmmOptions options;
        options.box_length = static_cast<double>(std::max(width, height));
        options.precision = 4;
        options.particles_in_leaf = 8;
        options.using_nmm = true;
        options.seed = seed;
        const auto first = lcn::release::fmmm_nmm_force(guard.value->nmm_state, options);
        const auto tree_epoch_before = guard.value->nmm_state.tree_epoch;
        guard.value->nmm_state.tree_epoch = tree_epoch_before + 1;
        guard.value->nmm_state.map_epoch = tree_epoch_before + 1;
        const auto second = lcn::release::fmmm_nmm_force(guard.value->nmm_state, options);
        const auto tree_epoch_after = guard.value->nmm_state.tree_epoch;

        py::dict out;
        out["status"] = second.status == lcn::release::FmmmPipelineStatus::ok
            ? "OK" : "BACKEND_ERROR";
        out["reason"] = second.status == lcn::release::FmmmPipelineStatus::ok
            ? "none" : second.reason;
        out["host_poll_groups"] = guard.value->host_iteration_poll_groups;
        out["intermediate_d2h_groups"] = guard.value->intermediate_d2h_groups;
        out["tree_epoch_before"] = tree_epoch_before;
        out["tree_epoch_after"] = tree_epoch_after;
        out["rebuild_device_side"] =
            first.status == lcn::release::FmmmPipelineStatus::ok &&
            second.status == lcn::release::FmmmPipelineStatus::ok &&
            guard.value->nmm_state.tree_rebuilt != 0 && tree_epoch_after > tree_epoch_before;
        return out;
    });

    module.def("fmmm_session_failure_probe", [](const std::string& phase,
                                                  const std::string& fault) {
        if (phase != "score" || fault != "forced_backend_error") {
            return resident_probe_error("unsupported_failure_probe");
        }
        const auto input = make_resident_probe_input(2, 1, 32, 32);
        const auto config = make_resident_probe_config(7, 3);
        ResidentSessionGuard guard;
        guard.value = lcn::release::fmmm_session_create(input, config);
        if (!guard.value) return resident_probe_error(lcn::release::fmmm_session_last_create_error());
        if (run_resident_session_to_nmm(*guard.value) != lcn::release::FmmmDeviceStatus::ok)
            return resident_probe_error("resident_failure_probe_setup_failed");

        // This uses the real score ABI failure guard: score_complete makes a
        // second score submission an internal_error, which must remain a
        // bounded error result and never invoke the legacy CPU route.
        guard.value->score_complete = true;
        const auto status = lcn::release::fmmm_session_score_and_admit(*guard.value);
        const std::string reason = status == lcn::release::FmmmDeviceStatus::internal_error
            ? "forced_backend_error" : "failure_injection_not_triggered";
        py::dict out = resident_device_failure(
            guard.value, "BACKEND_ERROR", reason, config.seed, config.iterations,
            config.config_string, input.max_width, input.max_height);
        out["phase"] = phase;
        out["fault"] = fault;
        out["device_status"] = static_cast<int>(status);
        out["final_download_groups"] = guard.value->final_download_groups;
        return out;
    });
}

} // namespace lcn::oracle
