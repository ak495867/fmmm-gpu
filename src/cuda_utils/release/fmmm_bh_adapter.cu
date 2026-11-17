#include "fmmm_device_state.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace lcn::release {
namespace {

constexpr std::int64_t kExactCoordinateLimit = 1000000000LL;
constexpr int kProjectionMaximumEdges = 256;
constexpr int kProjectionMaximumNodes = 32;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Every pointer is a kernel argument through FmmmDeviceStateView. This is the
// required repair for the historical candidate, whose kernels dereferenced
// host globals and consequently failed NVCC device compilation.
__global__ void initialize_tree(FmmmDeviceStateView state) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < state.tree_capacity) {
        state.child[index * 4 + 0] = -1;
        state.child[index * 4 + 1] = -1;
        state.child[index * 4 + 2] = -1;
        state.child[index * 4 + 3] = -1;
        state.mass[index] = 0.0F;
        state.center_x[index] = 0.0F;
        state.center_y[index] = 0.0F;
        state.width[index] = 0.0F;
        state.ready[index] = 0;
    }
    if (index < state.node_count) {
        state.mass[index] = 1.0F;
        state.center_x[index] = state.position_x[index];
        state.center_y[index] = state.position_y[index];
        state.ready[index] = 1;
    }
    if (index == 0) *state.next_tree_node = state.node_count + 1;
}

// Deterministic root summary is intentionally a serial bootstrap primitive.
// It verifies the explicit-state ABI before concurrent quadtree insertion is
// promoted; it is not yet presented as a complete Barnes-Hut traversal.
__global__ void summarize_root(FmmmDeviceStateView state) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || state.node_count == 0) return;
    const int root = state.node_count;
    float mass = 0.0F, center_x = 0.0F, center_y = 0.0F;
    for (int node = 0; node < state.node_count; ++node) {
        mass += state.mass[node];
        center_x += state.center_x[node];
        center_y += state.center_y[node];
    }
    state.mass[root] = mass;
    state.center_x[root] = center_x / mass;
    state.center_y[root] = center_y / mass;
    state.width[root] = 1.0F;
    state.ready[root] = 1;
}

__device__ int quadrant(float x, float y, float center_x, float center_y) {
    return (x >= center_x ? 1 : 0) | (y >= center_y ? 2 : 0);
}

__device__ void clear_children(FmmmDeviceStateView state, int node) {
    state.child[node * 4] = state.child[node * 4 + 1] = -1;
    state.child[node * 4 + 2] = state.child[node * 4 + 3] = -1;
}

__device__ void initialize_internal(FmmmDeviceStateView state, int node, float center_x, float center_y, float width) {
    clear_children(state, node);
    state.mass[node] = 0.0F;
    state.center_x[node] = center_x;
    state.center_y[node] = center_y;
    state.width[node] = width;
    state.ready[node] = 0;
}

// One thread gives deterministic insertion and avoids the hidden global state
// / atomic protocol of the historical candidate. Parallel insertion can be a
// later optimization once this exact ownership protocol is qualified.
__global__ void build_tree(FmmmDeviceStateView state) {
    if (blockIdx.x || threadIdx.x || state.node_count == 0) return;
    float lo_x = state.position_x[0], hi_x = lo_x, lo_y = state.position_y[0], hi_y = lo_y;
    for (int body = 1; body < state.node_count; ++body) {
        lo_x = fminf(lo_x, state.position_x[body]); hi_x = fmaxf(hi_x, state.position_x[body]);
        lo_y = fminf(lo_y, state.position_y[body]); hi_y = fmaxf(hi_y, state.position_y[body]);
    }
    const int root = state.node_count;
    initialize_internal(state, root, (lo_x + hi_x) * 0.5F, (lo_y + hi_y) * 0.5F,
                        fmaxf(hi_x - lo_x, hi_y - lo_y) + 1.0e-3F);
    *state.next_tree_node = root + 1;
    for (int body = 0; body < state.node_count; ++body) {
        int cursor = root;
        while (true) {
            const int q = quadrant(state.position_x[body], state.position_y[body], state.center_x[cursor], state.center_y[cursor]);
            const int location = cursor * 4 + q;
            const int occupant = state.child[location];
            if (occupant == -1) { state.child[location] = body; break; }
            if (occupant >= state.node_count) { cursor = occupant; continue; }
            if (*state.next_tree_node >= state.tree_capacity || state.width[cursor] < 1.0e-5F) {
                // Coincident/min-cell nodes remain represented by the first
                // leaf. The later exact full-square admission remains
                // authority for the generated integer layout.
                break;
            }
            const int internal = (*state.next_tree_node)++;
            const float child_width = state.width[cursor] * 0.5F;
            const float offset = child_width * 0.25F;
            const float internal_x = state.center_x[cursor] + ((q & 1) ? offset : -offset);
            const float internal_y = state.center_y[cursor] + ((q & 2) ? offset : -offset);
            initialize_internal(state, internal, internal_x, internal_y, child_width);
            state.child[location] = internal;
            const int old_q = quadrant(state.position_x[occupant], state.position_y[occupant], internal_x, internal_y);
            state.child[internal * 4 + old_q] = occupant;
            cursor = internal;
        }
    }
}

__global__ void summarize_tree(FmmmDeviceStateView state, int rounds) {
    if (blockIdx.x || threadIdx.x) return;
    for (int round = 0; round < rounds; ++round) {
        bool changed = false;
        for (int node = state.node_count; node < *state.next_tree_node; ++node) {
            if (state.ready[node]) continue;
            float mass = 0.0F, center_x = 0.0F, center_y = 0.0F; bool children_ready = true;
            for (int q = 0; q < 4; ++q) { const int child = state.child[node * 4 + q]; if (child < 0) continue; if (!state.ready[child]) { children_ready = false; break; } mass += state.mass[child]; center_x += state.mass[child] * state.center_x[child]; center_y += state.mass[child] * state.center_y[child]; }
            if (children_ready && mass > 0.0F) { state.mass[node] = mass; state.center_x[node] = center_x / mass; state.center_y[node] = center_y / mass; state.ready[node] = 1; changed = true; }
        }
        if (!changed || state.ready[state.node_count]) break;
    }
}

__global__ void compute_repulsion(FmmmDeviceStateView state, float theta) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= state.node_count) return;
    int stack[64]; int top = 0; stack[top++] = state.node_count; float fx = 0.0F, fy = 0.0F;
    while (top) {
        const int node = stack[--top]; if (node == body || state.mass[node] == 0.0F) continue;
        const float dx = state.position_x[body] - state.center_x[node], dy = state.position_y[body] - state.center_y[node];
        const float distance_squared = dx * dx + dy * dy + 1.0e-6F;
        const bool leaf = node < state.node_count;
        if (leaf || state.width[node] / sqrtf(distance_squared) < theta) { const float scale = state.mass[node] / distance_squared; fx += scale * dx; fy += scale * dy; }
        else for (int q = 0; q < 4 && top < 64; ++q) { const int child = state.child[node * 4 + q]; if (child >= 0) stack[top++] = child; }
    }
    state.force_x[body] = fx; state.force_y[body] = fy;
}

// Attraction uses the pinned OGDF New model with unitEdgeLength=20:
// f_attr = log2(d / L) * d / L^3 * Delta. Each endpoint computes its own
// contribution, avoiding a shared-force atomic protocol and retaining
// deterministic per-node accumulation order.
__global__ void add_edge_attraction(FmmmDeviceStateView state, float weight) {
    const int body = blockIdx.x * blockDim.x + threadIdx.x;
    if (body >= state.node_count || state.edge_source == nullptr || state.edge_target == nullptr) return;
    float fx = 0.0F, fy = 0.0F;
    for (int edge = 0; edge < state.edge_count; ++edge) {
        const int source = state.edge_source[edge], target = state.edge_target[edge];
        const int other = source == body ? target : (target == body ? source : -1);
        if (other < 0 || other >= state.node_count) continue;
        const float dx = state.position_x[other] - state.position_x[body];
        const float dy = state.position_y[other] - state.position_y[body];
        constexpr float kIdealLength = 20.0F;
        constexpr float kIdealLengthCubed = kIdealLength * kIdealLength * kIdealLength;
        const float distance = sqrtf(dx * dx + dy * dy);
        if (distance > 0.0F) {
            const float scalar = weight * log2f(distance / kIdealLength) * distance / kIdealLengthCubed;
            fx += scalar * dx;
            fy += scalar * dy;
        }
    }
    if (state.attraction_x != nullptr && state.attraction_y != nullptr) {
        state.attraction_x[body] = fx; state.attraction_y[body] = fy;
    } else {
        state.force_x[body] += fx; state.force_y[body] += fy;
    }
}

// These predicates intentionally mirror oracle/full_square.cu. Keeping the
// projection in this adapter makes candidate generation and acceptance both
// device-owned while preserving the oracle's integer geometry semantics.
__device__ inline std::int64_t projection_x(const oracle::DeviceGraphView& graph,
                                            int node, int moved, std::int64_t x) {
    return node == moved ? x : graph.x[node];
}

__device__ inline std::int64_t projection_y(const oracle::DeviceGraphView& graph,
                                            int node, int moved, std::int64_t y) {
    return node == moved ? y : graph.y[node];
}

__device__ inline int projection_sign(std::int64_t value) {
    return (value > 0) - (value < 0);
}

__device__ inline std::int64_t projection_orientation(std::int64_t ax, std::int64_t ay,
                                                       std::int64_t bx, std::int64_t by,
                                                       std::int64_t cx, std::int64_t cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline bool projection_between(std::int64_t value, std::int64_t a,
                                          std::int64_t b) {
    return value >= (a < b ? a : b) && value <= (a < b ? b : a);
}

__device__ inline bool projection_on_segment(std::int64_t px, std::int64_t py,
                                             std::int64_t ax, std::int64_t ay,
                                             std::int64_t bx, std::int64_t by) {
    return projection_orientation(ax, ay, bx, by, px, py) == 0 &&
           projection_between(px, ax, bx) && projection_between(py, ay, by);
}

struct ProjectionObjective {
    std::int64_t K;
    std::int64_t nK;
    std::int64_t Phi;
    std::int64_t C;
    bool valid;
};

__device__ ProjectionObjective projection_score(const oracle::DeviceGraphView& graph,
                                                bool layers_enabled, int moved,
                                                std::int64_t moved_x,
                                                std::int64_t moved_y) {
    ProjectionObjective rejected{0, 0, 0, 0, false};
    if (graph.edge_count > kProjectionMaximumEdges || graph.max_width < 2 ||
        graph.max_height < 2 || graph.max_width > kExactCoordinateLimit ||
        graph.max_height > kExactCoordinateLimit) return rejected;
    for (int i = 0; i < graph.node_count; ++i) {
        const auto x = projection_x(graph, i, moved, moved_x);
        const auto y = projection_y(graph, i, moved, moved_y);
        if (x <= 0 || x >= graph.max_width || y <= 0 || y >= graph.max_height ||
            x < -kExactCoordinateLimit || x > kExactCoordinateLimit ||
            y < -kExactCoordinateLimit || y > kExactCoordinateLimit) return rejected;
        for (int j = 0; j < i; ++j) {
            if (x == projection_x(graph, j, moved, moved_x) &&
                y == projection_y(graph, j, moved, moved_y)) return rejected;
        }
    }
    for (int node = 0; node < graph.node_count; ++node) {
        for (int edge = 0; edge < graph.edge_count; ++edge) {
            const int a = graph.source[edge], b = graph.target[edge];
            if (node == a || node == b) continue;
            if (projection_on_segment(projection_x(graph, node, moved, moved_x),
                                      projection_y(graph, node, moved, moved_y),
                                      projection_x(graph, a, moved, moved_x),
                                      projection_y(graph, a, moved, moved_y),
                                      projection_x(graph, b, moved, moved_x),
                                      projection_y(graph, b, moved, moved_y))) return rejected;
        }
    }
    std::int64_t counts[kProjectionMaximumEdges];
    for (int edge = 0; edge < graph.edge_count; ++edge) counts[edge] = 0;
    for (int left = 0; left < graph.edge_count; ++left) {
        const int la = graph.source[left], lb = graph.target[left];
        const auto lax = projection_x(graph, la, moved, moved_x);
        const auto lay = projection_y(graph, la, moved, moved_y);
        const auto lbx = projection_x(graph, lb, moved, moved_x);
        const auto lby = projection_y(graph, lb, moved, moved_y);
        for (int right = left + 1; right < graph.edge_count; ++right) {
            const int ra = graph.source[right], rb = graph.target[right];
            if (la == ra || la == rb || lb == ra || lb == rb ||
                (layers_enabled && graph.layer[left] != graph.layer[right])) continue;
            const auto rax = projection_x(graph, ra, moved, moved_x);
            const auto ray = projection_y(graph, ra, moved, moved_y);
            const auto rbx = projection_x(graph, rb, moved, moved_x);
            const auto rby = projection_y(graph, rb, moved, moved_y);
            const auto o1 = projection_orientation(lax, lay, lbx, lby, rax, ray);
            const auto o2 = projection_orientation(lax, lay, lbx, lby, rbx, rby);
            const auto o3 = projection_orientation(rax, ray, rbx, rby, lax, lay);
            const auto o4 = projection_orientation(rax, ray, rbx, rby, lbx, lby);
            if (projection_sign(o1) * projection_sign(o2) < 0 &&
                projection_sign(o3) * projection_sign(o4) < 0) {
                ++counts[left]; ++counts[right];
            } else if ((o1 == 0 && projection_on_segment(rax, ray, lax, lay, lbx, lby)) ||
                       (o2 == 0 && projection_on_segment(rbx, rby, lax, lay, lbx, lby)) ||
                       (o3 == 0 && projection_on_segment(lax, lay, rax, ray, rbx, rby)) ||
                       (o4 == 0 && projection_on_segment(lbx, lby, rax, ray, rbx, rby))) {
                return rejected;
            }
        }
    }
    ProjectionObjective score{0, 0, 0, 0, true};
    for (int edge = 0; edge < graph.edge_count; ++edge) {
        if (counts[edge] > score.K) score.K = counts[edge];
    }
    for (int edge = 0; edge < graph.edge_count; ++edge) {
        if (counts[edge] == score.K) ++score.nK;
        score.Phi += counts[edge] * counts[edge];
        score.C += counts[edge];
    }
    score.C /= 2;
    return score;
}

__device__ bool projection_better(const ProjectionObjective& candidate,
                                  const ProjectionObjective& current) {
    if (!candidate.valid) return false;
    if (!current.valid) return true;
    if (candidate.K != current.K) return candidate.K < current.K;
    if (candidate.nK != current.nK) return candidate.nK < current.nK;
    if (candidate.Phi != current.Phi) return candidate.Phi < current.Phi;
    return candidate.C < current.C;
}

// Fixed candidate order is part of the Phase-1 reference profile. The grid
// probes establish global structure; the local probes retain the fine-level
// FMMM placement. A single thread deliberately preserves this exact order.
__global__ void full_square_project(oracle::DeviceGraphView graph,
                                    bool layers_enabled, int rounds) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || graph.node_count == 0 ||
        graph.node_count > kProjectionMaximumNodes ||
        graph.edge_count > kProjectionMaximumEdges) return;
    ProjectionObjective current = projection_score(graph, layers_enabled, -1, 0, 0);
    if (!current.valid) return;
    const int columns = 13;
    const int rows = 11;
    for (int round = 0; round < rounds; ++round) {
        for (int node = 0; node < graph.node_count; ++node) {
            ProjectionObjective best = current;
            std::int64_t best_x = graph.x[node], best_y = graph.y[node];
            for (int column = 0; column < columns; ++column) {
                const std::int64_t x = 1 + ((graph.max_width - 2) * (column + 1)) / (columns + 1);
                for (int row = 0; row < rows; ++row) {
                    const std::int64_t y = 1 + ((graph.max_height - 2) * (row + 1)) / (rows + 1);
                    const ProjectionObjective candidate = projection_score(graph, layers_enabled, node, x, y);
                    if (projection_better(candidate, best)) { best = candidate; best_x = x; best_y = y; }
                }
            }
            constexpr int offsets[] = {5, 15, 35, 70};
            for (const int delta : offsets) {
                const std::int64_t x0 = graph.x[node], y0 = graph.y[node];
                const std::int64_t candidate_x[] = {x0 + delta, x0 - delta, x0, x0,
                                                    x0 + delta, x0 - delta, x0 + delta, x0 - delta};
                const std::int64_t candidate_y[] = {y0, y0, y0 + delta, y0 - delta,
                                                    y0 + delta, y0 - delta, y0 - delta, y0 + delta};
                for (int candidate_index = 0; candidate_index < 8; ++candidate_index) {
                    const ProjectionObjective candidate = projection_score(
                        graph, layers_enabled, node, candidate_x[candidate_index], candidate_y[candidate_index]);
                    if (projection_better(candidate, best)) {
                        best = candidate; best_x = candidate_x[candidate_index]; best_y = candidate_y[candidate_index];
                    }
                }
            }
            if (best_x != graph.x[node] || best_y != graph.y[node]) {
                graph.x[node] = best_x;
                graph.y[node] = best_y;
                current = best;
            }
        }
    }
}

__device__ std::uint64_t repair_mix(std::uint64_t value) {
    value ^= value >> 30U; value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U; value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// An invalid integer quantization cannot enter the lexicographic projection
// loop. This bounded device-only bootstrap rebuilds a complete candidate in
// fixed seed/node/round order and retains it only when the exact same device
// geometry predicate accepts the whole layout.
__global__ void repair_invalid_projection(oracle::DeviceGraphView graph,
                                          bool layers_enabled, std::uint64_t seed) {
    if (blockIdx.x || threadIdx.x || graph.node_count == 0 ||
        graph.edge_count > kProjectionMaximumEdges) return;
    if (projection_score(graph, layers_enabled, -1, 0, 0).valid) return;
    for (std::uint64_t round = 0; round < 256ULL; ++round) {
        for (int node = 0; node < graph.node_count; ++node) {
            std::uint64_t value = repair_mix(seed ^ (round + 1ULL) * 0x9e3779b97f4a7c15ULL ^
                                              (static_cast<std::uint64_t>(node) + 1ULL) * 0xbf58476d1ce4e5b9ULL);
            graph.x[node] = 1 + static_cast<std::int64_t>(value % static_cast<std::uint64_t>(graph.max_width - 2));
            value = repair_mix(value);
            graph.y[node] = 1 + static_cast<std::int64_t>(value % static_cast<std::uint64_t>(graph.max_height - 2));
        }
        if (projection_score(graph, layers_enabled, -1, 0, 0).valid) return;
    }
}

} // namespace

void fmmm_bh_initialize(FmmmDeviceStateView state) {
    if (state.node_count < 0 || state.tree_capacity < state.node_count + 1 ||
        state.child == nullptr || state.mass == nullptr || state.center_x == nullptr ||
        state.center_y == nullptr || state.width == nullptr || state.ready == nullptr ||
        state.next_tree_node == nullptr || (state.node_count > 0 &&
        (state.position_x == nullptr || state.position_y == nullptr))) {
        throw std::invalid_argument("invalid explicit Barnes-Hut device state");
    }
    const int blocks = (state.tree_capacity + 127) / 128;
    initialize_tree<<<blocks, 128>>>(state);
    check(cudaGetLastError(), "launch explicit Barnes-Hut tree initialization");
}

void fmmm_bh_summarize_root(FmmmDeviceStateView state) {
    summarize_root<<<1, 1>>>(state);
    check(cudaGetLastError(), "launch explicit Barnes-Hut root summary");
}

void fmmm_bh_build_tree(FmmmDeviceStateView state) {
    build_tree<<<1, 1>>>(state);
    check(cudaGetLastError(), "launch explicit Barnes-Hut tree build");
}

void fmmm_bh_compute_repulsion(FmmmDeviceStateView state, FmmmBarnesHutConfig config) {
    if (state.force_x == nullptr || state.force_y == nullptr) throw std::invalid_argument("Barnes-Hut force buffers are required");
    summarize_tree<<<1, 1>>>(state, config.summarize_iterations);
    check(cudaGetLastError(), "launch explicit Barnes-Hut tree summary");
    if (state.node_count > 0) compute_repulsion<<<(state.node_count + 127) / 128, 128>>>(state, config.theta);
    check(cudaGetLastError(), "launch explicit Barnes-Hut repulsion");
    if (state.node_count > 0 && state.edge_count > 0) {
        add_edge_attraction<<<(state.node_count + 127) / 128, 128>>>(state, config.attraction_weight);
        check(cudaGetLastError(), "launch explicit Barnes-Hut attraction");
    }
}

void fmmm_full_square_project(oracle::DeviceGraphView graph, bool layers_enabled,
                              std::int32_t rounds, std::uint64_t seed) {
    if (rounds <= 0 || graph.node_count == 0 || graph.edge_count > kProjectionMaximumEdges) return;
    repair_invalid_projection<<<1, 1>>>(graph, layers_enabled, seed);
    check(cudaGetLastError(), "launch FMMM invalid-quantization repair");
    if (graph.node_count > kProjectionMaximumNodes) return;
    full_square_project<<<1, 1>>>(graph, layers_enabled, rounds);
    check(cudaGetLastError(), "launch FMMM full-square projection");
}

} // namespace lcn::release
