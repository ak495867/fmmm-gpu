#include "full_square.h"

#include <cuda_runtime.h>

#include <climits>
#include <limits>
#include <stdexcept>

namespace lcn::oracle {
namespace {

constexpr std::int64_t kExactCoordinateLimit = 1000000000LL;

enum DeviceValidity : int { kValid = 0, kInvalid = 1, kOverflow = 2 };

__device__ inline std::int64_t candidate_x(const DeviceGraphView& graph, int node,
                                           int moved, std::int64_t x) {
    return node == moved ? x : graph.x[node];
}
__device__ inline std::int64_t candidate_y(const DeviceGraphView& graph, int node,
                                           int moved, std::int64_t y) {
    return node == moved ? y : graph.y[node];
}

__device__ inline int sign64(std::int64_t value) { return (value > 0) - (value < 0); }

// Coordinates are bounded before this is called: every delta is at most 1e9,
// so each product is <=1e18 and the signed difference is <=2e18 (< INT64_MAX).
__device__ inline std::int64_t orientation(std::int64_t ax, std::int64_t ay,
                                           std::int64_t bx, std::int64_t by,
                                           std::int64_t cx, std::int64_t cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline bool between_closed(std::int64_t value, std::int64_t a, std::int64_t b) {
    return value >= (a < b ? a : b) && value <= (a < b ? b : a);
}

__device__ inline bool point_on_segment(std::int64_t px, std::int64_t py,
                                        std::int64_t ax, std::int64_t ay,
                                        std::int64_t bx, std::int64_t by) {
    return orientation(ax, ay, bx, by, px, py) == 0 &&
           between_closed(px, ax, bx) && between_closed(py, ay, by);
}

__device__ int validate_layout(const DeviceGraphView& graph, int moved,
                               std::int64_t move_x, std::int64_t move_y) {
    if (graph.max_width > kExactCoordinateLimit || graph.max_height > kExactCoordinateLimit ||
        graph.max_width < 2 || graph.max_height < 2) return kOverflow;
    for (int i = 0; i < graph.node_count; ++i) {
        const auto x = candidate_x(graph, i, moved, move_x);
        const auto y = candidate_y(graph, i, moved, move_y);
        if (x > kExactCoordinateLimit || y > kExactCoordinateLimit || x < -kExactCoordinateLimit || y < -kExactCoordinateLimit)
            return kOverflow;
        if (x <= 0 || x >= graph.max_width || y <= 0 || y >= graph.max_height) return kInvalid;
        for (int j = 0; j < i; ++j) {
            if (x == candidate_x(graph, j, moved, move_x) && y == candidate_y(graph, j, moved, move_y)) return kInvalid;
        }
    }
    // A vertex on a nonincident edge is a forbidden contact even if it is an
    // isolated vertex and therefore would not appear in an edge-pair loop.
    for (int n = 0; n < graph.node_count; ++n) {
        for (int e = 0; e < graph.edge_count; ++e) {
            const int a = graph.source[e], b = graph.target[e];
            if (n == a || n == b) continue;
            if (point_on_segment(candidate_x(graph, n, moved, move_x), candidate_y(graph, n, moved, move_y),
                                 candidate_x(graph, a, moved, move_x), candidate_y(graph, a, moved, move_y),
                                 candidate_x(graph, b, moved, move_x), candidate_y(graph, b, moved, move_y))) return kInvalid;
        }
    }
    return kValid;
}

__device__ void set_invalid(DeviceObjective* result, ProposalStatus status, FallbackReason fallback) {
    result->K = result->nK = result->Phi = result->C = 0;
    result->valid = 0;
    result->exact_valid = 0;
    result->status = static_cast<int>(status);
    result->fallback_reason = static_cast<int>(fallback);
}

__device__ void score_candidate(const DeviceGraphView& graph, bool layers_enabled,
                                int moved, std::int64_t move_x, std::int64_t move_y,
                                std::int64_t* counts, DeviceObjective* result) {
    const int validity = validate_layout(graph, moved, move_x, move_y);
    if (validity != kValid) {
        set_invalid(result, validity == kOverflow ? ProposalStatus::rejected_overflow : ProposalStatus::rejected_invalid,
                    validity == kOverflow ? FallbackReason::overflow : FallbackReason::none);
        return;
    }
    for (int e = 0; e < graph.edge_count; ++e) counts[e] = 0;
    for (int i = 0; i < graph.edge_count; ++i) {
        const int ia = graph.source[i], ib = graph.target[i];
        const auto iax = candidate_x(graph, ia, moved, move_x), iay = candidate_y(graph, ia, moved, move_y);
        const auto ibx = candidate_x(graph, ib, moved, move_x), iby = candidate_y(graph, ib, moved, move_y);
        for (int j = i + 1; j < graph.edge_count; ++j) {
            const int ja = graph.source[j], jb = graph.target[j];
            if (ia == ja || ia == jb || ib == ja || ib == jb) continue;
            if (layers_enabled && graph.layer[i] != graph.layer[j]) continue;
            const auto jax = candidate_x(graph, ja, moved, move_x), jay = candidate_y(graph, ja, moved, move_y);
            const auto jbx = candidate_x(graph, jb, moved, move_x), jby = candidate_y(graph, jb, moved, move_y);
            const auto o1 = orientation(iax, iay, ibx, iby, jax, jay);
            const auto o2 = orientation(iax, iay, ibx, iby, jbx, jby);
            const auto o3 = orientation(jax, jay, jbx, jby, iax, iay);
            const auto o4 = orientation(jax, jay, jbx, jby, ibx, iby);
            if (sign64(o1) * sign64(o2) < 0 && sign64(o3) * sign64(o4) < 0) {
                ++counts[i]; ++counts[j];
            } else if ((o1 == 0 && point_on_segment(jax, jay, iax, iay, ibx, iby)) ||
                       (o2 == 0 && point_on_segment(jbx, jby, iax, iay, ibx, iby)) ||
                       (o3 == 0 && point_on_segment(iax, iay, jax, jay, jbx, jby)) ||
                       (o4 == 0 && point_on_segment(ibx, iby, jax, jay, jbx, jby))) {
                set_invalid(result, ProposalStatus::rejected_invalid, FallbackReason::none);
                return;
            }
        }
    }
    std::int64_t k = 0, nk = 0, phi = 0, degree_sum = 0;
    for (int e = 0; e < graph.edge_count; ++e) if (counts[e] > k) k = counts[e];
    for (int e = 0; e < graph.edge_count; ++e) {
        if (counts[e] == k) ++nk;
        // E is bounded by allocatable CUDA memory; this checked condition
        // still makes sum-of-squares overflow an explicit device rejection.
        if (counts[e] > 3037000499LL ||
            (counts[e] != 0 && phi > (LLONG_MAX - counts[e] * counts[e]))) {
            set_invalid(result, ProposalStatus::rejected_overflow, FallbackReason::overflow); return;
        }
        phi += counts[e] * counts[e];
        if (degree_sum > LLONG_MAX - counts[e]) { set_invalid(result, ProposalStatus::rejected_overflow, FallbackReason::overflow); return; }
        degree_sum += counts[e];
    }
    result->K = k; result->nK = nk; result->Phi = phi; result->C = degree_sum / 2;
    result->valid = 1; result->exact_valid = 1;
    result->status = static_cast<int>(ProposalStatus::initial);
    result->fallback_reason = static_cast<int>(FallbackReason::none);
}

__device__ void copy_counts(std::int64_t* to, const std::int64_t* from, int count) {
    for (int i = 0; i < count; ++i) to[i] = from[i];
}

__global__ void initial_score_kernel(DeviceGraphView graph, bool layers_enabled,
                                     std::int64_t* current_counts, std::int64_t* scratch_counts,
                                     std::int64_t* returned_counts, DeviceObjective* current,
                                     DeviceObjective* returned) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    score_candidate(graph, layers_enabled, -1, 0, 0, scratch_counts, returned);
    if (returned->valid) {
        copy_counts(current_counts, scratch_counts, graph.edge_count);
        *current = *returned;
        copy_counts(returned_counts, scratch_counts, graph.edge_count);
    } else {
        // Establish a defined, empty device state even for an initially
        // rejected replay. Later proposal records can therefore never read
        // uninitialised counts/objective fields.
        for (int e = 0; e < graph.edge_count; ++e) {
            current_counts[e] = 0;
            returned_counts[e] = 0;
        }
        *current = *returned;
    }
}

__global__ void proposal_score_kernel(DeviceGraphView graph, bool layers_enabled, int node_id,
                                      std::int64_t from_x, std::int64_t from_y,
                                      std::int64_t to_x, std::int64_t to_y,
                                      std::int64_t* current_counts, std::int64_t* scratch_counts,
                                      std::int64_t* returned_counts, DeviceObjective* current,
                                      DeviceObjective* returned) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (node_id < 0 || node_id >= graph.node_count || graph.x[node_id] != from_x || graph.y[node_id] != from_y) {
        *returned = *current;
        returned->valid = 0; returned->exact_valid = 0;
        returned->status = static_cast<int>(ProposalStatus::rejected_stale);
        copy_counts(returned_counts, current_counts, graph.edge_count);
        return;
    }
    score_candidate(graph, layers_enabled, node_id, to_x, to_y, scratch_counts, returned);
    if (!returned->valid) {
        const int status = returned->status, fallback = returned->fallback_reason;
        *returned = *current;
        returned->valid = 0; returned->exact_valid = 0; returned->status = status; returned->fallback_reason = fallback;
        copy_counts(returned_counts, current_counts, graph.edge_count);
        return;
    }
    const bool no_op = graph.x[node_id] == to_x && graph.y[node_id] == to_y;
    if (!no_op) { graph.x[node_id] = to_x; graph.y[node_id] = to_y; }
    copy_counts(current_counts, scratch_counts, graph.edge_count);
    *current = *returned;
    returned->status = static_cast<int>(no_op ? ProposalStatus::no_op : ProposalStatus::accepted);
    copy_counts(returned_counts, scratch_counts, graph.edge_count);
}

} // namespace

FullSquareSession::FullSquareSession(DeviceGraph& graph, bool layers_enabled)
    : graph_(graph), layers_enabled_(layers_enabled), current_counts_(graph.edge_count()),
      scratch_counts_(graph.edge_count()), returned_counts_(graph.edge_count()),
      current_objective_(1), returned_objective_(1) {}

ScoreResult FullSquareSession::initial() {
    initial_score_kernel<<<1, 1>>>(graph_.view(), layers_enabled_, current_counts_.get(), scratch_counts_.get(),
                                   returned_counts_.get(), current_objective_.get(), returned_objective_.get());
    cuda_check(cudaGetLastError(), "launch full-square initial kernel");
    cuda_check(cudaDeviceSynchronize(), "synchronize full-square initial kernel");
    initialized_ = true;
    return download(ProposalStatus::initial, "", false);
}

ScoreResult FullSquareSession::apply(const Proposal& proposal) {
    if (!initialized_) throw std::logic_error("full-square replay has no initial state");
    proposal_score_kernel<<<1, 1>>>(graph_.view(), layers_enabled_, proposal.node_id, proposal.from_x, proposal.from_y,
                                    proposal.to_x, proposal.to_y, current_counts_.get(), scratch_counts_.get(),
                                    returned_counts_.get(), current_objective_.get(), returned_objective_.get());
    cuda_check(cudaGetLastError(), "launch full-square proposal kernel");
    cuda_check(cudaDeviceSynchronize(), "synchronize full-square proposal kernel");
    return download(ProposalStatus::initial, proposal.id, true);
}

ScoreResult FullSquareSession::download(ProposalStatus, const std::string& id, bool preserve_on_reject) {
    DeviceObjective raw{};
    returned_objective_.download(&raw, 1);
    ScoreResult result;
    result.proposal_id = id;
    result.status = static_cast<ProposalStatus>(raw.status);
    result.valid = raw.valid != 0; result.exact_valid = raw.exact_valid != 0;
    result.K = raw.K; result.nK = raw.nK; result.Phi = raw.Phi; result.C = raw.C;
    result.fallback_required = raw.fallback_reason == static_cast<int>(FallbackReason::overflow);
    result.fallback_reason = result.fallback_required ? "overflow" : "none";
    if (result.valid || preserve_on_reject) {
        result.edge_crossings.resize(graph_.edge_count());
        returned_counts_.download(result.edge_crossings.data(), result.edge_crossings.size());
    } else {
        result.K = result.nK = result.Phi = result.C = 0;
    }
    return result;
}

ScoreResult evaluate_full_square(DeviceGraph& graph, bool layers_enabled) {
    FullSquareSession session(graph, layers_enabled);
    return session.initial();
}

ScoreResult evaluate_full_square_candidate(DeviceGraphView candidate, bool layers_enabled) {
    DeviceBuffer<std::int64_t> current_counts(candidate.edge_count);
    DeviceBuffer<std::int64_t> scratch_counts(candidate.edge_count);
    DeviceBuffer<std::int64_t> returned_counts(candidate.edge_count);
    DeviceBuffer<DeviceObjective> current_objective(1);
    DeviceBuffer<DeviceObjective> returned_objective(1);

    initial_score_kernel<<<1, 1>>>(candidate, layers_enabled, current_counts.get(), scratch_counts.get(),
                                   returned_counts.get(), current_objective.get(), returned_objective.get());
    cuda_check(cudaGetLastError(), "launch full-square initial kernel");
    cuda_check(cudaDeviceSynchronize(), "synchronize full-square initial kernel");

    DeviceObjective raw{};
    returned_objective.download(&raw, 1);
    ScoreResult result;
    result.proposal_id = "";
    result.status = static_cast<ProposalStatus>(raw.status);
    result.valid = raw.valid != 0; result.exact_valid = raw.exact_valid != 0;
    result.K = raw.K; result.nK = raw.nK; result.Phi = raw.Phi; result.C = raw.C;
    result.fallback_required = raw.fallback_reason == static_cast<int>(FallbackReason::overflow);
    result.fallback_reason = result.fallback_required ? "overflow" : "none";
    if (result.valid) {
        result.edge_crossings.resize(candidate.edge_count);
        returned_counts.download(result.edge_crossings.data(), result.edge_crossings.size());
    } else {
        result.K = result.nK = result.Phi = result.C = 0;
    }
    return result;
}

const char* status_name(ProposalStatus status) {
    switch (status) {
        case ProposalStatus::initial: return "initial";
        case ProposalStatus::accepted: return "accepted";
        case ProposalStatus::rejected_stale: return "rejected_stale";
        case ProposalStatus::rejected_invalid: return "rejected_invalid";
        case ProposalStatus::no_op: return "no_op";
        case ProposalStatus::rejected_overflow: return "rejected_overflow";
    }
    return "unsupported";
}

} // namespace lcn::oracle
