#include "incremental.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace lcn::oracle {
namespace {

__device__ inline int sgn(std::int64_t v) { return (v > 0) - (v < 0); }
__device__ inline std::int64_t orient(std::int64_t ax, std::int64_t ay,
                                      std::int64_t bx, std::int64_t by,
                                      std::int64_t cx, std::int64_t cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}
__device__ inline bool proper_cross(const DeviceGraphView& graph, int e, int f) {
    const int a = graph.source[e], b = graph.target[e], c = graph.source[f], d = graph.target[f];
    if (a == c || a == d || b == c || b == d) return false;
    const auto o1 = orient(graph.x[a], graph.y[a], graph.x[b], graph.y[b], graph.x[c], graph.y[c]);
    const auto o2 = orient(graph.x[a], graph.y[a], graph.x[b], graph.y[b], graph.x[d], graph.y[d]);
    const auto o3 = orient(graph.x[c], graph.y[c], graph.x[d], graph.y[d], graph.x[a], graph.y[a]);
    const auto o4 = orient(graph.x[c], graph.y[c], graph.x[d], graph.y[d], graph.x[b], graph.y[b]);
    return sgn(o1) * sgn(o2) < 0 && sgn(o3) * sgn(o4) < 0;
}
__device__ inline bool eligible(const DeviceGraphView& graph, int e, int f, bool layers_enabled) {
    if (e == f) return false;
    const int a = graph.source[e], b = graph.target[e], c = graph.source[f], d = graph.target[f];
    return a != c && a != d && b != c && b != d && (!layers_enabled || graph.layer[e] == graph.layer[f]);
}

// One thread per edge. Non-incident edges apply deltas against every edge in
// the moved node's device CSR range; incident edges are fully recounted. This
// updates both sides of every affected pair and leaves all others untouched.
__global__ void c1_counts_kernel(DeviceGraphView old_graph, DeviceGraphView new_graph,
                                 const std::int64_t* old_counts,
                                 const std::int32_t* incident, int incident_count,
                                 int moved, bool layers_enabled,
                                 std::int64_t* next_counts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= old_graph.edge_count) return;
    const bool is_incident = old_graph.source[e] == moved || old_graph.target[e] == moved;
    std::int64_t value = is_incident ? 0 : old_counts[e];
    if (is_incident) {
        for (int f = 0; f < old_graph.edge_count; ++f)
            if (eligible(new_graph, e, f, layers_enabled) && proper_cross(new_graph, e, f)) ++value;
    } else {
        for (int k = 0; k < incident_count; ++k) {
            const int f = incident[k];
            if (!eligible(old_graph, e, f, layers_enabled)) continue;
            const bool was = proper_cross(old_graph, e, f);
            const bool now = proper_cross(new_graph, e, f);
            value += static_cast<int>(now) - static_cast<int>(was);
        }
    }
    next_counts[e] = value;
}

struct C2State {
    std::vector<std::int64_t> counts;
    std::vector<std::int64_t> buckets;
    std::int64_t k = 0, nk = 0, phi = 0, c = 0;

    void reset(const ScoreResult& score) {
        counts = score.edge_crossings;
        buckets.assign(counts.size() + 1, 0);
        k = phi = c = 0;
        for (const auto n : counts) {
            if (n < 0 || n >= static_cast<std::int64_t>(buckets.size())) throw std::logic_error("invalid full-square count");
            ++buckets[n]; k = std::max(k, n); phi += n * n; c += n;
        }
        c /= 2; nk = counts.empty() ? 0 : buckets[k];
    }

    ScoreResult candidate(const std::vector<std::int64_t>& next, const ScoreResult& base) const {
        ScoreResult out = base;
        std::vector<std::int64_t> delta(buckets.size(), 0);
        std::int64_t dphi = 0, dsum = 0, max_new = 0;
        for (std::size_t e = 0; e < next.size(); ++e) if (next[e] != counts[e]) {
            if (next[e] < 0 || next[e] >= static_cast<std::int64_t>(buckets.size())) throw std::logic_error("incremental count out of range");
            dphi += next[e] * next[e] - counts[e] * counts[e];
            dsum += next[e] - counts[e];
            --delta[counts[e]]; ++delta[next[e]]; max_new = std::max(max_new, next[e]);
        }
        if ((dsum & 1) != 0) throw std::logic_error("incremental degree delta is odd");
        std::int64_t next_k = std::max(k, max_new);
        while (next_k > 0 && buckets[next_k] + delta[next_k] == 0) --next_k;
        out.edge_crossings = next; out.K = next_k;
        out.nK = next.empty() ? 0 : buckets[next_k] + delta[next_k];
        out.Phi = phi + dphi; out.C = c + dsum / 2;
        return out;
    }
};

bool same_score(const ScoreResult& a, const ScoreResult& b) {
    return a.edge_crossings == b.edge_crossings && a.K == b.K && a.nK == b.nK && a.Phi == b.Phi && a.C == b.C;
}

std::vector<std::int32_t> build_offsets(const GraphSpec& spec, std::vector<std::int32_t>& entries) {
    std::vector<std::int32_t> offsets(spec.nodes.size() + 1, 0);
    for (const auto& edge : spec.edges) { ++offsets[edge.source + 1]; ++offsets[edge.target + 1]; }
    for (std::size_t n = 1; n < offsets.size(); ++n) offsets[n] += offsets[n - 1];
    entries.assign(offsets.back(), 0); auto at = offsets;
    for (std::size_t e = 0; e < spec.edges.size(); ++e) {
        entries[at[spec.edges[e].source]++] = static_cast<std::int32_t>(e);
        entries[at[spec.edges[e].target]++] = static_cast<std::int32_t>(e);
    }
    return offsets;
}

std::vector<std::int64_t> calculate_c1(const GraphSpec& old_spec, const GraphSpec& next_spec,
                                        const C2State& state, const std::vector<std::int32_t>& offsets,
                                        const std::vector<std::int32_t>& csr, int moved, bool layers_enabled) {
    DeviceGraph old_graph(old_spec), next_graph(next_spec);
    old_graph.upload(false); next_graph.upload(false);
    const int edge_count = old_graph.edge_count();
    DeviceBuffer<std::int64_t> d_old(state.counts.size()), d_next(state.counts.size());
    DeviceBuffer<std::int32_t> d_incident(moved >= 0 && moved + 1 < static_cast<int>(offsets.size()) ? offsets[moved + 1] - offsets[moved] : 0);
    d_old.upload(state.counts.data(), state.counts.size());
    const int first = offsets[moved], count = offsets[moved + 1] - first;
    if (count) d_incident.upload(csr.data() + first, count);
    if (edge_count) c1_counts_kernel<<<(edge_count + 127) / 128, 128>>>(old_graph.view(), next_graph.view(), d_old.get(), d_incident.get(), count, moved, layers_enabled, d_next.get());
    cuda_check(cudaGetLastError(), "launch C1 affected-pair evaluator");
    cuda_check(cudaDeviceSynchronize(), "synchronize C1 affected-pair evaluator");
    std::vector<std::int64_t> out(state.counts.size()); d_next.download(out.data(), out.size());
    return out;
}

} // namespace

IncrementalReplay replay_incremental_exact(const GraphSpec& original, const std::vector<Proposal>& proposals,
                                            bool layers_enabled, const std::string& lane) {
    if (lane != "C1" && lane != "C2" && lane != "C3") throw std::invalid_argument("lane must be C1, C2, or C3");
    DeviceGraph authority_graph(original); authority_graph.upload(false);
    FullSquareSession authority(authority_graph, layers_enabled);
    IncrementalReplay replay; replay.initial = authority.initial();
    replay.csr_offsets = build_offsets(original, replay.csr_edges);
    C2State state;
    // An initial invalid layout has no count vector to increment from.  Keep
    // the session alive under the full-square authority until a proposal
    // establishes a valid state, then bootstrap C1/C2/C3 from that exact
    // returned vector.  In particular, never hand an empty buffer to C1.
    bool incremental_ready = replay.initial.valid;
    if (incremental_ready) state.reset(replay.initial);
    GraphSpec current = original;

    for (const Proposal& proposal : proposals) {
        const auto before = state.counts;
        ScoreResult exact = authority.apply(proposal); // admission and final objective authority
        IncrementalAudit audit; audit.lane = lane;
        audit.candidate_count = lane == "C3" ? static_cast<std::int64_t>(original.edges.size()) : 0;
        if (exact.valid && !incremental_ready) {
            // Structured, exact recovery fallback. FULL_SQUARE_GPU has just
            // admitted this candidate and returned every count/objective
            // field. It is the sole publisher for this transition.
            audit.fallback = true;
            audit.fallback_reason = "initial_invalid_full_square";
            state.reset(exact);
            incremental_ready = true;
            if (exact.status == ProposalStatus::accepted) {
                GraphSpec next = current;
                next.nodes[proposal.node_id] = {proposal.to_x, proposal.to_y};
                current = std::move(next);
            }
        } else if (exact.valid) {
            GraphSpec next = current;
            if (proposal.node_id >= 0 && proposal.node_id < static_cast<std::int32_t>(next.nodes.size()))
                next.nodes[proposal.node_id] = {proposal.to_x, proposal.to_y};
            const auto c1 = calculate_c1(current, next, state, replay.csr_offsets, replay.csr_edges,
                                         proposal.node_id, layers_enabled);
            const ScoreResult c2 = state.candidate(c1, exact);
            audit.compared = true;
            audit.counts_match = c1 == exact.edge_crossings;
            audit.objective_match = same_score(c2, exact);
            if (!audit.counts_match || !audit.objective_match)
                throw std::logic_error("incremental evaluator drifted from FULL_SQUARE_GPU");
            // C2 persistent state publishes only an accepted/no-op exact result.
            state.reset(exact);
            if (exact.status == ProposalStatus::accepted) current = std::move(next);
        } else {
            audit.state_preserved = state.counts == before;
            audit.fallback = exact.fallback_required || !incremental_ready;
            audit.fallback_reason = exact.fallback_required ? exact.fallback_reason :
                                    (!incremental_ready ? "initial_invalid_full_square" : "none");
        }
        replay.results.push_back(std::move(exact)); replay.audit.push_back(std::move(audit));
    }
    return replay;
}

} // namespace lcn::oracle
