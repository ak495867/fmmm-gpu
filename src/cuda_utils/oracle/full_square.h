#pragma once

#include "graph.h"

#include <cstdint>
#include <string>
#include <vector>

namespace lcn::oracle {

enum class ProposalStatus : std::int32_t {
    initial = 0, accepted, rejected_stale, rejected_invalid, no_op, rejected_overflow
};
enum class FallbackReason : std::int32_t { none = 0, overflow = 1 };

struct DeviceObjective {
    std::int64_t K;
    std::int64_t nK;
    std::int64_t Phi;
    std::int64_t C;
    std::int32_t valid;
    std::int32_t exact_valid;
    std::int32_t status;
    std::int32_t fallback_reason;
};

struct ScoreResult {
    std::string proposal_id;
    ProposalStatus status = ProposalStatus::initial;
    bool valid = false;
    bool exact_valid = false;
    std::vector<std::int64_t> edge_crossings;
    std::int64_t K = 0, nK = 0, Phi = 0, C = 0;
    bool fallback_required = false;
    std::string fallback_reason = "none";
};

struct Proposal {
    std::string id;
    std::int32_t node_id;
    std::int64_t from_x, from_y, to_x, to_y;
};

class FullSquareSession {
public:
    FullSquareSession(DeviceGraph& graph, bool layers_enabled);
    ScoreResult initial();
    ScoreResult apply(const Proposal& proposal);

private:
    ScoreResult download(ProposalStatus requested_status, const std::string& id, bool preserve_on_reject);
    DeviceGraph& graph_;
    bool layers_enabled_;
    DeviceBuffer<std::int64_t> current_counts_;
    DeviceBuffer<std::int64_t> scratch_counts_;
    DeviceBuffer<std::int64_t> returned_counts_;
    DeviceBuffer<DeviceObjective> current_objective_;
    DeviceBuffer<DeviceObjective> returned_objective_;
    bool initialized_ = false;
};

ScoreResult evaluate_full_square(DeviceGraph& graph, bool layers_enabled);
ScoreResult evaluate_full_square_candidate(DeviceGraphView candidate, bool layers_enabled);
const char* status_name(ProposalStatus status);

} // namespace lcn::oracle
