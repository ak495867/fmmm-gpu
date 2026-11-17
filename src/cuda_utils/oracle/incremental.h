#pragma once

#include "full_square.h"

#include <string>
#include <vector>

namespace lcn::oracle {

// C1/C2/C3 are proposal evaluators, not optimisers.  FULL_SQUARE_GPU remains
// the admission authority; this record makes the incremental comparison
// observable to the differential tests and fixture reports.
struct IncrementalAudit {
    std::string lane;
    bool compared = false;
    bool counts_match = true;
    bool objective_match = true;
    bool state_preserved = true;
    bool fallback = false;
    std::string fallback_reason = "none";
    std::int64_t candidate_count = 0;
};

struct IncrementalReplay {
    ScoreResult initial;
    std::vector<ScoreResult> results;
    std::vector<IncrementalAudit> audit;
    std::vector<std::int32_t> csr_offsets;
    std::vector<std::int32_t> csr_edges;
};

IncrementalReplay replay_incremental_exact(const GraphSpec& spec,
                                            const std::vector<Proposal>& proposals,
                                            bool layers_enabled,
                                            const std::string& lane);

} // namespace lcn::oracle
