#pragma once

#include "../oracle/graph.h"

#include <cstdint>

namespace lcn::release {

// Explicit ownership/view boundary for the FMMM adapter.  In particular, no
// CUDA kernel may reach a host global device pointer: every tree and layout
// buffer needed by a launch appears in this POD view.
struct FmmmDeviceStateView {
    float* position_x = nullptr;
    float* position_y = nullptr;
    float* force_x = nullptr;
    float* force_y = nullptr;
    // Kept separate from repulsion so the OGDF stage-specific spring and
    // repulsion strengths can be combined on device without host arithmetic.
    float* attraction_x = nullptr;
    float* attraction_y = nullptr;
    const std::int32_t* edge_source = nullptr;
    const std::int32_t* edge_target = nullptr;
    std::int32_t* child = nullptr;       // quadtree node × 4
    float* mass = nullptr;
    float* center_x = nullptr;
    float* center_y = nullptr;
    float* width = nullptr;
    std::int32_t* ready = nullptr;
    std::int32_t* next_tree_node = nullptr;
    std::int32_t node_count = 0;
    std::int32_t edge_count = 0;
    std::int32_t tree_capacity = 0;
};

struct FmmmBarnesHutConfig {
    float theta = 1.0F;
    // Applied only when an explicit edge list is present.  This is a seed
    // force, not an authority for accepting coordinates.
    float attraction_weight = 0.125F;
    std::int32_t summarize_iterations = 64;
    std::int32_t relaxation_iterations = 1;
};

// Narrow adapter entry points. Ownership stays with the release workspace;
// these functions never allocate hidden global CUDA state.
void fmmm_bh_initialize(FmmmDeviceStateView state);
void fmmm_bh_build_tree(FmmmDeviceStateView state);
void fmmm_bh_summarize_root(FmmmDeviceStateView state);
void fmmm_bh_compute_repulsion(FmmmDeviceStateView state, FmmmBarnesHutConfig config);

// A deliberately serial, device-only projection pass. It uses the same exact
// geometric predicates and lexicographic objective as FULL_SQUARE_GPU, but
// owns neither a host score nor a host acceptance loop. Phase 1 favours this
// audit-friendly order over parallel local-search throughput.
void fmmm_full_square_project(oracle::DeviceGraphView graph, bool layers_enabled,
                              std::int32_t rounds, std::uint64_t seed);

} // namespace lcn::release
