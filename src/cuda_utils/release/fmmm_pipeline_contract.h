#pragma once

// Frozen Phase-1 CUDA FMMM pipeline ABI.  These are non-owning POD views:
// every pointer must refer to device memory owned by the caller/workspace.

#include <cstdint>

namespace lcn::release {

enum class FmmmPipelineStatus : int {
    ok = 0,
    invalid_input = 1,
    backend_error = 2,
};

enum class FmmmPipelineStage : int {
    none = 0,
    normal_force = 1,
    postprocess = 2,
    fine_tuning = 3,
    resize = 4,
    hierarchy = 5,
    uncoarsen = 6,
    packing = 7,
};

enum class FmmmMaxIterChange : int { constant = 0, linearly_decreasing = 1, rapidly_decreasing = 2 };
enum class FmmmRepulsiveMethod : int { exact = 0, grid_approximation = 1, nmm = 2 };
enum class FmmmInitialPlacement : int { uniform_grid = 0, random_time = 1, random_rand_iter = 2, keep_positions = 3 };
enum class FmmmStopCriterion : int { fixed_iterations = 0, threshold = 1, fixed_or_threshold = 2 };

struct FmmmGraphView {
    const int* original_node_id = nullptr;
    const int* edge_source = nullptr;
    const int* edge_target = nullptr;
    const double* edge_length = nullptr;
    int node_count = 0;
    int edge_count = 0;
};

struct FmmmProfile {
    int normal_iterations = 0;
    int postprocess_iterations = 10;
    int fine_tuning_iterations = 20;
    int act_level = 0;
    int max_level = 0;
    double average_ideal_length = 0.0;
    double initial_box_length = 0.0;
    double initial_down_left_x = 0.0;
    double initial_down_left_y = 0.0;
    double normal_spring_strength = 1.0;
    double normal_repulsion_strength = 1.0;
    double post_spring_strength = 1.0;
    double post_repulsion_strength = 1.0;
    double fine_spring_strength = 2.0;
    double fine_repulsion_strength = 0.2;
    double cool_value = 0.99;
    double fine_tune_scalar = 0.2;
    // ENGINEERING_V1: exact G2 branch matches frozen coolTemperature=false witnesses.
    // cool_temperature=true is metadata-only until a separate kernel AB pass; phase-0
    // exact kernel still uses cool_factor=1.0 regardless of this flag.
    bool cool_temperature = false;
    bool resize_drawing = true;
    // When resize_drawing is set, also run the final no-box-refresh adapt (OGDF default).
    bool apply_final_resize = true;
    // After the force schedule, apply FMMMLayout::call's single-component
    // pack_subGraph_drawings + Integer adjust (rotate/tip/translate/floor).
    // Default true so G2 route finals match full-call gold for G1. Force-stage
    // AB disables this to compare pre-pack oracle endpoints.
    bool apply_single_component_pack = true;
    std::uint64_t seed = 0;
    int placement_draws_consumed = 0;
    FmmmMaxIterChange max_iter_change = FmmmMaxIterChange::linearly_decreasing;
    FmmmRepulsiveMethod repulsive_method = FmmmRepulsiveMethod::exact;
    FmmmInitialPlacement initial_placement = FmmmInitialPlacement::random_rand_iter;
    FmmmStopCriterion stop_criterion = FmmmStopCriterion::fixed_iterations;
    int max_iter_factor = 10;
    double threshold = 0.01;
};

struct FmmmLevelState {
    double* position_x = nullptr;
    double* position_y = nullptr;
    double* force_x = nullptr;
    double* force_y = nullptr;
    double* attraction_x = nullptr;
    double* attraction_y = nullptr;
    double* repulsion_x = nullptr;
    double* repulsion_y = nullptr;
    double* last_movement_x = nullptr;
    double* last_movement_y = nullptr;
    void* nmm_workspace = nullptr;
    // Caller/FmmmLevelState-owned scalar staging (no hidden alloc in G2).
    // G3 workspace does not yet expose dedicated stage_* slots; coordinator must
    // supply these four device scalars until G3 accept wires them.
    int* stage_status = nullptr;
    double* stage_box = nullptr;
    double* stage_left_x = nullptr;
    double* stage_left_y = nullptr;
    double box_length = 0.0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
    double resize_scale = 1.0;
    double cooling = 1.0;
    int iteration = 0;
    int phase = 0;
    int node_count = 0;
};

struct FmmmStageResult {
    FmmmPipelineStatus status = FmmmPipelineStatus::invalid_input;
    FmmmPipelineStage completed_stage = FmmmPipelineStage::none;
    int first_bad_node = -1;
    int first_bad_level = -1;
    const char* reason = "none";
};

// OGDF single-level route: n <= minGraphSize (default 50). G2 owns this branch only.
inline constexpr int kFmmmDefaultMinGraphSize = 50;
inline bool fmmm_is_single_level_route(int node_count,
                                       int min_graph_size = kFmmmDefaultMinGraphSize) {
    return node_count >= 1 && node_count <= min_graph_size;
}

// Coordinator/hierarchy gate: refuse multilevel hierarchy construction on the
// G2-owned single-level OGDF route (n <= minGraphSize).
inline FmmmStageResult fmmm_hierarchy_reject_if_single_level(
    int node_count, int min_graph_size = kFmmmDefaultMinGraphSize) {
    FmmmStageResult result;
    if (fmmm_is_single_level_route(node_count, min_graph_size)) {
        result.status = FmmmPipelineStatus::invalid_input;
        result.completed_stage = FmmmPipelineStage::none;
        result.reason = "single-level OGDF route: hierarchy construction refused";
        return result;
    }
    result.status = FmmmPipelineStatus::ok;
    result.reason = "ok";
    return result;
}

} // namespace lcn::release
