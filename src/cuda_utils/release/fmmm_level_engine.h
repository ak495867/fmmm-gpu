#pragma once
#include "fmmm_pipeline_contract.h"
#include "fmmm_workspace.h"
#include "../oracle/device_memory.h"
#include "../oracle/full_square.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <utility>
#include <vector>

namespace lcn::oracle {

enum class FmmmFinalStatus { ok, invalid_input, rejected_by_admission, backend_error };
struct FmmmFinalResult {
    FmmmFinalStatus status;
    // Continuous G1 (default): double coordinates in the affine-mapped box.
    // Integer diagnostic path also fills this with integral values as doubles.
    std::vector<std::pair<double, double>> coordinates; // only on ok
    ScoreResult score;                                  // only on ok
    int first_bad_node = -1;
    const char* reason = "none";
};

// Default G1: continuous affine (no integer round) + continuous crossings K.
FmmmFinalResult fmmm_finalize_ogdf_profile_raw(int nodes, int edges_count, const int* d_source, const int* d_target, const int* d_layer, const double* raw_x, const double* raw_y, std::int64_t width, std::int64_t height, bool layers_enabled);

FmmmFinalResult fmmm_finalize_ogdf_profile(
    DeviceGraph& graph,
    const DeviceBuffer<double>& raw_x,
    const DeviceBuffer<double>& raw_y,
    std::int64_t width,
    std::int64_t height,
    bool layers_enabled);

// Diagnostic: legacy integer quantize + FULL_SQUARE_GPU admission.
FmmmFinalResult fmmm_finalize_ogdf_profile_integer(
    DeviceGraph& graph,
    const DeviceBuffer<double>& raw_x,
    const DeviceBuffer<double>& raw_y,
    std::int64_t width,
    std::int64_t height,
    bool layers_enabled);

// A deliberately narrow, device-owned checkpoint for the Phase-1 full-path
// profile.  It covers FMMMLayout::create_initial_placement exactly: geometry
// is used to construct the first box, the seeded OGDF draw stream fills it,
// and the mutable dynamic box is then recalculated.  It does not alter the
// legacy fmmm_seed execution path.
struct FmmmFullPathPlacementDiagnostic {
    int status = 1;
    double initial_box_length = 0.0;
    double dynamic_box_length = 0.0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
    std::vector<std::pair<double, double>> positions;
};

FmmmFullPathPlacementDiagnostic fmmm_full_path_coarsest_placement(
    int node_count, double node_width, double node_height, std::uint64_t seed);

// Bounded single-step witness for the pinned New-force / exact-repulsion
// branch.  The caller supplies one frozen level state, so this is an AB
// primitive rather than a second layout implementation.
struct FmmmExactForceStepDiagnostic {
    int status = 1;
    std::vector<std::pair<double, double>> adjusted_positions;
    std::vector<std::pair<double, double>> attraction;
    std::vector<std::pair<double, double>> repulsion;
    std::vector<std::pair<double, double>> capped_force;
    std::vector<std::pair<double, double>> moved_positions;
    double box_length = 0.0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
};

FmmmExactForceStepDiagnostic fmmm_full_path_exact_force_step(
    const std::vector<std::pair<double, double>>& positions,
    const std::vector<int>& edge_source, const std::vector<int>& edge_target,
    const std::vector<double>& edge_length, double box_length, double down_left_x,
    double down_left_y, double average_ideal_length, int iteration);

struct FmmmExactForceLoopDiagnostic {
    int status = 1;
    std::vector<std::pair<double, double>> final_positions;
    double box_length = 0.0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
};

FmmmExactForceLoopDiagnostic fmmm_full_path_exact_force_loop(
    const std::vector<std::pair<double, double>>& positions,
    const std::vector<int>& edge_source, const std::vector<int>& edge_target,
    const std::vector<double>& edge_length, double box_length, double down_left_x,
    double down_left_y, double average_ideal_length, int iterations);

// Device-buffer counterpart used only by the full-path assembly seam. It
// performs no allocation, no coordinate download, and no synchronization;
// the caller can pass its output position planes directly to the next GPU
// stage (Advanced uncoarsening).
void fmmm_full_path_exact_force_loop_device(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, double* position_x, double* position_y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box_length, double* out_down_left_x, double* out_down_left_y,
    int force_phase = 0);

// Device-owned output of Multilevel::create_suns_and_planets() followed by
// create_multilevel_representations() for the frozen Phase-1 profile.  This
// deliberately has no dependency on DeviceGraph: its inputs are the already
// normalized, ordered FMMM level graph, including the double ideal lengths
// calculated by BoundingCircle.  Keeping the edge vectors explicit makes it
// impossible for an accidental host graph/container traversal to change the
// selection or first-parallel-edge order.
struct FmmmFullPathHierarchyEdge {
    int source = 0;
    int target = 0;
    double ideal_length = 0.0;
};

struct FmmmFullPathHierarchyDiagnostic {
    // 0 = OK, 1 = unsupported input/capacity, 2 = a moon has no valid
    // planet/PM neighbour.  The latter is an OGDF hierarchy precondition,
    // not a CPU fallback signal.
    int status = 1;
    int coarse_node_count = 0;
    std::vector<int> parent;
    std::vector<int> node_type;
    std::vector<int> dedicated_sun;
    std::vector<double> sun_distance;
    std::vector<FmmmFullPathHierarchyEdge> coarse_edges;
};

// The ordered, flattened attributes which OGDF's Multilevel construction
// hands to Advanced placement.  This is an AB surface first; its layout is
// intentionally identical to the device plan consumed by the next seam so a
// verified capture never requires a host-side reconstruction of hierarchy.
struct FmmmAdvancedPlanDiagnostic {
    int status = 1;
    int lower_node_count = 0;
    int lower_edge_ordinal_span = 0;
    int high_node_count = 0;
    int high_edge_ordinal_span = 0;
    std::vector<int> lower_type, lower_dedicated_sun, high_lower_node;
    std::vector<double> lower_dedicated_sun_distance;
    std::vector<int> lower_edge_source, lower_edge_target, lower_edge_is_moon;
    std::vector<double> lower_edge_length;
    std::vector<int> lower_adjacency_begin, lower_adjacency_edge;
    std::vector<int> lambda_begin, neighbour_sun_begin, neighbour_sun_nodes;
    std::vector<double> lambda_values;
    std::vector<int> moon_begin, moon_nodes;
    std::vector<int> high_edge_source, high_edge_target, high_edge_is_extra;
    std::vector<int> high_adjacency_begin, high_adjacency_edge;
};

// `edge_source`, `edge_target`, and `edge_length` must have equal length and
// be in the normalized OGDF Graph::edges() order. `edge_length` is normally
// the one frozen BoundingCircle fine length (48.284271247461902 for the
// 20x20 profile), but remains an input so later levels retain their actual
// independently derived lengths.  All hierarchy selection and coarse-edge
// construction occur in one ordered CUDA thread.
FmmmFullPathHierarchyDiagnostic fmmm_full_path_solar_hierarchy(
    int node_count, const std::vector<int>& edge_source,
    const std::vector<int>& edge_target, const std::vector<double>& edge_length,
    std::uint64_t seed);

FmmmAdvancedPlanDiagnostic fmmm_full_path_advanced_plan(
    int node_count, const std::vector<int>& edge_source,
    const std::vector<int>& edge_target, const std::vector<double>& edge_length,
    std::uint64_t seed);

// First native coordinator seam. The fine graph enters once, then hierarchy
// metadata, coarse force state, and Advanced placement remain device-resident
// until this diagnostic's final download. `coarse_initial_*` is temporary
// oracle input while coarsest placement is folded into the coordinator.
struct FmmmNativeUncoarsenDiagnostic {
    int status = 1;
    std::vector<std::pair<double, double>> positions;
    double box_length = 0.0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
};

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_hierarchy_force_uncoarsen(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, const std::vector<std::pair<double, double>>& coarse_initial,
    double coarse_box_length, double coarse_down_left_x, double coarse_down_left_y,
    double average_ideal_length, int force_iterations, std::uint64_t seed,
    int placement_draws_consumed, int fine_normal_iterations = 0,
    int postprocess_iterations = 0, int fine_tuning_iterations = 0, int pack_single_component = 0);

// The same bounded seam with coarsest placement generated on GPU from the
// frozen 20x20 profile.  It is intentionally diagnostic-only until repeated
// hierarchy levels, post-uncoarsen force, projection, and recount are owned.
FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed);

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_normal_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed);

FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_postprocessed_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed);
FmmmNativeUncoarsenDiagnostic fmmm_full_path_native_profile_packed_candidate(
    int node_count, const std::vector<int>& fine_source, const std::vector<int>& fine_target,
    const std::vector<double>& fine_length, double average_ideal_length,
    int force_iterations, std::uint64_t seed);

// Single-component FMMMLayout::call export (rotate/tip/translate/floor).
void fmmm_launch_single_component_pack_export(int node_count, double* x, double* y, int* status);

void launch_resize_to_ideal_length_kernel(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    int refresh_box, double* out_box, double* out_down_left_x, double* out_down_left_y,
    int* status, cudaStream_t stream = nullptr);

} // namespace lcn::oracle

namespace lcn::release {
struct DeviceFmmmSession;
FmmmStageResult fmmm_run_single_level(FmmmLevelState& state, const FmmmGraphView& view, const FmmmProfile& profile);

// G0 device-resident schedule coordinator. Upload, normalization, and the
// initial hierarchy phase are owned by the caller; this seam submits the
// remaining session-owned device phases in order and performs no host graph,
// component, iteration, resize, score, or fallback control.
//
// The historical name is retained as an ABI seam for resident_api.cu. Its
// implementation is the device schedule below, not the legacy host route.
struct FmmmCoordinatorResult {
    oracle::FmmmFinalResult finalization{};
    FmmmStageResult stage{};
    const char* route = "none";
};

FmmmCoordinatorResult fmmm_run_ogdf_coordinator(
    DeviceFmmmSession& session,
    std::uint64_t seed,
    int fixed_iterations,
    bool layers_enabled);
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
    cudaStream_t stream = nullptr);

void launch_full_path_coarsest_placement_kernel(
    int node_count, double node_width, double node_height,
    std::uint64_t seed, int* status, double* initial_box,
    double* dynamic_box, double* down_left_x, double* down_left_y,
    double* x, double* y, cudaStream_t stream = nullptr);

void launch_exact_force_loop_kernel(
    int node_count, int edge_count, const int* edge_source,
    const int* edge_target, const double* edge_length, double* x, double* y,
    double box_length, double down_left_x, double down_left_y,
    double average_ideal_length, int iterations, int* status,
    double* out_box, double* out_down_left_x, double* out_down_left_y,
    int force_phase, const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream = nullptr);

// Host-orchestrated OGDF calculate_forces schedule with G5 NMM repulsion.
// Used when profile.repulsive_method == nmm and node_count >= 175.
// Springs/cool/oscillation/move stay on device; each iteration calls fmmm_nmm_force.
FmmmStageResult fmmm_run_nmm_force_loop(
    int node_count, int edge_count, const int* edge_source, const int* edge_target,
    const double* edge_length, double* x, double* y, double box_length,
    double down_left_x, double down_left_y, double average_ideal_length, int iterations,
    int* status, double* out_box, double* out_down_left_x, double* out_down_left_y,
    int force_phase, double* last_movement_x, double* last_movement_y, double* attraction_x,
    double* attraction_y, double* repulsion_x, double* repulsion_y, std::uint64_t seed,
    void* persistent_workspace = nullptr);
} // namespace lcn::release

