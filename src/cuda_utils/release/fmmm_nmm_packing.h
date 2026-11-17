#pragma once

#include "fmmm_pipeline_contract.h"
#include <cuda_runtime_api.h>

#include <cstdint>

namespace lcn::release {

// Device tree build diagnostics.  The first six counters are the published
// relation/tree counts; the overflow counters are kept in a separate tail of
// the build-counter allocation.
constexpr int kNmmBuildCounterCount = 8;
constexpr int kNmmBuildOverflowCount = 8;
constexpr int kNmmBuildStorageCount =
    kNmmBuildCounterCount + kNmmBuildOverflowCount;
constexpr int kNmmOverflowFrontierQueue = 0;
constexpr int kNmmOverflowTraversalQueue = 1;
constexpr int kNmmOverflowIList = 2;
constexpr int kNmmOverflowD1List = 3;
constexpr int kNmmOverflowD2List = 4;
constexpr int kNmmOverflowMList = 5;
constexpr int kNmmOverflowContained = 6;
constexpr int kNmmOverflowCell = 7;
constexpr int kNmmBuildOverflowBase = kNmmBuildCounterCount;

enum class FmmmNmmTreeConstruction : int { path_by_path = 0, subtree_by_subtree = 1 };
enum class FmmmNmmSmallCell : int { iteratively = 0, aluru = 1 };
enum class FmmmPackPreSort : int { none = 0, decreasing_height = 1, decreasing_width = 2 };
enum class FmmmPackTipOver : int { none = 0, no_growing_row = 1, always = 2 };

struct FmmmNmmState {
    double* position_x = nullptr; double* position_y = nullptr;
    double* force_x = nullptr; double* force_y = nullptr;
    const int* edge_source = nullptr; const int* edge_target = nullptr;
    const double* edge_length = nullptr;
    double* cell_downleft_x = nullptr; double* cell_downleft_y = nullptr;
    double* cell_box_length = nullptr; int* cell_level = nullptr;
    int* cell_parent = nullptr; int* cell_child4 = nullptr; int* cell_particle_count = nullptr;
    double* cell_center_x = nullptr; double* cell_center_y = nullptr;
    int* contained_begin = nullptr; int* contained_nodes = nullptr;
    int* temp_contained_nodes = nullptr;
    int* temp_contained_scratch = nullptr;
    int* temp_contained_begin = nullptr;
    int* I_begin = nullptr; int* I_nodes = nullptr;
    int* D1_begin = nullptr; int* D1_nodes = nullptr;
    int* D2_begin = nullptr; int* D2_nodes = nullptr;
    int* M_begin = nullptr; int* M_nodes = nullptr;
    double* multipole_re = nullptr; double* multipole_im = nullptr;
    double* local_re = nullptr; double* local_im = nullptr;
    int* particle_to_leaf = nullptr;
    int* status = nullptr;
    int node_count = 0; int edge_count = 0; int cell_count = 0;
    int contained_node_count = 0; int I_count = 0; int D1_count = 0;
    int D2_count = 0; int M_count = 0; int precision = 0;
    int expansion_stride = 0;
    // Optional caller-owned persistent allocation cache.  When non-null,
    // fmmm_nmm_force reuses capacity across iterations and never frees these
    // arrays.  The cache is invalidated/rebuilt by the owner when the graph or
    // level changes.
    void* persistent_workspace = nullptr;
    // Explicit ownership/invalidation epochs.  The owner increments the
    // relevant epoch when graph topology, level numbering, particle order,
    // tree relation, or particle-to-leaf map identity changes.
    std::uint64_t graph_epoch = 0;
    std::uint64_t level_epoch = 0;
    std::uint64_t particle_order_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t map_epoch = 0;
    // Per-call diagnostics.  The owner resets these before invoking the
    // force stage; they are never used to steer numerical results.
    int tree_rebuilt = 0;
    int particle_to_leaf_rebuilt = 0;
    int particle_to_leaf_launches = 0;

    // Session-owned device scratch for the device tree builder.  The first
    // six entries are the tree/list counters and entry six is the builder
    // error flag.  The queue is caller-owned and may alias M_nodes because
    // the builder consumes it before it emits the M list.
    int* device_build_counters = nullptr;
    int device_build_counter_capacity = 0;
    int* device_build_queue = nullptr;
    int* device_wsprl_queue = nullptr;
    int max_cells = 0;
    int max_contained = 0;
    int max_list = 0;
    int queue_capacity = 0;
    int frontier_queue_capacity = 0;
    int traversal_queue_capacity = 0;
    int max_I = 0;
    int max_D1 = 0;
    int max_D2 = 0;
    int max_M = 0;
};

struct FmmmNmmOptions {
    double box_length = 0.0, down_left_x = 0.0, down_left_y = 0.0;
    int precision = 0; int particles_in_leaf = 0;
    FmmmNmmTreeConstruction tree_construction = FmmmNmmTreeConstruction::subtree_by_subtree;
    FmmmNmmSmallCell small_cell = FmmmNmmSmallCell::iteratively;
    bool using_nmm = false; std::uint64_t seed = 0;
};

struct FmmmPackingState {
    double* position_x = nullptr; double* position_y = nullptr;
    int* component_id = nullptr; const int* component_begin = nullptr;
    const int* component_nodes = nullptr;
    const double* node_width = nullptr; const double* node_height = nullptr;
    double* rect_left = nullptr; double* rect_bottom = nullptr;
    double* rect_right = nullptr; double* rect_top = nullptr;
    double* old_dlc_x = nullptr; double* old_dlc_y = nullptr;
    double* new_dlc_x = nullptr; double* new_dlc_y = nullptr;
    int* tipped = nullptr; int* chosen_candidate = nullptr;
    int* row_index = nullptr;
    double* chosen_angle = nullptr; double* translate_x = nullptr; double* translate_y = nullptr;
    int node_count = 0; int component_count = 0; int component_node_count = 0;
};

struct FmmmPackingOptions {
    double page_ratio = 1.0; FmmmPackPreSort presort_components = FmmmPackPreSort::decreasing_height;
    FmmmPackTipOver tip_over_components = FmmmPackTipOver::no_growing_row;
    double min_dist_cc = 30.0; int rotation_steps = 0;
};

FmmmStageResult fmmm_nmm_force(FmmmNmmState&, const FmmmNmmOptions&);
// Bind a caller-owned persistent workspace to a complete NMM state view.
// This is used by the legacy force-loop coordinator; it does not build a
// tree or change any numerical algorithm.
FmmmStageResult fmmm_nmm_bind_workspace(FmmmNmmState&, void* persistent_workspace);
// Validate the session-owned capacity contract without allocating or copying.
// The resident route must satisfy this contract before any NMM kernel is
// enqueued; a false result is a bounded backend error, never a request to
// grow workspace from inside a phase.
FmmmStageResult fmmm_nmm_validate_preallocated_workspace(const FmmmNmmState&);
FmmmStageResult fmmm_nmm_build_async(FmmmNmmState&, const FmmmNmmOptions&, cudaStream_t);
FmmmStageResult fmmm_nmm_force_async(FmmmNmmState&, const FmmmNmmOptions&, cudaStream_t);
void* fmmm_nmm_workspace_create();
void fmmm_nmm_workspace_release(void*) noexcept;
int fmmm_nmm_workspace_build_count(const void*) noexcept;
int fmmm_nmm_workspace_map_build_count(const void*) noexcept;
int fmmm_nmm_workspace_map_launch_count(const void*) noexcept;
FmmmStageResult fmmm_pack_components(FmmmPackingState&, const FmmmPackingOptions&);

} // namespace lcn::release
