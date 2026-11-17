#pragma once

#include "fmmm_pipeline_contract.h"
#include "fmmm_advanced_uncoarsen.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace lcn::release {

// Host-side control table. The caller allocates the level array; build owns
// each level's plan/state device SoA (cudaMalloc). Execute may free parent
// plan arrays after child consume; fmmm_release_hierarchy frees what remains.
// This struct itself performs no allocation.
struct FmmmHierarchyLevel {
    int level = 0;
    std::uint64_t prng_before = 0;
    std::uint64_t prng_after = 0;
    oracle::FmmmAdvancedUncoarsenPlanView plan{};
    oracle::FmmmAdvancedUncoarsenStateView state{};
};

struct FmmmHierarchyState {
    FmmmHierarchyLevel* levels = nullptr;
    int level_count = 0;
    int capacity = 0;
};

FmmmStageResult fmmm_build_hierarchy(FmmmHierarchyState& state,
                                     const FmmmGraphView& graph,
                                     const FmmmProfile& profile);
FmmmStageResult fmmm_execute_hierarchy(FmmmHierarchyState& state);

// Frees every level's plan/state device arrays allocated by build/execute.
// Idempotent on already-null pointers. Callers (including diagnostics) must
// invoke this after consuming downloaded endpoints.
void fmmm_release_hierarchy(FmmmHierarchyState& state);

// Device-resident hierarchy ABI.  The caller owns every pointer in the input
// and output views for the complete session lifetime.  The launch helper only
// submits kernels and device-to-device initialization; it never downloads a
// count, position, geometry, or status value to the host.
struct FmmmDeviceHierarchyInput {
    const double* position_x = nullptr;
    const double* position_y = nullptr;
    int node_count = 0;
    double down_left_x = 0.0;
    double down_left_y = 0.0;
    double box_length = 1.0;
    std::uint64_t position_epoch = 0;
    std::uint64_t box_epoch = 0;
};

struct FmmmDeviceHierarchyOutput {
    double* cell_down_left_x = nullptr;
    double* cell_down_left_y = nullptr;
    double* cell_box_length = nullptr;
    double* cell_center_x = nullptr;
    double* cell_center_y = nullptr;
    int* cell_parent = nullptr;
    int* cell_children4 = nullptr;
    int* cell_level = nullptr;
    int* cell_compact_id = nullptr;
    int* cell_particle_count = nullptr;
    int* particle_to_leaf = nullptr;
    int* cell_count_device = nullptr;
    int* root_count_device = nullptr;
    std::uint64_t* tree_epoch_device = nullptr;
    std::uint64_t* position_epoch_device = nullptr;
    std::uint64_t* box_epoch_device = nullptr;
    int* status_device = nullptr;
};

struct FmmmDeviceHierarchyLaunchConfig {
    int cell_capacity = 0;
    int max_depth = 0;
    int leaf_capacity = 32;
    std::uint64_t tree_epoch = 0;
};

// Builds a deterministic breadth-first quadtree in device memory.  The tree
// is fully materialized to the selected depth, so parent/child geometry and
// particle-to-leaf assignment are independent of launch order.  The return
// value reports only launch/API errors; semantic status is written to
// output.status_device and is therefore still device-resident.
cudaError_t fmmm_launch_device_hierarchy(
    const FmmmDeviceHierarchyInput& input,
    const FmmmDeviceHierarchyLaunchConfig& config,
    const FmmmDeviceHierarchyOutput& output,
    cudaStream_t stream = nullptr);

} // namespace lcn::release
