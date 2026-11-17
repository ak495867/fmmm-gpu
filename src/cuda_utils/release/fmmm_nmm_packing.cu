// G5: NMM repulsive-force evaluation and MAAR component packing.
//
// NMM reference: third_party/ogdf/src/ogdf/energybased/fmmm/NewMultipoleMethod.cpp
// Packing reference: third_party/ogdf/src/ogdf/energybased/fmmm/MAARPacking.cpp
//                   third_party/ogdf/src/ogdf/energybased/FMMMLayout.cpp
//
// Rules:
//   - NMM is selected only when node_count >= 175 AND options.using_nmm.
//   - Packing emits raw doubles; G1 alone quantizes.
//   - No CPU force/score fallback, no partial state.
//   - Exact OGDF float ordering preserved throughout.

#include "fmmm_nmm_packing.h"
#include "fmmm_nmm_tree_device.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <vector>

namespace lcn::release {

namespace {

bool g1_parallel_wsprl_enabled() noexcept {
    const char* value = std::getenv("LCN_FMMM_G1_PARALLEL_WSPRL");
    return value && std::strcmp(value, "1") == 0;
}

// Host-side quadtree depth cap: avoids 64 empty level launches on small n.
// Margin covers unbalanced splits; still << 64 for Automatic-10 (n=180).
int nmm_host_level_cap(int node_count) noexcept {
    if (node_count <= 1) return 1;
    int cap = 1;
    while (cap < 63 && (1 << cap) < node_count) ++cap;
    return std::min(64, cap + 6);
}

// The interaction-tile implementation is still an experiment.  The latest
// end-to-end qualification showed regressions on the production NMM fixtures,
// so it must never become the implicit default until a fresh A/B is accepted.
// Keeping the switch runtime-selectable also makes NCU comparisons explicit.
bool experimental_nmm_force_tile_enabled() noexcept {
    const char* value = std::getenv("LCN_FMMM_NMM_FORCE_IMPL");
    return value && std::strcmp(value, "tiled") == 0;
}

// G6-large candidate: one warp is the scheduling unit for a target-leaf
// particle tile.  This is intentionally separate from the rejected G3.1
// 128-thread CTA path: large leaves are still grouped by contained_nodes, but
// small leaves no longer pay for three idle warps at every source barrier.
bool experimental_nmm_force_warp_tile_enabled() noexcept {
    const char* value = std::getenv("LCN_FMMM_NMM_FORCE_IMPL");
    return value && std::strcmp(value, "large_warp_tile") == 0;
}

FmmmStageResult enqueue_parallel_tree_topology(
        FmmmNmmState& state, const FmmmNmmOptions& options, cudaStream_t stream) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (!state.device_build_counters || !state.device_build_queue ||
            !state.device_wsprl_queue || !state.temp_contained_nodes ||
            !state.temp_contained_begin || !state.temp_contained_scratch ||
            !state.contained_begin || !state.contained_nodes || state.max_cells <= 0 ||
            state.max_contained <= 0 || state.queue_capacity < state.max_cells) {
        result.reason = "parallel frontier workspace contract is incomplete";
        return result;
    }
    constexpr int threads = 256;
    const int blocks = std::max(1, (state.max_cells + threads - 1) / threads);
    const int init_blocks = std::max(1, (state.node_count + threads - 1) / threads);
    int* current_frontier = state.device_build_queue;
    int* next_frontier = state.device_wsprl_queue;
    int* child_counts = state.temp_contained_scratch;
    int* scan_temp = state.device_wsprl_queue + state.max_cells;
    const std::size_t scan_temp_capacity = static_cast<std::size_t>(
            std::max(0, state.traversal_queue_capacity - state.max_cells)) * sizeof(int);
    std::size_t scan_bytes = 0;
    cudaError_t err = cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes, state.contained_begin, state.D1_begin,
            state.max_cells + 1, stream);
    if (err != cudaSuccess || scan_bytes > scan_temp_capacity) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "parallel frontier scan workspace is insufficient";
        return result;
    }

    nmm_frontier_init_kernel<<<init_blocks, threads, 0, stream>>>(
            state, state.device_build_counters,
            state.device_build_counters + kNmmBuildOverflowBase,
            state.max_cells, state.max_contained,
            options.down_left_x, options.down_left_y,
            options.down_left_x + options.box_length,
            options.down_left_y + options.box_length,
            current_frontier, state.status);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }

    const int leaf_size = options.particles_in_leaf > 0 ? options.particles_in_leaf : 25;
    const int level_cap = nmm_host_level_cap(state.node_count);
    for (int level = 0; level < level_cap; ++level) {
        err = cudaMemsetAsync(state.contained_begin, 0,
                static_cast<std::size_t>(state.max_cells + 1) * sizeof(int), stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_frontier_classify_prepare_kernel<<<blocks, threads, 0, stream>>>(
                state, state.device_build_counters, leaf_size,
                current_frontier, state.max_cells, child_counts,
                state.contained_begin, state.D1_begin);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        err = cub::DeviceScan::ExclusiveSum(
                scan_temp, scan_bytes, state.contained_begin, state.D2_begin,
                state.max_cells + 1, stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        err = cub::DeviceScan::ExclusiveSum(
                scan_temp, scan_bytes, state.D1_begin, state.M_begin,
                state.max_cells + 1, stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_frontier_scatter_kernel<<<blocks, threads, 0, stream>>>(
                state, state.device_build_counters,
                state.device_build_counters + kNmmBuildOverflowBase,
                leaf_size, current_frontier, state.max_cells, child_counts,
                state.D2_begin, state.M_begin,
                next_frontier, state.contained_begin, state.status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        err = cub::DeviceScan::ExclusiveSum(
                scan_temp, scan_bytes, state.contained_begin, state.I_begin,
                state.max_cells + 1, stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_frontier_compact_update_kernel<<<blocks, threads, 0, stream>>>(
                state.device_build_counters, current_frontier, state.max_cells,
                leaf_size, child_counts, state.D2_begin, state.M_begin,
                state.I_begin, next_frontier, state.status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        std::swap(current_frontier, next_frontier);
    }

    // Canonicalize the breadth-first cell allocation to the exact OGDF
    // preorder contract.  All passes remain device-side and level-parallel:
    // subtree sizes are reduced bottom-up, then preorder ranks are assigned
    // top-down before a pair of grid-stride snapshots applies the permutation.
    if (state.max_contained < state.max_cells * 4 || !state.local_re ||
            !state.cell_center_x || !state.cell_center_y || !state.I_nodes) {
        result.status = FmmmPipelineStatus::invalid_input;
        result.reason = "parallel frontier preorder workspace is incomplete";
        return result;
    }
    int* subtree = state.temp_contained_scratch;
    int* rank = state.temp_contained_scratch + state.max_cells;
    nmm_tree_subtree_init_kernel<<<blocks, threads, 0, stream>>>(
            state.device_build_counters, state.max_cells, subtree);
    for (int level = level_cap - 1; level >= 1; --level) {
        nmm_tree_subtree_accumulate_kernel<<<blocks, threads, 0, stream>>>(
                state, state.device_build_counters, state.max_cells, level,
                subtree);
    }
    for (int level = 0; level < level_cap; ++level) {
        nmm_tree_preorder_rank_kernel<<<blocks, threads, 0, stream>>>(
                state, state.device_build_counters, state.max_cells, level,
                rank, subtree);
    }
    nmm_tree_reorder_snapshot_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, rank, subtree);
    nmm_tree_reorder_apply_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, rank, subtree);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }

    nmm_leaf_count_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, state.contained_begin);
    err = cub::DeviceScan::ExclusiveSum(
            scan_temp, scan_bytes, state.contained_begin, state.I_begin,
            state.max_cells + 1, stream);
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    nmm_leaf_scatter_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, state.I_begin);
    nmm_finalize_contained_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, state.I_begin);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "multi-block frontier topology enqueued";
    return result;
}

FmmmStageResult enqueue_parallel_wsprl(FmmmNmmState& state, cudaStream_t stream) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (!state.device_build_counters || !state.temp_contained_scratch ||
            !state.device_wsprl_queue || state.max_cells <= 0 ||
            state.max_contained < state.max_cells * 4) {
        result.reason = "parallel WSPRL workspace contract is incomplete";
        return result;
    }
    // Count planes: 4*(max_cells+1).  CUB temp follows in the same queue buffer.
    const int stride = state.max_cells + 1;
    const std::size_t count_items = static_cast<std::size_t>(stride) * 4u;
    int* counts = state.device_wsprl_queue;
    int* scan_temp = state.device_wsprl_queue + static_cast<std::ptrdiff_t>(count_items);
    const std::size_t scan_temp_capacity =
            static_cast<std::size_t>(std::max(0,
                state.traversal_queue_capacity - static_cast<int>(count_items))) *
            sizeof(int);

    std::size_t scan_bytes = 0;
    cudaError_t err = cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, counts, state.I_begin, stride, stream);
    if (err != cudaSuccess || scan_bytes > scan_temp_capacity) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "parallel WSPRL scan workspace is insufficient";
        return result;
    }

    // Multipole persistent leaves max cell level in counters[7].  Bound WSPRL
    // to that depth so trailing empty levels cannot zero begin[cell_count].
    int max_level = 0;
    err = cudaMemcpyAsync(&max_level, state.device_build_counters + 7, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    const int level_cap = std::max(
        1, std::min(nmm_host_level_cap(state.node_count), std::max(0, max_level) + 1));

    const int threads = 256;
    const int blocks = std::max(1, (state.max_cells + threads - 1) / threads);
    // Four begin snapshots occupy the first 4*max_cells scratch entries.
    nmm_wsprl_snapshot_begin_all_kernel<<<blocks, threads, 0, stream>>>(
        state.I_begin, state.D1_begin, state.D2_begin, state.M_begin,
        state.max_cells, state.max_cells, state.temp_contained_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }

    int* outputs[4] = {state.I_begin, state.D1_begin, state.D2_begin, state.M_begin};
    for (int level = 0; level < level_cap; ++level) {
        err = cudaMemsetAsync(counts, 0, count_items * sizeof(int), stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_wsprl_count_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, level,
            state.traversal_queue_capacity, counts,
            &state.device_build_counters[kNmmBuildOverflowBase], state.status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        for (int category = 0; category < 4; ++category) {
            err = cub::DeviceScan::ExclusiveSum(
                scan_temp, scan_bytes, counts + category * stride, outputs[category],
                stride, stream);
            if (err != cudaSuccess) {
                result.status = FmmmPipelineStatus::backend_error;
                result.reason = cudaGetErrorString(err);
                return result;
            }
        }
        nmm_wsprl_restore_completed_begin_all_kernel<<<blocks, threads, 0, stream>>>(
            state.cell_level, state.device_build_counters, state.max_cells, level,
            state.temp_contained_scratch,
            state.I_begin, state.D1_begin, state.D2_begin, state.M_begin);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_wsprl_scatter_kernel<<<blocks, threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, level,
            state.traversal_queue_capacity, counts,
            state.temp_contained_scratch + 0 * state.max_cells,
            state.temp_contained_scratch + 1 * state.max_cells,
            &state.device_build_counters[kNmmBuildOverflowBase], state.status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        nmm_wsprl_snapshot_begin_all_kernel<<<blocks, threads, 0, stream>>>(
            state.I_begin, state.D1_begin, state.D2_begin, state.M_begin,
            state.max_cells, state.max_cells, state.temp_contained_scratch);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
    }
    nmm_wsprl_finalize_kernel<<<1, 1, 0, stream>>>(
        state.device_build_counters, state.max_cells,
        state.I_begin, state.D1_begin, state.D2_begin, state.M_begin,
        state.device_build_counters);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "fused WSPRL count/scan/scatter enqueued";
    return result;
}

// The synchronous allocator/copy helpers below are reference-only.  The
// production session is required to enter fmmm_nmm_build_async/force_async
// with a complete session-owned contract; those paths never call these
// wrappers.  Keeping the legacy ABI behind named wrappers makes accidental
// production use fail-closed in review and keeps its audit surface explicit.
template <typename T>
cudaError_t nmm_reference_cuda_malloc(T** pointer, std::size_t bytes) {
    return cudaMalloc(pointer, bytes);
}

template <typename T>
cudaError_t nmm_reference_cuda_free(T* pointer) {
    return cudaFree(pointer);
}

cudaError_t nmm_reference_cuda_memcpy(void* destination, const void* source,
                                      std::size_t bytes, cudaMemcpyKind direction) {
    return cudaMemcpy(destination, source, bytes, direction);
}

cudaError_t nmm_reference_cuda_sync() {
    return cudaDeviceSynchronize();
}

const char* nmm_fault_injection() noexcept {
    const char* value = std::getenv("LCN_FMMM_NMM_FAULT");
    return value != nullptr ? value : "";
}

bool fault_is(const char* name) noexcept {
    return std::strcmp(nmm_fault_injection(), name) == 0;
}

struct NmmPersistentWorkspace {
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
    int* device_build_counters = nullptr;
    int* device_build_queue = nullptr;
    int* device_wsprl_queue = nullptr;
    std::size_t cap_device_build_counters = 0, cap_device_build_queue = 0;
    std::size_t cap_device_wsprl_queue = 0;
    std::size_t cap_cell_downleft_x = 0, cap_cell_downleft_y = 0, cap_cell_box_length = 0;
    std::size_t cap_cell_level = 0, cap_cell_parent = 0, cap_cell_child4 = 0, cap_cell_particle_count = 0;
    std::size_t cap_cell_center_x = 0, cap_cell_center_y = 0;
    std::size_t cap_contained_begin = 0, cap_contained_nodes = 0;
    std::size_t cap_temp_contained_nodes = 0, cap_temp_contained_scratch = 0;
    std::size_t cap_temp_contained_begin = 0;
    std::size_t cap_I_begin = 0, cap_I_nodes = 0, cap_D1_begin = 0, cap_D1_nodes = 0;
    std::size_t cap_D2_begin = 0, cap_D2_nodes = 0, cap_M_begin = 0, cap_M_nodes = 0;
    std::size_t cap_multipole_re = 0, cap_multipole_im = 0;
    std::size_t cap_local_re = 0, cap_local_im = 0;
    std::size_t cap_particle_to_leaf = 0;
    int build_count = 0;
    int map_build_count = 0;
    int map_launch_count = 0;
    bool epoch_initialized = false;
    std::uint64_t graph_epoch = 0, level_epoch = 0, particle_order_epoch = 0;
    std::uint64_t tree_epoch = 0, map_epoch = 0;
    int cell_count = 0, contained_node_count = 0, I_count = 0;
    int D1_count = 0, D2_count = 0, M_count = 0;
    int precision = 0, expansion_stride = 0;
};

template <typename T>
bool ensure_device_capacity(T*& ptr, std::size_t& capacity, std::size_t needed) {
    if (needed <= capacity) return true;
    if (ptr && nmm_reference_cuda_free(ptr) != cudaSuccess) return false;
    ptr = nullptr;
    if (needed == 0) { capacity = 0; return true; }
    if (nmm_reference_cuda_malloc(&ptr, needed * sizeof(T)) != cudaSuccess) {
        ptr = nullptr; capacity = 0; return false;
    }
    capacity = needed;
    return true;
}

template <typename T>
bool upload_cached(T*& ptr, std::size_t& capacity, const std::vector<T>& host) {
    if (!ensure_device_capacity(ptr, capacity, host.size())) return false;
    if (!host.empty() && nmm_reference_cuda_memcpy(ptr, host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}

void release_workspace(NmmPersistentWorkspace& w) noexcept {
    auto free_d = [](double*& p) { if (p) { nmm_reference_cuda_free(p); p = nullptr; } };
    auto free_i = [](int*& p) { if (p) { nmm_reference_cuda_free(p); p = nullptr; } };
    free_d(w.cell_downleft_x); free_d(w.cell_downleft_y); free_d(w.cell_box_length);
    free_i(w.cell_level); free_i(w.cell_parent); free_i(w.cell_child4); free_i(w.cell_particle_count);
    free_d(w.cell_center_x); free_d(w.cell_center_y);
    free_i(w.contained_begin); free_i(w.contained_nodes); free_i(w.I_begin); free_i(w.I_nodes);
    free_i(w.temp_contained_nodes); free_i(w.temp_contained_scratch);
    free_i(w.temp_contained_begin);
    free_i(w.D1_begin); free_i(w.D1_nodes); free_i(w.D2_begin); free_i(w.D2_nodes);
    free_i(w.M_begin); free_i(w.M_nodes);
    free_d(w.multipole_re); free_d(w.multipole_im); free_d(w.local_re); free_d(w.local_im);
    free_i(w.particle_to_leaf);
    free_i(w.device_build_counters); free_i(w.device_build_queue);
    free_i(w.device_wsprl_queue);
    w = NmmPersistentWorkspace{};
}

} // namespace

// ---------------------------------------------------------------------------
// NMM QuadTree cell: each cell is a node in a pre-order flat array.
// ---------------------------------------------------------------------------

// MAAR packing row (mirrors OGDF PackingRowInfo).
struct PackingRow {
    double max_height = 0.0;
    double total_width = 0.0;
    int row_index = 0;
};

// ---------------------------------------------------------------------------
// kernel_maar_pack: single-threaded kernel for MAAR packing
// ---------------------------------------------------------------------------

__device__ static double calc_aspect_ratio_area(double width, double height, double aspect_ratio) {
    double ratio = (height == 0.0) ? 1e300 : width / height;
    if (ratio < aspect_ratio)
        return width * height * (aspect_ratio / ratio);
    else
        return width * height * (ratio / aspect_ratio);
}

__device__ static bool better_tip_new_row(double r_w, double r_h,
                                double area_w, double area_h,
                                double aspect_ratio, int allow_tip_over,
                                double& best_area) {
    double w = area_w > r_w ? area_w : r_w;
    double h = area_h + r_h;
    best_area = calc_aspect_ratio_area(w, h, aspect_ratio);
    if (allow_tip_over == 0) return false;
    double w2 = area_w > r_h ? area_w : r_h;
    double h2 = area_h + r_w;
    double act_area = calc_aspect_ratio_area(w2, h2, aspect_ratio);
    if (act_area < 0.99999 * best_area) {
        best_area = act_area;
        return true;
    }
    return false;
}

__device__ static bool better_tip_this_row(double r_w, double r_h,
                                 double area_w, double area_h,
                                 double aspect_ratio, int allow_tip_over,
                                 double row_total_width, double row_max_height,
                                 double& best_area) {
    double w = area_w > row_total_width + r_w ? area_w : row_total_width + r_w;
    double h = area_h > area_h - row_max_height + r_h ? area_h : area_h - row_max_height + r_h;
    best_area = calc_aspect_ratio_area(w, h, aspect_ratio);
    if (allow_tip_over == 0) return false;
    if (allow_tip_over == 1 && r_w > row_max_height) return false;
    double w2 = area_w > row_total_width + r_h ? area_w : row_total_width + r_h;
    double h2 = area_h > area_h - row_max_height + r_w ? area_h : area_h - row_max_height + r_w;
    double act_area = calc_aspect_ratio_area(w2, h2, aspect_ratio);
    if (act_area < 0.99999 * best_area) {
        best_area = act_area;
        return true;
    }
    return false;
}

__device__ void device_stable_sort_descending(int* rect_order, int nc, const double* values) {
    for (int i = 1; i < nc; ++i) {
        int key = rect_order[i];
        double key_val = values[key];
        int j = i - 1;
        while (j >= 0 && values[rect_order[j]] < key_val) {
            rect_order[j + 1] = rect_order[j];
            j = j - 1;
        }
        rect_order[j + 1] = key;
    }
}

__global__ void kernel_maar_pack(
    int nc, int nn,
    double* pos_x, double* pos_y,
    const double* node_width, const double* node_height,
    const int* comp_begin, const int* comp_nodes,
    double page_ratio, int presort, int allow_tip, double min_dist,
    double* rect_width, double* rect_height,
    double* rect_old_dlc_x, double* rect_old_dlc_y,
    double* rect_new_dlc_x, double* rect_new_dlc_y,
    int* rect_tipped, int* rect_order,
    double* row_max_height, double* row_total_width,
    int* row_of_rect, double* row_y_min, double* act_row_x_max,
    double* out_rect_left, double* out_rect_bottom, double* out_rect_right, double* out_rect_top,
    double* out_old_dlc_x, double* out_old_dlc_y, double* out_new_dlc_x, double* out_new_dlc_y,
    int* out_tipped, int* out_row_index, double* out_translate_x, double* out_translate_y,
    int* out_chosen_candidate, double* out_chosen_angle)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 1. Compute bounding rectangle for each component
    for (int c = 0; c < nc; ++c) {
        int begin = comp_begin[c];
        int end = comp_begin[c + 1];
        if (begin >= end) {
            rect_width[c] = 0; rect_height[c] = 0;
            rect_old_dlc_x[c] = 0; rect_old_dlc_y[c] = 0;
            rect_tipped[c] = 0;
            continue;
        }
        int v0 = comp_nodes[begin];
        double max_bound = (node_width[v0] > node_height[v0] ? node_width[v0] : node_height[v0]) / 2.0;
        double x_min = pos_x[v0] - max_bound, x_max = pos_x[v0] + max_bound;
        double y_min = pos_y[v0] - max_bound, y_max = pos_y[v0] + max_bound;
        for (int ni = begin + 1; ni < end; ++ni) {
            int v = comp_nodes[ni];
            max_bound = (node_width[v] > node_height[v] ? node_width[v] : node_height[v]) / 2.0;
            double ax_min = pos_x[v] - max_bound, ax_max = pos_x[v] + max_bound;
            double ay_min = pos_y[v] - max_bound, ay_max = pos_y[v] + max_bound;
            if (ax_min < x_min) x_min = ax_min;
            if (ax_max > x_max) x_max = ax_max;
            if (ay_min < y_min) y_min = ay_min;
            if (ay_max > y_max) y_max = ay_max;
        }
        rect_width[c] = x_max - x_min;
        rect_height[c] = y_max - y_min;
        rect_old_dlc_x[c] = x_min;
        rect_old_dlc_y[c] = y_min;
        rect_tipped[c] = 0;
    }

    // Add min_dist
    for (int c = 0; c < nc; ++c) {
        rect_width[c] += min_dist;
        rect_height[c] += min_dist;
        rect_order[c] = c;
        if (out_chosen_candidate) out_chosen_candidate[c] = 0;
        if (out_chosen_angle) out_chosen_angle[c] = 0.0;
    }

    // Presort
    if (presort == 1) {
        device_stable_sort_descending(rect_order, nc, rect_height);
    } else if (presort == 2) {
        device_stable_sort_descending(rect_order, nc, rect_width);
    }

    // 2. Packing state
    int num_rows = 0;
    double area_w = 0, area_h = 0;
    double aspect_ratio_area = 0.0;
    for (int i = 0; i < nc; ++i) row_of_rect[i] = -1;

    for (int i = 0; i < nc; ++i) {
        int ri = rect_order[i];
        double rw = rect_width[ri];
        double rh = rect_height[ri];

        int best_try = 1;
        if (num_rows == 0) {
            double dummy_best = 0;
            bool tip = better_tip_new_row(rw, rh, area_w, area_h, page_ratio, allow_tip, dummy_best);
            best_try = tip ? 2 : 1;
            if (tip) {
                double tipped_dlc_x = rect_old_dlc_y[ri] * (-1.0) - rh;
                double tipped_dlc_y = rect_old_dlc_x[ri];
                rect_old_dlc_x[ri] = tipped_dlc_x;
                rect_old_dlc_y[ri] = tipped_dlc_y;
                double tmp = rect_width[ri]; rect_width[ri] = rect_height[ri]; rect_height[ri] = tmp;
                rect_tipped[ri] = !rect_tipped[ri];
                rw = rect_width[ri]; rh = rect_height[ri];
            }
            double max_w = area_w > rw ? area_w : rw;
            aspect_ratio_area = calc_aspect_ratio_area(max_w, area_h + rh, page_ratio);
            row_max_height[num_rows] = rh;
            row_total_width[num_rows] = rw;
            area_w = max_w;
            area_h += rh;
            row_of_rect[ri] = num_rows;
            num_rows++;
        } else {
            // OGDF find_Best_Fit_insert_position: better_tipp_in_new_row writes
            // the new-row aspect area into aspect_ratio_area before BF compare.
            double new_row_area = 0;
            bool tip_new = better_tip_new_row(rw, rh, area_w, area_h, page_ratio, allow_tip, new_row_area);
            best_try = tip_new ? 2 : 1;
            aspect_ratio_area = new_row_area;

            int bf_idx = 0;
            for (int r = 1; r < num_rows; ++r) {
                if (row_total_width[r] < row_total_width[bf_idx]) bf_idx = r;
            }

            double bf_row_area = 0;
            bool tip_bf = better_tip_this_row(rw, rh, area_w, area_h, page_ratio, allow_tip,
                                               row_total_width[bf_idx], row_max_height[bf_idx], bf_row_area);
            int index_2 = tip_bf ? 4 : 3;

            // OGDF: nearly_equal uses ~1e-10 relative on aspect_ratio_area vs area_2.
            const double delta = 1e-10;
            double small_a, big_a;
            if (bf_row_area > 0) {
                small_a = bf_row_area * (1.0 - delta);
                big_a = bf_row_area * (1.0 + delta);
            } else {
                small_a = bf_row_area * (1.0 + delta);
                big_a = bf_row_area * (1.0 - delta);
            }
            const bool nearly = (small_a <= aspect_ratio_area) && (aspect_ratio_area <= big_a);
            if (bf_row_area <= aspect_ratio_area || nearly) {
                aspect_ratio_area = bf_row_area;
                best_try = index_2;
            }

            if (best_try == 2 || best_try == 4) {
                double tipped_dlc_x, tipped_dlc_y;
                if (!rect_tipped[ri]) {
                    tipped_dlc_x = rect_old_dlc_y[ri] * (-1.0) - rh;
                    tipped_dlc_y = rect_old_dlc_x[ri];
                } else {
                    tipped_dlc_x = rect_old_dlc_y[ri];
                    tipped_dlc_y = rect_old_dlc_x[ri] * (-1.0) - rw;
                }
                rect_old_dlc_x[ri] = tipped_dlc_x;
                rect_old_dlc_y[ri] = tipped_dlc_y;
                double tmp = rect_width[ri]; rect_width[ri] = rect_height[ri]; rect_height[ri] = tmp;
                rect_tipped[ri] = !rect_tipped[ri];
                rw = rect_width[ri]; rh = rect_height[ri];
            }

            if (best_try == 1 || best_try == 2) {
                row_max_height[num_rows] = rh;
                row_total_width[num_rows] = rw;
                area_w = area_w > rw ? area_w : rw;
                area_h += rh;
                row_of_rect[ri] = num_rows;
                num_rows++;
            } else {
                double old_max_height = row_max_height[bf_idx];
                row_max_height[bf_idx] = old_max_height > rh ? old_max_height : rh;
                row_total_width[bf_idx] += rw;
                area_w = area_w > row_total_width[bf_idx] ? area_w : row_total_width[bf_idx];
                area_h = area_h > area_h - old_max_height + rh ? area_h : area_h - old_max_height + rh;
                row_of_rect[ri] = bf_idx;
            }
        }
        if (out_chosen_candidate) out_chosen_candidate[ri] = best_try;
        if (out_chosen_angle) out_chosen_angle[ri] = rect_tipped[ri] ? 90.0 : 0.0;
    }

    // 3. Export new positions
    if (num_rows > 0) {
        row_y_min[0] = 0.0;
        act_row_x_max[0] = 0.0;
    }
    for (int r = 1; r < num_rows; ++r) {
        row_y_min[r] = row_y_min[r - 1] + row_max_height[r - 1];
        act_row_x_max[r] = 0.0;
    }

    for (int i = 0; i < nc; ++i) {
        int ri = rect_order[i];
        int row_i = row_of_rect[ri];
        double new_x = act_row_x_max[row_i];
        act_row_x_max[row_i] += rect_width[ri];
        double new_y = row_y_min[row_i] + (row_max_height[row_i] - rect_height[ri]) / 2.0;
        rect_new_dlc_x[ri] = new_x;
        rect_new_dlc_y[ri] = new_y;
    }

    // 4. Transform node positions and populate output arrays
    for (int c = 0; c < nc; ++c) {
        rect_width[c] -= min_dist;
        rect_height[c] -= min_dist;
    }

    for (int c = 0; c < nc; ++c) {
        int begin = comp_begin[c];
        int end = comp_begin[c + 1];
        if (rect_tipped[c]) {
            for (int ni = begin; ni < end; ++ni) {
                int v = comp_nodes[ni];
                double tx = pos_x[v], ty = pos_y[v];
                pos_x[v] = -ty;
                pos_y[v] = tx;
            }
        }
        double dx = rect_new_dlc_x[c] - rect_old_dlc_x[c];
        double dy = rect_new_dlc_y[c] - rect_old_dlc_y[c];
        for (int ni = begin; ni < end; ++ni) {
            int v = comp_nodes[ni];
            pos_x[v] += dx;
            pos_y[v] += dy;
        }

        // populate outputs
        if (out_rect_left) out_rect_left[c] = rect_old_dlc_x[c];
        if (out_rect_bottom) out_rect_bottom[c] = rect_old_dlc_y[c];
        if (out_rect_right) out_rect_right[c] = rect_old_dlc_x[c] + rect_width[c];
        if (out_rect_top) out_rect_top[c] = rect_old_dlc_y[c] + rect_height[c];
        if (out_old_dlc_x) out_old_dlc_x[c] = rect_old_dlc_x[c];
        if (out_old_dlc_y) out_old_dlc_y[c] = rect_old_dlc_y[c];
        if (out_new_dlc_x) out_new_dlc_x[c] = rect_new_dlc_x[c];
        if (out_new_dlc_y) out_new_dlc_y[c] = rect_new_dlc_y[c];
        if (out_tipped) out_tipped[c] = rect_tipped[c];
        if (out_row_index) out_row_index[c] = row_of_rect[c];
        if (out_translate_x) out_translate_x[c] = dx;
        if (out_translate_y) out_translate_y[c] = dy;
        if (out_chosen_angle) out_chosen_angle[c] = rect_tipped[c] ? 90.0 : 0.0;
    }
}

// ---------------------------------------------------------------------------
// fmmm_pack_components: launches kernel_maar_pack
// ---------------------------------------------------------------------------
FmmmStageResult fmmm_pack_components(FmmmPackingState& state,
                                      const FmmmPackingOptions& options) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;

    const int nc = state.component_count;
    if (nc <= 0) {
        result.reason = "component_count must be positive";
        return result;
    }
    const int nn = state.node_count;

    // Allocate scratch memory
    double *d_rect_width, *d_rect_height, *d_rect_old_dlc_x, *d_rect_old_dlc_y;
    double *d_rect_new_dlc_x, *d_rect_new_dlc_y;
    int *d_rect_tipped, *d_rect_order;
    double *d_row_max_height, *d_row_total_width;
    int *d_row_of_rect;
    double *d_row_y_min, *d_act_row_x_max;

    nmm_reference_cuda_malloc(&d_rect_width, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_height, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_old_dlc_x, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_old_dlc_y, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_new_dlc_x, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_new_dlc_y, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_rect_tipped, nc * sizeof(int));
    nmm_reference_cuda_malloc(&d_rect_order, nc * sizeof(int));
    nmm_reference_cuda_malloc(&d_row_max_height, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_row_total_width, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_row_of_rect, nc * sizeof(int));
    nmm_reference_cuda_malloc(&d_row_y_min, nc * sizeof(double));
    nmm_reference_cuda_malloc(&d_act_row_x_max, nc * sizeof(double));

    kernel_maar_pack<<<1, 1>>>(
        nc, nn,
        state.position_x, state.position_y,
        state.node_width, state.node_height,
        state.component_begin, state.component_nodes,
        options.page_ratio, static_cast<int>(options.presort_components), static_cast<int>(options.tip_over_components), options.min_dist_cc,
        d_rect_width, d_rect_height,
        d_rect_old_dlc_x, d_rect_old_dlc_y,
        d_rect_new_dlc_x, d_rect_new_dlc_y,
        d_rect_tipped, d_rect_order,
        d_row_max_height, d_row_total_width,
        d_row_of_rect, d_row_y_min, d_act_row_x_max,
        state.rect_left, state.rect_bottom, state.rect_right, state.rect_top,
        state.old_dlc_x, state.old_dlc_y, state.new_dlc_x, state.new_dlc_y,
        state.tipped, state.row_index, state.translate_x, state.translate_y,
        state.chosen_candidate, state.chosen_angle
    );

    cudaError_t err = cudaGetLastError();
    nmm_reference_cuda_sync();

    nmm_reference_cuda_free(d_rect_width);
    nmm_reference_cuda_free(d_rect_height);
    nmm_reference_cuda_free(d_rect_old_dlc_x);
    nmm_reference_cuda_free(d_rect_old_dlc_y);
    nmm_reference_cuda_free(d_rect_new_dlc_x);
    nmm_reference_cuda_free(d_rect_new_dlc_y);
    nmm_reference_cuda_free(d_rect_tipped);
    nmm_reference_cuda_free(d_rect_order);
    nmm_reference_cuda_free(d_row_max_height);
    nmm_reference_cuda_free(d_row_total_width);
    nmm_reference_cuda_free(d_row_of_rect);
    nmm_reference_cuda_free(d_row_y_min);
    nmm_reference_cuda_free(d_act_row_x_max);

    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }

    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::packing;
    result.reason = "ok";
    return result;
}

// ---------------------------------------------------------------------------
// NMM force: package builds the reduced quadtree into the declared ABI arrays
// when geometry is absent, then evaluates leaf traversal contributions on GPU.
// ---------------------------------------------------------------------------

// OGDF transform_multipole_exp_to_forces:
// sum = a0/z - sum_{k=1..p} k*a_k / z^{k+1}; force=(Re,-Im)
__device__ static void accum_multipole_to_particle(
    double x_v, double y_v,
    double cx, double cy,
    const double* mp_re, const double* mp_im,
    int precision,
    double& fx, double& fy)
{
    double zr = x_v - cx, zi = y_v - cy;
    double zr2 = zr * zr + zi * zi;
    if (zr2 < 1e-300) return;

    // one_over_z = 1/z
    const double oz_r = zr / zr2;
    const double oz_i = -zi / zr2;
    double inv_r = oz_r;
    double inv_i = oz_i;
    double sr = mp_re[0] * inv_r - mp_im[0] * inv_i;
    double si = mp_re[0] * inv_i + mp_im[0] * inv_r;
    for (int k = 1; k <= precision; ++k) {
        // inv *= 1/z  -> 1/z^{k+1}
        const double nr = inv_r * oz_r - inv_i * oz_i;
        const double ni = inv_r * oz_i + inv_i * oz_r;
        inv_r = nr;
        inv_i = ni;
        sr -= (double)k * (mp_re[k] * inv_r - mp_im[k] * inv_i);
        si -= (double)k * (mp_re[k] * inv_i + mp_im[k] * inv_r);
    }
    fx += sr;
    fy -= si;
}

// OGDF transform_local_exp_to_forces on the leaf's own local expansion.
__device__ static void accum_local_to_particle(
    double x_v, double y_v,
    double cx, double cy,
    const double* loc_re, const double* loc_im,
    int precision,
    double& fx, double& fy)
{
    double zr = x_v - cx, zi = y_v - cy;
    double sr = 0.0, si = 0.0;
    double pow_r = 1.0, pow_i = 0.0; // (z_v - z_0)^{k-1}
    for (int k = 1; k <= precision; ++k) {
        sr += (double)k * (loc_re[k] * pow_r - loc_im[k] * pow_i);
        si += (double)k * (loc_re[k] * pow_i + loc_im[k] * pow_r);
        double nr = pow_r * zr - pow_i * zi;
        double ni = pow_r * zi + pow_i * zr;
        pow_r = nr;
        pow_i = ni;
    }
    fx += sr;
    fy -= si;
}

// Direct particle-particle repulsion (used in leaf cells and near neighbours).
__device__ static void accum_direct(
    double x_v, double y_v,
    double x_u, double y_u,
    double& fx, double& fy)
{
    double dx = x_v - x_u, dy = y_v - y_u;
    double dist2 = dx * dx + dy * dy;
    if (dist2 < 1e-20) return;
    double dist = sqrt(dist2);
    // OGDF: F_rep = (v_pos - u_pos) / dist^2  (unit repulsion)
    double scale = 1.0 / dist2;
    fx += dx * scale;
    fy += dy * scale;
}

__global__ void kernel_nmm_build_particle_to_leaf(
    int cell_count, const int* __restrict__ cell_child4,
    const int* __restrict__ contained_begin, const int* __restrict__ contained_nodes,
    int* __restrict__ particle_to_leaf) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count) return;
    const int base = cell * 4;
    if (cell_child4[base] >= 0 || cell_child4[base + 1] >= 0 ||
        cell_child4[base + 2] >= 0 || cell_child4[base + 3] >= 0) return;
    for (int i = contained_begin[cell]; i < contained_begin[cell + 1]; ++i) {
        particle_to_leaf[contained_nodes[i]] = cell;
    }
}

__global__ void kernel_nmm_build_particle_to_leaf_device_counts(
    const int* __restrict__ build_counters, int max_cells,
    const int* __restrict__ cell_child4,
    const int* __restrict__ contained_begin,
    const int* __restrict__ contained_nodes,
    int* __restrict__ particle_to_leaf) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_count = build_counters ? build_counters[0] : max_cells;
    if (cell_count < 0) cell_count = 0;
    if (cell_count > max_cells) cell_count = max_cells;
    if (cell >= cell_count) return;
    const int base = cell * 4;
    if (cell_child4[base] >= 0 || cell_child4[base + 1] >= 0 ||
        cell_child4[base + 2] >= 0 || cell_child4[base + 3] >= 0) return;
    for (int i = contained_begin[cell]; i < contained_begin[cell + 1]; ++i)
        particle_to_leaf[contained_nodes[i]] = cell;
}

// NMM force kernel: one thread per particle.
// Each particle collects:
//   1. Local expansion contributions from cells in its leaf's L-list.
//   2. Multipole contributions from cells in its leaf's M-list.
//   3. Direct forces from particles in cells in its leaf's D1/D2-lists.
//   4. Direct forces from particles in the same leaf (contained_nodes).
__global__ void kernel_nmm_force(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    int node_count,
    const double* __restrict__ cell_dlc_x,
    const double* __restrict__ cell_dlc_y,
    const double* __restrict__ cell_box_length,
    const int* __restrict__ cell_level,
    const int* __restrict__ cell_parent,
    const int* __restrict__ cell_child4,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const int* __restrict__ I_begin,
    const int* __restrict__ I_nodes,
    const int* __restrict__ D1_begin,
    const int* __restrict__ D1_nodes,
    const int* __restrict__ D2_begin,
    const int* __restrict__ D2_nodes,
    const int* __restrict__ M_begin,
    const int* __restrict__ M_nodes,
    const int* __restrict__ contained_begin,
    const int* __restrict__ contained_nodes,
    const double* __restrict__ mp_re,
    const double* __restrict__ mp_im,
    const double* __restrict__ loc_re,
    const double* __restrict__ loc_im,
    const int* __restrict__ particle_to_leaf,
    int precision,
    int expansion_stride,
    const int* __restrict__ cell_particle_count,
    int cell_count,
    const int* __restrict__ device_build_counters,
    const int* __restrict__ status)
{
    if (status && *status != 0) return;
    if (device_build_counters && device_build_counters[6] != 0) return;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= node_count) return;

    double x_v = pos_x[v], y_v = pos_y[v];
    double fx = 0.0, fy = 0.0;

    (void)cell_dlc_x; (void)cell_dlc_y; (void)cell_box_length;
    (void)cell_level; (void)cell_parent; (void)cell_child4;
    (void)cell_particle_count;
    if (device_build_counters) {
        cell_count = device_build_counters[0];
        if (cell_count < 0) cell_count = 0;
        if (cell_count > 0 && cell_count > 100000000) cell_count = 100000000;
    }
    int leaf_cell = particle_to_leaf ? particle_to_leaf[v] : -1;
    if (!particle_to_leaf) {
        for (int c = 0; c < cell_count && leaf_cell < 0; ++c) {
            const int base = c * 4;
            if (cell_child4[base] >= 0 || cell_child4[base + 1] >= 0 ||
                cell_child4[base + 2] >= 0 || cell_child4[base + 3] >= 0) continue;
            for (int i = contained_begin[c]; i < contained_begin[c + 1]; ++i) {
                if (contained_nodes[i] == v) { leaf_cell = c; break; }
            }
        }
    }
    if (leaf_cell < 0) {
        force_x[v] = fx; force_y[v] = fy;
        return;
    }

    const int p = precision;
    const int stride = expansion_stride;
    (void)I_begin;
    (void)I_nodes;

    // 1. Leaf local expansion (OGDF transform_local_exp_to_forces).
    accum_local_to_particle(x_v, y_v,
        cell_cx[leaf_cell], cell_cy[leaf_cell],
        loc_re + leaf_cell * stride, loc_im + leaf_cell * stride, p,
        fx, fy);

    // 2. Multipole contributions (M-list of leaf_cell).
    {
        int mb = M_begin[leaf_cell], me = M_begin[leaf_cell + 1];
        for (int ni = mb; ni < me; ++ni) {
            int cell = M_nodes[ni];
            accum_multipole_to_particle(x_v, y_v,
                cell_cx[cell], cell_cy[cell],
                mp_re + cell * stride, mp_im + cell * stride, p,
                fx, fy);
        }
    }

    // 3. Direct forces from D1-list (near cells, direct particle-particle).
    {
        int d1b = D1_begin[leaf_cell], d1e = D1_begin[leaf_cell + 1];
        for (int ni = d1b; ni < d1e; ++ni) {
            int cell = D1_nodes[ni];
            int cb = contained_begin[cell], ce = contained_begin[cell + 1];
            for (int pi = cb; pi < ce; ++pi) {
                int u = contained_nodes[pi];
                if (u == v) continue;
                accum_direct(x_v, y_v, pos_x[u], pos_y[u], fx, fy);
            }
        }
    }

    // 4. Direct forces from D2-list.
    {
        int d2b = D2_begin[leaf_cell], d2e = D2_begin[leaf_cell + 1];
        for (int ni = d2b; ni < d2e; ++ni) {
            int cell = D2_nodes[ni];
            int cb = contained_begin[cell], ce = contained_begin[cell + 1];
            for (int pi = cb; pi < ce; ++pi) {
                int u = contained_nodes[pi];
                if (u == v) continue;
                accum_direct(x_v, y_v, pos_x[u], pos_y[u], fx, fy);
            }
        }
    }

    // 5. Direct forces from particles in same leaf cell.
    {
        int cb = contained_begin[leaf_cell], ce = contained_begin[leaf_cell + 1];
        for (int pi = cb; pi < ce; ++pi) {
            int u = contained_nodes[pi];
            if (u == v) continue;
            accum_direct(x_v, y_v, pos_x[u], pos_y[u], fx, fy);
        }
    }

    force_x[v] = fx;
    force_y[v] = fy;
}

// Interaction-tile candidate (G3.1): one CTA owns one target leaf cell.
// Every output particle in that leaf remains exclusively owned by one thread,
// preserving the original summation order and avoiding atomics.  Direct
// source particles are loaded in fixed-size chunks into shared memory so all
// target particles in the CTA reuse the same source positions.
__global__ void kernel_nmm_force_interaction_tile(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    int node_count,
    const int* __restrict__ cell_child4,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const int* __restrict__ D1_begin,
    const int* __restrict__ D1_nodes,
    const int* __restrict__ D2_begin,
    const int* __restrict__ D2_nodes,
    const int* __restrict__ contained_begin,
    const int* __restrict__ contained_nodes,
    const int* __restrict__ M_begin,
    const int* __restrict__ M_nodes,
    const double* __restrict__ mp_re,
    const double* __restrict__ mp_im,
    const double* __restrict__ loc_re,
    const double* __restrict__ loc_im,
    const int* __restrict__ particle_to_leaf,
    int precision,
    int expansion_stride,
    int cell_count,
    const int* __restrict__ device_build_counters,
    const int* __restrict__ status)
{
    extern __shared__ double source_tile[];
    double* sx = source_tile;
    double* sy = source_tile + blockDim.x;
    if (status && *status != 0) return;
    if (device_build_counters && device_build_counters[6] != 0) return;

    int actual_cells = cell_count;
    if (device_build_counters && device_build_counters[0] > 0) {
        actual_cells = device_build_counters[0];
        if (actual_cells < 0) actual_cells = 0;
        if (actual_cells > cell_count) actual_cells = cell_count;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int p = precision;
    const int stride = expansion_stride;
    // The launch uses a bounded number of persistent tile workers rather than
    // workspace capacity.  Each worker strides over the actual device cell
    // count, so capacity-sized empty CTAs are not launched.
    for (int leaf = static_cast<int>(blockIdx.x);
         leaf < actual_cells; leaf += static_cast<int>(gridDim.x)) {
        const int child_base = leaf * 4;
        if (cell_child4[child_base] >= 0 || cell_child4[child_base + 1] >= 0 ||
            cell_child4[child_base + 2] >= 0 || cell_child4[child_base + 3] >= 0) continue;

        const int out_begin = contained_begin[leaf];
        const int out_end = contained_begin[leaf + 1];
        const int out_count = out_end - out_begin;

        // Batch target particles so every thread reaches every barrier even
        // when a leaf contains more than one CTA tile of outputs.
        for (int out_base = 0; out_base < out_count; out_base += blockDim.x) {
        const int out_offset = out_base + lane;
        const bool active = out_offset < out_count;
        const int v = active ? contained_nodes[out_begin + out_offset] : -1;
        const double x_v = active ? pos_x[v] : 0.0;
        const double y_v = active ? pos_y[v] : 0.0;
        double fx = 0.0, fy = 0.0;

        if (active) {
            accum_local_to_particle(x_v, y_v, cell_cx[leaf], cell_cy[leaf],
                loc_re + leaf * stride, loc_im + leaf * stride, p, fx, fy);
            const int mb = M_begin[leaf];
            const int me = M_begin[leaf + 1];
            for (int ni = mb; ni < me; ++ni) {
                const int cell = M_nodes[ni];
                accum_multipole_to_particle(x_v, y_v, cell_cx[cell], cell_cy[cell],
                    mp_re + cell * stride, mp_im + cell * stride, p, fx, fy);
            }
        }

        // Keep the exact D1, D2, same-leaf order.  Only source position loads
        // are shared; arithmetic remains private to the owning output thread.
        for (int pass = 0; pass < 3; ++pass) {
            const int* list_begin = pass == 0 ? D1_begin : D2_begin;
            const int* list_nodes = pass == 0 ? D1_nodes : D2_nodes;
            const int lb = pass < 2 ? list_begin[leaf] : 0;
            const int le = pass < 2 ? list_begin[leaf + 1] : 1;
            for (int ni = lb; ni < le; ++ni) {
                const int source_cell = pass < 2 ? list_nodes[ni] : leaf;
                const int cb = contained_begin[source_cell];
                const int ce = contained_begin[source_cell + 1];
                for (int base = cb; base < ce; base += blockDim.x) {
                    const int count = min(static_cast<int>(blockDim.x), ce - base);
                    if (lane < count) {
                        const int u = contained_nodes[base + lane];
                        sx[lane] = pos_x[u];
                        sy[lane] = pos_y[u];
                    }
                    __syncthreads();
                    if (active) {
                        for (int j = 0; j < count; ++j) {
                            const int u = contained_nodes[base + j];
                            if (u != v) accum_direct(x_v, y_v, sx[j], sy[j], fx, fy);
                        }
                    }
                    __syncthreads();
                }
            }
        }
            if (active) {
                force_x[v] = fx;
                force_y[v] = fy;
            }
        }
    }
}

// G6-large candidate.  A single warp owns one target-leaf tile (up to 32
// particles).  The target particle keeps the legacy accumulation order:
// local -> M -> D1 -> D2 -> same leaf.  Only source indices and coordinates
// are cooperatively staged, so the arithmetic and output ownership remain
// deterministic and no cross-particle atomics are needed.
__global__ void kernel_nmm_force_large_warp_tile(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    double* __restrict__ force_x,
    double* __restrict__ force_y,
    int node_count,
    const int* __restrict__ cell_child4,
    const double* __restrict__ cell_cx,
    const double* __restrict__ cell_cy,
    const int* __restrict__ D1_begin,
    const int* __restrict__ D1_nodes,
    const int* __restrict__ D2_begin,
    const int* __restrict__ D2_nodes,
    const int* __restrict__ M_begin,
    const int* __restrict__ M_nodes,
    const int* __restrict__ contained_begin,
    const int* __restrict__ contained_nodes,
    const double* __restrict__ mp_re,
    const double* __restrict__ mp_im,
    const double* __restrict__ loc_re,
    const double* __restrict__ loc_im,
    const int* __restrict__ particle_to_leaf,
    int precision,
    int expansion_stride,
    int cell_count,
    const int* __restrict__ device_build_counters,
    const int* __restrict__ status)
{
    extern __shared__ unsigned char shared_bytes[];
    int* source_index = reinterpret_cast<int*>(shared_bytes);
    double* source_x = reinterpret_cast<double*>(source_index + 32);
    double* source_y = source_x + 32;
    if (status && *status != 0) return;
    if (device_build_counters && device_build_counters[6] != 0) return;

    int actual_cells = cell_count;
    if (device_build_counters && device_build_counters[0] > 0) {
        actual_cells = device_build_counters[0];
        if (actual_cells < 0) actual_cells = 0;
        if (actual_cells > cell_count) actual_cells = cell_count;
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int p = precision;
    const int stride = expansion_stride;

    // The launch is deliberately bounded by node_count.  The loop over the
    // actual device cell count avoids capacity-sized empty work while keeping
    // the dispatch fully asynchronous (no host readback of cell_count).
    for (int leaf = static_cast<int>(blockIdx.x);
         leaf < actual_cells; leaf += static_cast<int>(gridDim.x)) {
        const int child_base = leaf * 4;
        if (cell_child4[child_base] >= 0 || cell_child4[child_base + 1] >= 0 ||
            cell_child4[child_base + 2] >= 0 || cell_child4[child_base + 3] >= 0) {
            continue;
        }

        const int out_begin = contained_begin[leaf];
        const int out_end = contained_begin[leaf + 1];
        const int out_count = out_end - out_begin;
        for (int out_base = 0; out_base < out_count; out_base += 32) {
            const int out_offset = out_base + lane;
            const bool active = out_offset < out_count;
            const int v = active ? contained_nodes[out_begin + out_offset] : -1;
            const double x_v = active ? pos_x[v] : 0.0;
            const double y_v = active ? pos_y[v] : 0.0;
            double fx = 0.0;
            double fy = 0.0;

            if (active) {
                accum_local_to_particle(
                    x_v, y_v, cell_cx[leaf], cell_cy[leaf],
                    loc_re + leaf * stride, loc_im + leaf * stride, p, fx, fy);
                const int mb = M_begin[leaf];
                const int me = M_begin[leaf + 1];
                for (int ni = mb; ni < me; ++ni) {
                    const int cell = M_nodes[ni];
                    accum_multipole_to_particle(
                        x_v, y_v, cell_cx[cell], cell_cy[cell],
                        mp_re + cell * stride, mp_im + cell * stride,
                        p, fx, fy);
                }
            }

            for (int pass = 0; pass < 3; ++pass) {
                const int* list_begin = pass == 0 ? D1_begin : D2_begin;
                const int* list_nodes = pass == 0 ? D1_nodes : D2_nodes;
                const int lb = pass < 2 ? list_begin[leaf] : 0;
                const int le = pass < 2 ? list_begin[leaf + 1] : 1;
                for (int ni = lb; ni < le; ++ni) {
                    const int source_cell = pass < 2 ? list_nodes[ni] : leaf;
                    const int cb = contained_begin[source_cell];
                    const int ce = contained_begin[source_cell + 1];
                    for (int base = cb; base < ce; base += 32) {
                        const int count = min(32, ce - base);
                        if (lane < count) {
                            const int u = contained_nodes[base + lane];
                            source_index[lane] = u;
                            source_x[lane] = pos_x[u];
                            source_y[lane] = pos_y[u];
                        }
                        __syncwarp();
                        if (active) {
                            for (int j = 0; j < count; ++j) {
                                if (source_index[j] != v) {
                                    accum_direct(
                                        x_v, y_v, source_x[j], source_y[j], fx, fy);
                                }
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            if (active) {
                force_x[v] = fx;
                force_y[v] = fy;
            }
        }
    }
}

__global__ void latch_nmm_build_status_kernel(const int* build_counters, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !build_counters || !status) return;
    if (build_counters[6] != 0) *status = static_cast<int>(FmmmPipelineStatus::backend_error);
}

namespace {

FmmmStageResult enqueue_session_tree_build(FmmmNmmState& state,
                                           const FmmmNmmOptions& options,
                                           cudaStream_t stream,
                                           bool force_rebuild) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (!state.device_build_counters || !state.device_build_queue || !state.device_wsprl_queue ||
            state.max_cells <= 0 || state.max_contained <= 0 || state.max_list <= 0) {
        result.reason = "session-owned NMM scratch is incomplete";
        return result;
    }
    if (!state.cell_downleft_x || !state.cell_downleft_y ||
            !state.cell_box_length || !state.cell_child4 ||
            !state.contained_begin || !state.contained_nodes ||
            !state.temp_contained_nodes || !state.temp_contained_scratch ||
            !state.temp_contained_begin ||
            !state.I_begin || !state.I_nodes || !state.D1_begin || !state.D1_nodes ||
            !state.D2_begin || !state.D2_nodes || !state.M_begin || !state.M_nodes ||
            !state.multipole_re || !state.multipole_im ||
            !state.local_re || !state.local_im) {
        result.reason = "session-owned NMM arrays are incomplete";
        return result;
    }

    NmmPersistentWorkspace* persistent =
            static_cast<NmmPersistentWorkspace*>(state.persistent_workspace);
    bool rebuild = force_rebuild || persistent == nullptr || !persistent->epoch_initialized;
    if (persistent && persistent->epoch_initialized && !force_rebuild) {
        rebuild = persistent->graph_epoch != state.graph_epoch ||
                  persistent->level_epoch != state.level_epoch ||
                  persistent->particle_order_epoch != state.particle_order_epoch ||
                  persistent->tree_epoch != state.tree_epoch;
    }

    if (fault_is("allocation") || fault_is("upload") || fault_is("stale_epoch")) {
        result.reason = "fault injection: NMM device build";
        return result;
    }

    state.precision = options.precision > 0 ? options.precision : 4;
    state.expansion_stride = state.precision + 1;
    cudaError_t err = cudaSuccess;
    // The counters are the authoritative device result of the tree build.
    // Do not clear them on a force-only reuse call: doing so would make a
    // valid tree look empty without any explicit failure.
    if (rebuild) {
        err = cudaMemsetAsync(
            state.device_build_counters, 0,
            static_cast<std::size_t>(kNmmBuildStorageCount) * sizeof(int), stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        if (state.status) {
            err = cudaMemsetAsync(state.status, 0, sizeof(int), stream);
            if (err != cudaSuccess) {
                result.status = FmmmPipelineStatus::backend_error;
                result.reason = cudaGetErrorString(err);
                return result;
            }
        }
    }

    if (rebuild) {
        const FmmmStageResult topology = enqueue_parallel_tree_topology(state, options, stream);
        if (topology.status != FmmmPipelineStatus::ok) return topology;
        const int phase_threads = 256;
        const int phase_blocks = std::max(1, (state.max_cells + phase_threads - 1) / phase_threads);
        nmm_set_centers_kernel<<<phase_blocks, phase_threads, 0, stream>>>(
            state, state.device_build_counters, state.max_cells, options.seed);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        int device = 0;
        cudaDeviceProp device_properties{};
        int cooperative_blocks_per_sm = 0;
#if defined(LCN_DISABLE_G2_PERSISTENT)
        const bool cooperative_available = false;
#else
        const bool cooperative_available =
                cudaGetDevice(&device) == cudaSuccess &&
                cudaGetDeviceProperties(&device_properties, device) == cudaSuccess &&
                device_properties.cooperativeLaunch &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &cooperative_blocks_per_sm,
                        nmm_multipole_persistent_kernel,
                        phase_threads, 0) == cudaSuccess &&
                cooperative_blocks_per_sm > 0;
#endif
        if (cooperative_available) {
            const int cooperative_blocks = std::max(1, std::min(
                    phase_blocks,
                    cooperative_blocks_per_sm * device_properties.multiProcessorCount));
            void* kernel_args[] = {
                &state, &state.device_build_counters, &state.max_cells};
            err = cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(nmm_multipole_persistent_kernel),
                    cooperative_blocks, phase_threads, kernel_args, 0, stream);
            if (err != cudaSuccess) {
                result.status = FmmmPipelineStatus::backend_error;
                result.reason = cudaGetErrorString(err);
                return result;
            }
        } else {
            // Explicit compatibility fallback for devices without cooperative
            // launch support.  It preserves the accepted G1 semantics.
            const int multipole_levels = nmm_host_level_cap(state.node_count);
            for (int level = multipole_levels - 1; level >= 0; --level) {
                nmm_multipole_level_kernel<<<phase_blocks, phase_threads, 0, stream>>>(
                    state, state.device_build_counters, state.max_cells, level);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    result.status = FmmmPipelineStatus::backend_error;
                    result.reason = cudaGetErrorString(err);
                    return result;
                }
            }
        }
        for (int* begin : {state.I_begin, state.D1_begin, state.D2_begin, state.M_begin}) {
            err = cudaMemsetAsync(begin, 0,
                                  static_cast<std::size_t>(state.max_cells + 1) * sizeof(int),
                                  stream);
            if (err != cudaSuccess) {
                result.status = FmmmPipelineStatus::backend_error;
                result.reason = cudaGetErrorString(err);
                return result;
            }
        }
        const FmmmStageResult wsprl = enqueue_parallel_wsprl(state, stream);
        if (wsprl.status != FmmmPipelineStatus::ok) return wsprl;
        state.tree_rebuilt = 1;
        if (persistent) {
            persistent->epoch_initialized = true;
            persistent->graph_epoch = state.graph_epoch;
            persistent->level_epoch = state.level_epoch;
            persistent->particle_order_epoch = state.particle_order_epoch;
            persistent->tree_epoch = state.tree_epoch;
            ++persistent->build_count;
        }
    } else {
        state.tree_rebuilt = 0;
    }

    const bool rebuild_map = rebuild || !persistent || persistent->map_build_count == 0 ||
                             persistent->map_epoch != state.map_epoch;
    if (rebuild_map) {
        err = cudaMemsetAsync(state.particle_to_leaf, 0xff,
                              static_cast<std::size_t>(state.node_count) * sizeof(int), stream);
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        const int blocks = std::max(1, (state.max_cells + 127) / 128);
        kernel_nmm_build_particle_to_leaf_device_counts<<<blocks, 128, 0, stream>>>(
            state.device_build_counters, state.max_cells,
            state.cell_child4, state.contained_begin, state.contained_nodes,
            state.particle_to_leaf);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = cudaGetErrorString(err);
            return result;
        }
        if (persistent) {
            ++persistent->map_build_count;
            ++persistent->map_launch_count;
            persistent->map_epoch = state.map_epoch;
        }
        state.particle_to_leaf_rebuilt = 1;
        state.particle_to_leaf_launches = 1;
    } else {
        state.particle_to_leaf_rebuilt = 0;
        state.particle_to_leaf_launches = 0;
    }

    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "device NMM tree/list build enqueued";
    return result;
}

} // namespace

FmmmStageResult fmmm_nmm_bind_workspace(FmmmNmmState& state,
                                        void* opaque_workspace) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    result.completed_stage = FmmmPipelineStage::none;
    if (!opaque_workspace || state.node_count < 175) {
        result.reason = "NMM workspace bind requires a persistent workspace and node_count >= 175";
        return result;
    }
    auto* workspace = static_cast<NmmPersistentWorkspace*>(opaque_workspace);
    const std::size_t max_cells = static_cast<std::size_t>(state.node_count) * 15u + 1000u;
    const std::size_t max_contained = std::max(
        static_cast<std::size_t>(state.node_count) * 30u + 1000u,
        max_cells * 4u);
    const std::size_t max_list = max_cells * 30u;
    const std::size_t stride = state.expansion_stride > 0
            ? static_cast<std::size_t>(state.expansion_stride) : 5u;

    auto alloc_d = [&](double*& ptr, std::size_t& cap, std::size_t needed) {
        return ensure_device_capacity(ptr, cap, needed);
    };
    auto alloc_i = [&](int*& ptr, std::size_t& cap, std::size_t needed) {
        return ensure_device_capacity(ptr, cap, needed);
    };
    const bool allocated =
        alloc_d(workspace->cell_downleft_x, workspace->cap_cell_downleft_x, max_cells) &&
        alloc_d(workspace->cell_downleft_y, workspace->cap_cell_downleft_y, max_cells) &&
        alloc_d(workspace->cell_box_length, workspace->cap_cell_box_length, max_cells) &&
        alloc_i(workspace->cell_level, workspace->cap_cell_level, max_cells) &&
        alloc_i(workspace->cell_parent, workspace->cap_cell_parent, max_cells) &&
        alloc_i(workspace->cell_child4, workspace->cap_cell_child4, max_cells * 4u) &&
        alloc_i(workspace->cell_particle_count, workspace->cap_cell_particle_count, max_cells) &&
        alloc_d(workspace->cell_center_x, workspace->cap_cell_center_x, max_cells) &&
        alloc_d(workspace->cell_center_y, workspace->cap_cell_center_y, max_cells) &&
        alloc_i(workspace->contained_begin, workspace->cap_contained_begin, max_cells + 1u) &&
        alloc_i(workspace->contained_nodes, workspace->cap_contained_nodes, max_contained) &&
        alloc_i(workspace->temp_contained_nodes, workspace->cap_temp_contained_nodes, max_contained) &&
        alloc_i(workspace->temp_contained_scratch, workspace->cap_temp_contained_scratch, max_contained) &&
        alloc_i(workspace->temp_contained_begin, workspace->cap_temp_contained_begin, max_cells + 1u) &&
        alloc_i(workspace->I_begin, workspace->cap_I_begin, max_cells + 1u) &&
        alloc_i(workspace->I_nodes, workspace->cap_I_nodes, max_list) &&
        alloc_i(workspace->D1_begin, workspace->cap_D1_begin, max_cells + 1u) &&
        alloc_i(workspace->D1_nodes, workspace->cap_D1_nodes, max_list) &&
        alloc_i(workspace->D2_begin, workspace->cap_D2_begin, max_cells + 1u) &&
        alloc_i(workspace->D2_nodes, workspace->cap_D2_nodes, max_list) &&
        alloc_i(workspace->M_begin, workspace->cap_M_begin, max_cells + 1u) &&
        alloc_i(workspace->M_nodes, workspace->cap_M_nodes, max_list) &&
        alloc_d(workspace->multipole_re, workspace->cap_multipole_re, max_cells * stride) &&
        alloc_d(workspace->multipole_im, workspace->cap_multipole_im, max_cells * stride) &&
        alloc_d(workspace->local_re, workspace->cap_local_re, max_cells * stride) &&
        alloc_d(workspace->local_im, workspace->cap_local_im, max_cells * stride) &&
        alloc_i(workspace->particle_to_leaf, workspace->cap_particle_to_leaf,
                static_cast<std::size_t>(state.node_count)) &&
        alloc_i(workspace->device_build_counters, workspace->cap_device_build_counters,
                static_cast<std::size_t>(kNmmBuildStorageCount)) &&
        alloc_i(workspace->device_build_queue, workspace->cap_device_build_queue, max_cells) &&
        alloc_i(workspace->device_wsprl_queue, workspace->cap_device_wsprl_queue, max_list);
    if (!allocated) {
        release_workspace(*workspace);
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "persistent NMM workspace allocation failed";
        return result;
    }

    state.persistent_workspace = opaque_workspace;
    state.cell_downleft_x = workspace->cell_downleft_x;
    state.cell_downleft_y = workspace->cell_downleft_y;
    state.cell_box_length = workspace->cell_box_length;
    state.cell_level = workspace->cell_level;
    state.cell_parent = workspace->cell_parent;
    state.cell_child4 = workspace->cell_child4;
    state.cell_particle_count = workspace->cell_particle_count;
    state.cell_center_x = workspace->cell_center_x;
    state.cell_center_y = workspace->cell_center_y;
    state.contained_begin = workspace->contained_begin;
    state.contained_nodes = workspace->contained_nodes;
    state.temp_contained_nodes = workspace->temp_contained_nodes;
    state.temp_contained_scratch = workspace->temp_contained_scratch;
    state.temp_contained_begin = workspace->temp_contained_begin;
    state.I_begin = workspace->I_begin;
    state.I_nodes = workspace->I_nodes;
    state.D1_begin = workspace->D1_begin;
    state.D1_nodes = workspace->D1_nodes;
    state.D2_begin = workspace->D2_begin;
    state.D2_nodes = workspace->D2_nodes;
    state.M_begin = workspace->M_begin;
    state.M_nodes = workspace->M_nodes;
    state.multipole_re = workspace->multipole_re;
    state.multipole_im = workspace->multipole_im;
    state.local_re = workspace->local_re;
    state.local_im = workspace->local_im;
    state.particle_to_leaf = workspace->particle_to_leaf;
    state.device_build_counters = workspace->device_build_counters;
    state.device_build_counter_capacity =
        static_cast<int>(workspace->cap_device_build_counters);
    state.device_build_queue = workspace->device_build_queue;
    state.device_wsprl_queue = workspace->device_wsprl_queue;
    state.max_cells = static_cast<int>(max_cells);
    state.max_contained = static_cast<int>(max_contained);
    state.max_list = static_cast<int>(max_list);
    state.queue_capacity = static_cast<int>(max_list);
    state.frontier_queue_capacity = static_cast<int>(max_cells);
    state.traversal_queue_capacity = static_cast<int>(max_list);
    state.max_I = static_cast<int>(max_list);
    state.max_D1 = static_cast<int>(max_list);
    state.max_D2 = static_cast<int>(max_list);
    state.max_M = static_cast<int>(max_list);
    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "persistent NMM workspace bound";
    return result;
}

FmmmStageResult fmmm_nmm_validate_preallocated_workspace(const FmmmNmmState& state) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    result.completed_stage = FmmmPipelineStage::none;

    // These views are the explicit session-lifetime ownership boundary.  The
    // queue and counters are allocated by the session together with the tree
    // arrays; their presence distinguishes the resident route from the
    // standalone/reference ABI, which may still supply host-uploaded arrays.
    if (!state.persistent_workspace || !state.device_build_counters ||
            state.device_build_counter_capacity < kNmmBuildStorageCount ||
            !state.device_build_queue || state.max_cells <= 0 ||
            state.max_contained < std::max(1, state.node_count) ||
            state.max_list <= 0 || state.queue_capacity <= 0 ||
            state.queue_capacity > state.max_list ||
            state.frontier_queue_capacity <= 0 ||
            state.traversal_queue_capacity <= 0 ||
            state.max_I <= 0 || state.max_D1 <= 0 ||
            state.max_D2 <= 0 || state.max_M <= 0) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "session-owned NMM capacity contract is incomplete";
        return result;
    }
    if (!state.position_x || !state.position_y || !state.force_x || !state.force_y ||
            !state.cell_downleft_x || !state.cell_downleft_y ||
            !state.cell_box_length || !state.cell_child4 ||
            !state.contained_begin || !state.contained_nodes ||
            !state.temp_contained_nodes || !state.temp_contained_scratch ||
            !state.temp_contained_begin ||
            !state.I_begin || !state.I_nodes || !state.D1_begin || !state.D1_nodes ||
            !state.D2_begin || !state.D2_nodes || !state.M_begin || !state.M_nodes ||
            !state.multipole_re || !state.multipole_im ||
            !state.local_re || !state.local_im || !state.particle_to_leaf) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "session-owned NMM arrays are incomplete";
        return result;
    }

    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "session-owned NMM capacity contract is valid";
    return result;
}

FmmmStageResult fmmm_nmm_build_async(FmmmNmmState& state,
                                      const FmmmNmmOptions& options,
                                      cudaStream_t stream) {
    if (!options.using_nmm) {
        FmmmStageResult result;
        result.status = FmmmPipelineStatus::ok;
        result.completed_stage = FmmmPipelineStage::normal_force;
        result.reason = "NMM not selected";
        return result;
    }
    if (state.node_count < 175) {
        FmmmStageResult result;
        result.status = FmmmPipelineStatus::ok;
        result.completed_stage = FmmmPipelineStage::normal_force;
        result.reason = "node_count below NMM threshold";
        return result;
    }
    const FmmmStageResult contract = fmmm_nmm_validate_preallocated_workspace(state);
    if (contract.status != FmmmPipelineStatus::ok) return contract;
    return enqueue_session_tree_build(state, options, stream, true);
}

FmmmStageResult fmmm_nmm_force_async(FmmmNmmState& state,
                                     const FmmmNmmOptions& options,
                                     cudaStream_t stream) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;
    if (!options.using_nmm) {
        result.reason = "using_nmm is false; NMM force not selected";
        return result;
    }
    if (state.node_count < 175) {
        result.status = FmmmPipelineStatus::ok;
        result.completed_stage = FmmmPipelineStage::normal_force;
        result.reason = "node_count below NMM threshold";
        return result;
    }
    const FmmmStageResult contract = fmmm_nmm_validate_preallocated_workspace(state);
    if (contract.status != FmmmPipelineStatus::ok) return contract;

    result = enqueue_session_tree_build(state, options, stream, false);
    if (result.status != FmmmPipelineStatus::ok) return result;
    // G3.1 remains an explicit experiment only.  The latest independent
    // qualification found end-to-end regressions when it was the default.
    // Production therefore uses the previously qualified node-parallel
    // kernel unless the caller opts into the tiled candidate at runtime.
    if (experimental_nmm_force_warp_tile_enabled()) {
    constexpr int kForceTileThreads = 32;
    const int blocks = std::max(1, std::min(state.max_cells,
                                             std::max(1, state.node_count)));
    kernel_nmm_force_large_warp_tile<<<blocks, kForceTileThreads,
                                        32 * sizeof(int) + 64 * sizeof(double),
                                        stream>>>(
        state.position_x, state.position_y, state.force_x, state.force_y,
        state.node_count, state.cell_child4, state.cell_center_x,
        state.cell_center_y, state.D1_begin, state.D1_nodes,
        state.D2_begin, state.D2_nodes, state.M_begin, state.M_nodes,
        state.contained_begin,
        state.contained_nodes, state.multipole_re, state.multipole_im,
        state.local_re, state.local_im, state.particle_to_leaf,
        state.precision, state.expansion_stride,
        state.cell_count > 0 ? state.cell_count : state.max_cells,
        state.cell_count > 0 ? nullptr : state.device_build_counters,
        state.status);
    } else if (!experimental_nmm_force_tile_enabled()) {
    constexpr int kForceTileThreads = 256;
    const int blocks = std::max(1, (state.node_count + kForceTileThreads - 1) /
                                      kForceTileThreads);
    kernel_nmm_force<<<blocks, kForceTileThreads, 0, stream>>>(
        state.position_x, state.position_y, state.force_x, state.force_y,
        state.node_count, state.cell_downleft_x, state.cell_downleft_y,
        state.cell_box_length, state.cell_level, state.cell_parent,
        state.cell_child4, state.cell_center_x, state.cell_center_y,
        state.I_begin, state.I_nodes, state.D1_begin, state.D1_nodes,
        state.D2_begin, state.D2_nodes, state.M_begin, state.M_nodes,
        state.contained_begin, state.contained_nodes, state.multipole_re,
        state.multipole_im, state.local_re, state.local_im,
        state.particle_to_leaf, state.precision, state.expansion_stride,
        state.cell_particle_count, state.max_cells, state.device_build_counters,
        state.status);
    } else {
    constexpr int kForceTileThreads = 128;
    const int blocks = std::max(1, std::min(state.max_cells,
                                             std::max(1, state.node_count)));
    kernel_nmm_force_interaction_tile<<<blocks, kForceTileThreads,
                                        2 * kForceTileThreads * sizeof(double),
                                        stream>>>(
        state.position_x, state.position_y, state.force_x, state.force_y,
        state.node_count, state.cell_child4, state.cell_center_x,
        state.cell_center_y, state.D1_begin, state.D1_nodes,
        state.D2_begin, state.D2_nodes, state.contained_begin,
        state.contained_nodes, state.M_begin, state.M_nodes,
        state.multipole_re, state.multipole_im, state.local_re,
        state.local_im, state.particle_to_leaf, state.precision,
        state.expansion_stride,
        state.cell_count > 0 ? state.cell_count : state.max_cells,
        state.cell_count > 0 ? nullptr : state.device_build_counters,
        state.status);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }
    latch_nmm_build_status_kernel<<<1, 1, 0, stream>>>(
        state.device_build_counters, state.status);
    if (cudaGetLastError() != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = "NMM build status latch launch failed";
        return result;
    }
    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "device NMM force enqueued";
    return result;
}

FmmmStageResult fmmm_nmm_force(FmmmNmmState& state,
                                const FmmmNmmOptions& options) {
    FmmmStageResult result;
    result.status = FmmmPipelineStatus::invalid_input;

    if (!options.using_nmm) {
        result.reason = "using_nmm is false; NMM force not selected";
        return result;
    }
    if (state.node_count < 175) {
        result.reason = "node_count < 175; NMM is not selected for small graphs";
        return result;
    }
    // Fault injection is an ABI contract and must be observed before pointer
    // validation, so callers can verify that no mutation/allocation occurs.
    if (fault_is("allocation")) {
        result.reason = "fault injection: allocation";
        return result;
    }
    if (fault_is("upload")) {
        result.reason = "fault injection: upload";
        return result;
    }
    if (fault_is("stale_epoch")) {
        result.reason = "fault injection: stale epoch";
        return result;
    }
    if (!state.position_x || !state.position_y || !state.force_x || !state.force_y) {
        result.reason = "missing position/force device pointers";
        return result;
    }

    // The production resident session always supplies the complete
    // session-owned contract.  Never enter the legacy synchronous/reference
    // implementation in that case: it contains compatibility-only staging
    // and capacity growth.  The async call below performs only device memset,
    // kernel launches, and event/error enqueue operations; all buffers were
    // reserved by fmmm_session_create.
    if (state.persistent_workspace) {
        const FmmmStageResult contract = fmmm_nmm_validate_preallocated_workspace(state);
        if (contract.status != FmmmPipelineStatus::ok) {
            // A session owner is authoritative: never downgrade an invalid
            // session view to the compatibility allocator below.
            return contract;
        }
        return fmmm_nmm_force_async(state, options, nullptr);
    }

    state.tree_rebuilt = 0;
    state.particle_to_leaf_rebuilt = 0;
    state.particle_to_leaf_launches = 0;

    // Faults are injected before any output or persistent workspace mutation.
    // This is intentionally a deterministic contract test seam, not a
    // production error path: callers can prove that allocation/upload and
    // stale-epoch failures publish no partial tree or force state.
    // Standalone/reference callers do not own a session arena.  Give that
    // compatibility path the same allocation bookkeeping as the resident
    // path, while keeping the state pointer null so it cannot enter the
    // resident async validation route.  The binding owns the resulting
    // device pointers and releases them after publication.
    NmmPersistentWorkspace compatibility_workspace{};
    NmmPersistentWorkspace* persistent =
            static_cast<NmmPersistentWorkspace*>(state.persistent_workspace);
    const bool owns_compatibility_workspace = persistent == nullptr;
    if (!persistent) persistent = &compatibility_workspace;
    bool epoch_invalidated = false;
    if (persistent) {
        const bool epoch_changed = persistent->epoch_initialized &&
                (persistent->graph_epoch != state.graph_epoch ||
                 persistent->level_epoch != state.level_epoch ||
                 persistent->particle_order_epoch != state.particle_order_epoch ||
                 persistent->tree_epoch != state.tree_epoch);
        if (epoch_changed) {
            const int prior_build_count = persistent->build_count;
            const int prior_map_build_count = persistent->map_build_count;
            const int prior_map_launch_count = persistent->map_launch_count;
            release_workspace(*persistent);
            // Allocations are new after invalidation, but diagnostics remain
            // monotonic over the workspace owner lifetime.
            persistent->build_count = prior_build_count;
            persistent->map_build_count = prior_map_build_count;
            persistent->map_launch_count = prior_map_launch_count;
            // The caller's state views pointed into the invalidated
            // persistent arena.  Clear them before deciding whether a new
            // tree is needed; otherwise a stale cell_count would suppress
            // the rebuild and leave dangling device pointers in the ABI.
            state.cell_count = 0;
            state.contained_node_count = 0;
            state.I_count = 0;
            state.D1_count = 0;
            state.D2_count = 0;
            state.M_count = 0;
            state.cell_downleft_x = nullptr;
            state.cell_downleft_y = nullptr;
            state.cell_box_length = nullptr;
            state.cell_level = nullptr;
            state.cell_parent = nullptr;
            state.cell_child4 = nullptr;
            state.cell_particle_count = nullptr;
            state.cell_center_x = nullptr;
            state.cell_center_y = nullptr;
            state.contained_begin = nullptr;
            state.contained_nodes = nullptr;
            state.temp_contained_nodes = nullptr;
            state.temp_contained_scratch = nullptr;
            state.temp_contained_begin = nullptr;
            state.I_begin = nullptr;
            state.I_nodes = nullptr;
            state.D1_begin = nullptr;
            state.D1_nodes = nullptr;
            state.D2_begin = nullptr;
            state.D2_nodes = nullptr;
            state.M_begin = nullptr;
            state.M_nodes = nullptr;
            state.multipole_re = nullptr;
            state.multipole_im = nullptr;
            state.local_re = nullptr;
            state.local_im = nullptr;
            state.particle_to_leaf = nullptr;
            epoch_invalidated = true;
        }
        persistent->epoch_initialized = true;
        persistent->graph_epoch = state.graph_epoch;
        persistent->level_epoch = state.level_epoch;
        persistent->particle_order_epoch = state.particle_order_epoch;
        persistent->tree_epoch = state.tree_epoch;
    }
    std::vector<double*> owned_d;
    std::vector<int*> owned_i;
    auto adopt_d = [&](double* p) {
        if (p) {
            owned_d.push_back(p);
        }
        return p;
    };
    auto adopt_i = [&](int* p) {
        if (p) {
            owned_i.push_back(p);
        }
        return p;
    };
    auto upload_d = [&](const std::vector<double>& host) -> double* {
        if (host.empty()) {
            return nullptr;
        }
        double* d = nullptr;
        if (nmm_reference_cuda_malloc(&d, host.size() * sizeof(double)) != cudaSuccess) return nullptr;
        if (nmm_reference_cuda_memcpy(d, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            nmm_reference_cuda_free(d);
            return nullptr;
        }
        return adopt_d(d);
    };
    auto upload_i = [&](const std::vector<int>& host) -> int* {
        if (host.empty()) {
            return nullptr;
        }
        int* d = nullptr;
        if (nmm_reference_cuda_malloc(&d, host.size() * sizeof(int)) != cudaSuccess) return nullptr;
        if (nmm_reference_cuda_memcpy(d, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            nmm_reference_cuda_free(d);
            return nullptr;
        }
        return adopt_i(d);
    };
    auto rollback = [&]() {
        if (persistent) {
            release_workspace(*persistent);
        }
        if (owns_compatibility_workspace) {
            for (double* p : owned_d) if (p) nmm_reference_cuda_free(p);
            for (int* p : owned_i) if (p) nmm_reference_cuda_free(p);
        }
        state.cell_downleft_x = nullptr; state.cell_downleft_y = nullptr;
        state.cell_box_length = nullptr; state.cell_level = nullptr;
        state.cell_parent = nullptr; state.cell_child4 = nullptr;
        state.cell_particle_count = nullptr; state.cell_center_x = nullptr;
        state.cell_center_y = nullptr; state.contained_begin = nullptr;
        state.contained_nodes = nullptr; state.I_begin = nullptr; state.I_nodes = nullptr;
        state.temp_contained_nodes = nullptr; state.temp_contained_scratch = nullptr;
        state.temp_contained_begin = nullptr;
        state.D1_begin = nullptr; state.D1_nodes = nullptr; state.D2_begin = nullptr;
        state.D2_nodes = nullptr; state.M_begin = nullptr; state.M_nodes = nullptr;
        state.multipole_re = nullptr; state.multipole_im = nullptr;
        state.local_re = nullptr; state.local_im = nullptr;
        state.particle_to_leaf = nullptr;
        state.device_build_counters = nullptr;
        state.device_build_counter_capacity = 0;
        state.device_build_queue = nullptr;
        state.device_wsprl_queue = nullptr;
        state.cell_count = state.contained_node_count = state.I_count = 0;
        state.D1_count = state.D2_count = state.M_count = 0;
    };
    auto cached_d = [&](double*& ptr, std::size_t& cap, const std::vector<double>& host) -> double* {
        return (persistent && upload_cached(ptr, cap, host)) ? ptr : nullptr;
    };
    auto cached_i = [&](int*& ptr, std::size_t& cap, const std::vector<int>& host) -> int* {
        return (persistent && upload_cached(ptr, cap, host)) ? ptr : nullptr;
    };
    if (persistent && !epoch_invalidated && state.cell_count <= 0 && persistent->build_count > 0) {
        // A new FmmmNmmState can attach to the still-valid owner workspace.
        // This is the hierarchy-lifetime reuse path: no host tree rebuild and
        // no device allocation is needed to begin the next force call.
        state.cell_count = persistent->cell_count;
        state.contained_node_count = persistent->contained_node_count;
        state.I_count = persistent->I_count;
        state.D1_count = persistent->D1_count;
        state.D2_count = persistent->D2_count;
        state.M_count = persistent->M_count;
        state.precision = persistent->precision;
        state.expansion_stride = persistent->expansion_stride;
        state.cell_downleft_x = persistent->cell_downleft_x;
        state.cell_downleft_y = persistent->cell_downleft_y;
        state.cell_box_length = persistent->cell_box_length;
        state.cell_level = persistent->cell_level;
        state.cell_parent = persistent->cell_parent;
        state.cell_child4 = persistent->cell_child4;
        state.cell_particle_count = persistent->cell_particle_count;
        state.cell_center_x = persistent->cell_center_x;
        state.cell_center_y = persistent->cell_center_y;
        state.contained_begin = persistent->contained_begin;
        state.contained_nodes = persistent->contained_nodes;
        state.temp_contained_nodes = persistent->temp_contained_nodes;
        state.temp_contained_scratch = persistent->temp_contained_scratch;
        state.temp_contained_begin = persistent->temp_contained_begin;
        state.I_begin = persistent->I_begin;
        state.I_nodes = persistent->I_nodes;
        state.D1_begin = persistent->D1_begin;
        state.D1_nodes = persistent->D1_nodes;
        state.D2_begin = persistent->D2_begin;
        state.D2_nodes = persistent->D2_nodes;
        state.M_begin = persistent->M_begin;
        state.M_nodes = persistent->M_nodes;
        state.multipole_re = persistent->multipole_re;
        state.multipole_im = persistent->multipole_im;
        state.local_re = persistent->local_re;
        state.local_im = persistent->local_im;
    }
    const bool build_tree = epoch_invalidated || (state.cell_count <= 0) || (options.box_length > 0.0 &&
            state.cell_downleft_x == nullptr && state.cell_child4 == nullptr);

    if (build_tree) {
        if (options.box_length <= 0.0) {
            result.reason = "box_length required to build NMM tree";
            return result;
        }

        const int precision = options.precision > 0 ? options.precision : 4;
        const int stride = precision + 1;
        state.precision = precision;
        state.expansion_stride = stride;

        // Standalone/reference capacity is intentionally more generous than
        // the resident shape so degenerate inputs do not fail before the ABI
        // compatibility path can publish its tree.
        int max_cells = state.node_count * 64 + 1000;
        int max_contained = std::max(state.node_count * 128 + 1000,
                                     max_cells * 4);
        // The compatibility/reference ABI must accommodate the complete
        // ordered WSPRL relation set.  Resident capacity remains governed by
        // its session shape contract; this larger transient bound prevents a
        // standalone build from being rejected solely by list cardinality.
        int max_list = max_cells * 512;

        auto alloc_d = [&](double*& ptr, std::size_t& cap, std::size_t needed) {
            return ensure_device_capacity(ptr, cap, needed);
        };
        auto alloc_i = [&](int*& ptr, std::size_t& cap, std::size_t needed) {
            return ensure_device_capacity(ptr, cap, needed);
        };

        if (!alloc_d(persistent->cell_downleft_x, persistent->cap_cell_downleft_x, max_cells) ||
            !alloc_d(persistent->cell_downleft_y, persistent->cap_cell_downleft_y, max_cells) ||
            !alloc_d(persistent->cell_box_length, persistent->cap_cell_box_length, max_cells) ||
            !alloc_i(persistent->cell_level, persistent->cap_cell_level, max_cells) ||
            !alloc_i(persistent->cell_parent, persistent->cap_cell_parent, max_cells) ||
            !alloc_i(persistent->cell_child4, persistent->cap_cell_child4, max_cells * 4) ||
            !alloc_i(persistent->cell_particle_count, persistent->cap_cell_particle_count, max_cells) ||
            !alloc_d(persistent->cell_center_x, persistent->cap_cell_center_x, max_cells) ||
            !alloc_d(persistent->cell_center_y, persistent->cap_cell_center_y, max_cells) ||
            !alloc_i(persistent->contained_begin, persistent->cap_contained_begin, max_cells + 1) ||
            !alloc_i(persistent->contained_nodes, persistent->cap_contained_nodes, max_contained) ||
            !alloc_i(persistent->temp_contained_nodes, persistent->cap_temp_contained_nodes, max_contained) ||
            !alloc_i(persistent->temp_contained_scratch, persistent->cap_temp_contained_scratch, max_contained) ||
            !alloc_i(persistent->temp_contained_begin, persistent->cap_temp_contained_begin, max_cells + 1) ||
            !alloc_i(persistent->I_begin, persistent->cap_I_begin, max_cells + 1) ||
            !alloc_i(persistent->I_nodes, persistent->cap_I_nodes, max_list) ||
            !alloc_i(persistent->D1_begin, persistent->cap_D1_begin, max_cells + 1) ||
            !alloc_i(persistent->D1_nodes, persistent->cap_D1_nodes, max_list) ||
            !alloc_i(persistent->D2_begin, persistent->cap_D2_begin, max_cells + 1) ||
            !alloc_i(persistent->D2_nodes, persistent->cap_D2_nodes, max_list) ||
            !alloc_i(persistent->M_begin, persistent->cap_M_begin, max_cells + 1) ||
            !alloc_i(persistent->M_nodes, persistent->cap_M_nodes, max_list) ||
            !alloc_d(persistent->multipole_re, persistent->cap_multipole_re, max_cells * stride) ||
            !alloc_d(persistent->multipole_im, persistent->cap_multipole_im, max_cells * stride) ||
            !alloc_d(persistent->local_re, persistent->cap_local_re, max_cells * stride) ||
            !alloc_d(persistent->local_im, persistent->cap_local_im, max_cells * stride)) {
            rollback();
            result.reason = "persistent NMM workspace allocation failed";
            return result;
        }

        state.cell_downleft_x = persistent->cell_downleft_x;
        state.cell_downleft_y = persistent->cell_downleft_y;
        state.cell_box_length = persistent->cell_box_length;
        state.cell_level = persistent->cell_level;
        state.cell_parent = persistent->cell_parent;
        state.cell_child4 = persistent->cell_child4;
        state.cell_particle_count = persistent->cell_particle_count;
        state.cell_center_x = persistent->cell_center_x;
        state.cell_center_y = persistent->cell_center_y;
        state.contained_begin = persistent->contained_begin;
        state.contained_nodes = persistent->contained_nodes;
        state.temp_contained_nodes = persistent->temp_contained_nodes;
        state.temp_contained_scratch = persistent->temp_contained_scratch;
        state.temp_contained_begin = persistent->temp_contained_begin;
        state.I_begin = persistent->I_begin;
        state.I_nodes = persistent->I_nodes;
        state.D1_begin = persistent->D1_begin;
        state.D1_nodes = persistent->D1_nodes;
        state.D2_begin = persistent->D2_begin;
        state.D2_nodes = persistent->D2_nodes;
        state.M_begin = persistent->M_begin;
        state.M_nodes = persistent->M_nodes;
        state.multipole_re = persistent->multipole_re;
        state.multipole_im = persistent->multipole_im;
        state.local_re = persistent->local_re;
        state.local_im = persistent->local_im;
        state.max_cells = max_cells;
        state.max_contained = max_contained;
        state.max_list = max_list;
        state.queue_capacity = max_list;
        state.frontier_queue_capacity = max_cells;
        state.traversal_queue_capacity = max_list;
        state.max_I = max_list;
        state.max_D1 = max_list;
        state.max_D2 = max_list;
        state.max_M = max_list;

        int* d_counters = nullptr;
        const std::size_t counter_count =
                static_cast<std::size_t>(kNmmBuildStorageCount);
        if (nmm_reference_cuda_malloc(&d_counters, counter_count * sizeof(int)) != cudaSuccess ||
                cudaMemset(d_counters, 0, counter_count * sizeof(int)) != cudaSuccess) {
            if (d_counters) nmm_reference_cuda_free(d_counters);
            rollback();
            result.reason = "persistent NMM counter allocation failed";
            return result;
        }
        int* q_cells = nullptr;
        int* traversal_q_cells = nullptr;
        const int queue_capacity = max_list;
        if (nmm_reference_cuda_malloc(&q_cells,
                                      static_cast<std::size_t>(max_cells) * sizeof(int)) != cudaSuccess ||
                nmm_reference_cuda_malloc(&traversal_q_cells,
                                          static_cast<std::size_t>(queue_capacity) * sizeof(int)) != cudaSuccess) {
            nmm_reference_cuda_free(d_counters);
            if (q_cells) nmm_reference_cuda_free(q_cells);
            if (traversal_q_cells) nmm_reference_cuda_free(traversal_q_cells);
            rollback();
            result.reason = "persistent NMM queue allocation failed";
            return result;
        }

        double root_min_x = options.down_left_x;
        double root_min_y = options.down_left_y;
        double root_max_x = options.down_left_x + options.box_length;
        double root_max_y = options.down_left_y + options.box_length;

        int p_leaf = options.particles_in_leaf > 0 ? options.particles_in_leaf : 1;

        state.device_build_queue = q_cells;
        state.device_wsprl_queue = traversal_q_cells;
        state.queue_capacity = queue_capacity;
        state.frontier_queue_capacity = max_cells;
        state.traversal_queue_capacity = queue_capacity;
        state.device_build_counters = d_counters;
        state.device_build_counter_capacity = static_cast<int>(counter_count);
        state.status = &d_counters[6];
        FmmmNmmOptions topology_options = options;
        topology_options.particles_in_leaf = p_leaf;
        const FmmmStageResult topology = enqueue_parallel_tree_topology(state, topology_options, nullptr);
        if (topology.status != FmmmPipelineStatus::ok) {
            nmm_reference_cuda_free(d_counters);
            nmm_reference_cuda_free(q_cells);
            nmm_reference_cuda_free(traversal_q_cells);
            rollback();
            result.reason = topology.reason;
            return result;
        }

        const int phase_threads = 256;
        const int phase_blocks = std::max(1, (max_cells + phase_threads - 1) / phase_threads);
        nmm_set_centers_kernel<<<phase_blocks, phase_threads>>>(
            state, d_counters, max_cells, options.seed);
        int device = 0;
        cudaDeviceProp device_properties{};
        int cooperative_blocks_per_sm = 0;
#if defined(LCN_DISABLE_G2_PERSISTENT)
        const bool cooperative_available = false;
#else
        const bool cooperative_available =
                cudaGetDevice(&device) == cudaSuccess &&
                cudaGetDeviceProperties(&device_properties, device) == cudaSuccess &&
                device_properties.cooperativeLaunch &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &cooperative_blocks_per_sm,
                        nmm_multipole_persistent_kernel,
                        phase_threads, 0) == cudaSuccess &&
                cooperative_blocks_per_sm > 0;
#endif
        cudaError_t multipole_err = cudaSuccess;
        if (cooperative_available) {
            const int cooperative_blocks = std::max(1, std::min(
                    phase_blocks,
                    cooperative_blocks_per_sm * device_properties.multiProcessorCount));
            void* kernel_args[] = {&state, &d_counters, &max_cells};
            multipole_err = cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(nmm_multipole_persistent_kernel),
                    dim3(cooperative_blocks, 1, 1),
                    dim3(phase_threads, 1, 1),
                    kernel_args, 0, nullptr);
        } else {
            // Compatibility fallback for devices without cooperative launch.
            const int multipole_levels = nmm_host_level_cap(state.node_count);
            for (int level = multipole_levels - 1; level >= 0; --level) {
                nmm_multipole_level_kernel<<<phase_blocks, phase_threads>>>(
                    state, d_counters, max_cells, level);
                multipole_err = cudaGetLastError();
                if (multipole_err != cudaSuccess) break;
            }
        }
        if (multipole_err != cudaSuccess) {
            nmm_reference_cuda_free(d_counters);
            nmm_reference_cuda_free(q_cells);
            nmm_reference_cuda_free(traversal_q_cells);
            rollback();
            result.reason = "standalone multipole launch failed";
            return result;
        }

        // The standalone ABI uses the same deterministic resident WSPRL
        // count/scan/scatter implementation before the bounded counter readback.
        for (int* begin : {state.I_begin, state.D1_begin, state.D2_begin, state.M_begin}) {
            if (cudaMemset(begin, 0,
                           static_cast<std::size_t>(max_cells + 1) * sizeof(int)) != cudaSuccess) {
                nmm_reference_cuda_free(d_counters);
                nmm_reference_cuda_free(q_cells);
                nmm_reference_cuda_free(traversal_q_cells);
                rollback();
                result.reason = "standalone WSPRL begin reset failed";
                return result;
            }
        }
        FmmmStageResult wsprl = enqueue_parallel_wsprl(state, nullptr);
        if (wsprl.status != FmmmPipelineStatus::ok) {
            nmm_reference_cuda_free(d_counters);
            nmm_reference_cuda_free(q_cells);
            nmm_reference_cuda_free(traversal_q_cells);
            rollback();
            result.reason = wsprl.reason;
            return result;
        }

        int h_counters[8];
        int h_build[kNmmBuildStorageCount] = {};
        nmm_reference_cuda_memcpy(h_build, d_counters,
                                  sizeof(h_build), cudaMemcpyDeviceToHost);
        nmm_reference_cuda_free(d_counters);
        nmm_reference_cuda_free(q_cells);
        nmm_reference_cuda_free(traversal_q_cells);
        state.device_build_counters = nullptr;
        state.device_build_counter_capacity = 0;
        state.device_wsprl_queue = nullptr;
        state.status = nullptr;

        for (int i = 0; i < kNmmBuildCounterCount; ++i) h_counters[i] = h_build[i];

        state.cell_count = h_counters[0];
        state.contained_node_count = h_counters[1];
        state.I_count = h_counters[2];
        state.D1_count = h_counters[3];
        state.D2_count = h_counters[4];
        state.M_count = h_counters[5];
        if (h_counters[6] != 0) {
            const int code = h_counters[6] - 1;
            switch (code) {
                case kNmmOverflowFrontierQueue:
                    result.reason = "device tree build overflow: frontier queue"; break;
                case kNmmOverflowTraversalQueue:
                    result.reason = "device tree build overflow: WSPRL traversal queue"; break;
                case kNmmOverflowIList:
                    result.reason = "device tree build overflow: I relation list"; break;
                case kNmmOverflowD1List:
                    result.reason = "device tree build overflow: D1 relation list"; break;
                case kNmmOverflowD2List:
                    result.reason = "device tree build overflow: D2 relation list"; break;
                case kNmmOverflowMList:
                    result.reason = "device tree build overflow: M relation list"; break;
                case kNmmOverflowContained:
                    result.reason = "device tree build overflow: contained storage"; break;
                case kNmmOverflowCell:
                    result.reason = "device tree build overflow: cell storage"; break;
                default:
                    result.reason = "device tree build overflow: unknown category"; break;
            }
            return result;
        }

        persistent->cell_count = state.cell_count;
        persistent->contained_node_count = state.contained_node_count;
        persistent->I_count = state.I_count;
        persistent->D1_count = state.D1_count;
        persistent->D2_count = state.D2_count;
        persistent->M_count = state.M_count;
        persistent->precision = state.precision;
        persistent->expansion_stride = state.expansion_stride;
        ++persistent->build_count;
        state.tree_rebuilt = 1;
    } else {
        // Prebuilt relation path: synthesize missing geometry so ABI arrays are used.
        if (!state.cell_child4) {
            std::vector<int> child4(state.cell_count * 4, -1);
            state.cell_child4 = upload_i(child4);
        }
        if (!state.cell_parent) {
            std::vector<int> parent(state.cell_count, -1);
            state.cell_parent = upload_i(parent);
        }
        if (!state.cell_level) {
            std::vector<int> level(state.cell_count, 0);
            state.cell_level = upload_i(level);
        }
        if (!state.cell_downleft_x) {
            std::vector<double> zeros(state.cell_count, 0.0);
            state.cell_downleft_x = upload_d(zeros);
        }
        if (!state.cell_downleft_y) {
            std::vector<double> zeros(state.cell_count, 0.0);
            state.cell_downleft_y = upload_d(zeros);
        }
        if (!state.cell_box_length) {
            std::vector<double> zeros(state.cell_count, 0.0);
            state.cell_box_length = upload_d(zeros);
        }
        if (!state.cell_particle_count) {
            std::vector<int> zeros(state.cell_count, 0);
            state.cell_particle_count = upload_i(zeros);
        }
    }

    // Validate the complete tree ABI before any particle-to-leaf kernel is
    // launched.  An incomplete prebuilt tree is an invalid input contract,
    // not a CUDA backend failure.
    const int expansion_elements = state.cell_count * state.expansion_stride;
    if (state.cell_count <= 0 || !state.cell_center_x || !state.cell_center_y ||
            !state.contained_begin || (state.contained_node_count > 0 && !state.contained_nodes) ||
            !state.I_begin || (state.I_count > 0 && !state.I_nodes) ||
            !state.D1_begin || (state.D1_count > 0 && !state.D1_nodes) ||
            !state.D2_begin || (state.D2_count > 0 && !state.D2_nodes) ||
            !state.M_begin || (state.M_count > 0 && !state.M_nodes) ||
            expansion_elements <= 0 || !state.multipole_re || !state.multipole_im ||
            !state.local_re || !state.local_im || !state.cell_child4 ||
            !state.cell_parent || !state.cell_level ||
            !state.cell_particle_count) {
        rollback();
        result.status = FmmmPipelineStatus::invalid_input;
        result.reason = "incomplete NMM tree ABI arrays";
        return result;
    }

    if (persistent) {
        if (!ensure_device_capacity(persistent->particle_to_leaf,
                                    persistent->cap_particle_to_leaf,
                                    static_cast<std::size_t>(state.node_count))) {
            rollback();
            result.reason = "particle_to_leaf allocation failed";
            return result;
        }
        state.particle_to_leaf = persistent->particle_to_leaf;
        const bool map_valid = persistent->map_build_count > 0 &&
                persistent->map_epoch == state.map_epoch && !epoch_invalidated &&
                !state.tree_rebuilt;
        if (!map_valid) {
            cudaMemset(state.particle_to_leaf, 0xff,
                       static_cast<std::size_t>(state.node_count) * sizeof(int));
            const int map_threads = 128;
            const int map_blocks = (state.cell_count + map_threads - 1) / map_threads;
            kernel_nmm_build_particle_to_leaf<<<map_blocks, map_threads>>>(
                state.cell_count, state.cell_child4, state.contained_begin,
                state.contained_nodes, state.particle_to_leaf);
            ++persistent->map_launch_count;
            if (cudaGetLastError() != cudaSuccess || nmm_reference_cuda_sync() != cudaSuccess) {
                rollback();
                result.status = FmmmPipelineStatus::backend_error;
                result.reason = "particle_to_leaf build failed";
                return result;
            }
            ++persistent->map_build_count;
            state.particle_to_leaf_rebuilt = 1;
            state.particle_to_leaf_launches = 1;
            persistent->map_epoch = state.map_epoch;
        }
    }

#if defined(LCN_G3_LEGACY_FORCE_TILE)
    constexpr int threads = 256;
    const int blocks = (state.node_count + threads - 1) / threads;

    kernel_nmm_force<<<blocks, threads>>>(
        state.position_x, state.position_y,
        state.force_x, state.force_y,
        state.node_count,
        state.cell_downleft_x, state.cell_downleft_y,
        state.cell_box_length, state.cell_level, state.cell_parent, state.cell_child4,
        state.cell_center_x, state.cell_center_y,
        state.I_begin, state.I_nodes,
        state.D1_begin, state.D1_nodes,
        state.D2_begin, state.D2_nodes,
        state.M_begin, state.M_nodes,
        state.contained_begin, state.contained_nodes,
        state.multipole_re, state.multipole_im,
        state.local_re, state.local_im, state.particle_to_leaf,
        state.precision,
        state.expansion_stride,
        state.cell_particle_count,
        state.cell_count,
        state.cell_count > 0 ? nullptr : state.device_build_counters,
        state.status);
#else
    constexpr int threads = 128;
    const int blocks = std::max(1, std::min(state.max_cells,
                                             std::max(1, state.node_count)));
    kernel_nmm_force_interaction_tile<<<blocks, threads,
                                        2 * threads * sizeof(double)>>>(
        state.position_x, state.position_y, state.force_x, state.force_y,
        state.node_count, state.cell_child4, state.cell_center_x,
        state.cell_center_y, state.D1_begin, state.D1_nodes,
        state.D2_begin, state.D2_nodes, state.contained_begin,
        state.contained_nodes, state.M_begin, state.M_nodes,
        state.multipole_re, state.multipole_im, state.local_re,
        state.local_im, state.particle_to_leaf, state.precision,
        state.expansion_stride,
        state.cell_count > 0 ? state.cell_count : state.max_cells,
        state.device_build_counters,
        state.status);
#endif

    cudaError_t err = cudaGetLastError();
    nmm_reference_cuda_sync();
    (void)owned_d;
    (void)owned_i;

    if (err != cudaSuccess) {
        result.status = FmmmPipelineStatus::backend_error;
        result.reason = cudaGetErrorString(err);
        return result;
    }

    result.status = FmmmPipelineStatus::ok;
    result.completed_stage = FmmmPipelineStage::normal_force;
    result.reason = "ok";
    return result;
}

void* fmmm_nmm_workspace_create() {
    return new (std::nothrow) NmmPersistentWorkspace();
}

void fmmm_nmm_workspace_release(void* opaque) noexcept {
    if (!opaque) return;
    auto* workspace = static_cast<NmmPersistentWorkspace*>(opaque);
    release_workspace(*workspace);
    delete workspace;
}

int fmmm_nmm_workspace_build_count(const void* opaque) noexcept {
    if (!opaque) return 0;
    return static_cast<const NmmPersistentWorkspace*>(opaque)->build_count;
}

int fmmm_nmm_workspace_map_build_count(const void* opaque) noexcept {
    if (!opaque) return 0;
    return static_cast<const NmmPersistentWorkspace*>(opaque)->map_build_count;
}

int fmmm_nmm_workspace_map_launch_count(const void* opaque) noexcept {
    if (!opaque) return 0;
    return static_cast<const NmmPersistentWorkspace*>(opaque)->map_launch_count;
}

} // namespace lcn::release
