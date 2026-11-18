// fmmm_hierarchy.cu  --  G4 recursive multilevel coordinator
//
// Implements fmmm_build_hierarchy (fine-to-coarse) and fmmm_execute_hierarchy
// (coarsest-to-finest) as declared in fmmm_hierarchy_contract.h.
//
// Ownership rules:
//   * Plan/state SoA for each level are allocated on device by build; callers
//     must release them via fmmm_release_hierarchy after consuming results.
//   * Execute frees a parent level's plan after the child has copied
//     high_position and completed Advanced uncoarsen (release-after-consume).
//     Parent state positions are retained until fmmm_release_hierarchy so
//     diagnostics can still download coarsest/per-level force endpoints.
//   * No host graph/vector reconstruction happens between levels.
//   * Build termination mirrors OGDF create_multilevel_representations:
//     loop while (nodes > minGraphSize) && edge_linearity_ok, where
//     linearity allows up to 5 "bad" levels before stopping.
//   * Iteration count mirrors OGDF get_max_mult_iter with LinearlyDecreasing:
//       iter = fixedIter + (act_level/max_level)*(maxIterFactor-1)*fixedIter

#include "fmmm_hierarchy_contract.h"
#include "fmmm_workspace.h"
#include "fmmm_level_engine.h"
#include "fmmm_advanced_uncoarsen.h"
#include "fmmm_nmm_packing.h"
#include "../oracle/device_memory.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lcn::release {
namespace {

// --------------------------------------------------------------------------
// Device-resident hierarchy builder
//
// This path is deliberately independent of the legacy OGDF multilevel
// coordinator below.  It materializes a breadth-first quadtree from a single
// device launch sequence.  No host code reads a count or geometry value while
// the tree is being built; the only host decision is the requested capacity
// bounded depth, which is derived from immutable input/configuration.

__device__ __forceinline__ int hierarchy_level_for_index(int index,
                                                          int depth) {
    int level = 0;
    int first = 0;
    int width = 1;
    while (level < depth && index >= first + width) {
        first += width;
        width *= 4;
        ++level;
    }
    return level;
}

__device__ __forceinline__ int hierarchy_level_first(int level) {
    int first = 0;
    int width = 1;
    for (int i = 0; i < level; ++i) {
        first += width;
        width *= 4;
    }
    return first;
}

__device__ __forceinline__ int hierarchy_leaf_for_position(
    double px, double py, double down_left_x, double down_left_y,
    double box_length, int depth, int cell_count) {
    if (!isfinite(px) || !isfinite(py) || !isfinite(down_left_x) ||
        !isfinite(down_left_y) || !isfinite(box_length) || box_length <= 0.0) {
        return -1;
    }
    double ux = (px - down_left_x) / box_length;
    double uy = (py - down_left_y) / box_length;
    ux = fmin(1.0, fmax(0.0, ux));
    uy = fmin(1.0, fmax(0.0, uy));

    const int side = 1 << depth;
    int ix = static_cast<int>(floor(ux * static_cast<double>(side)));
    int iy = static_cast<int>(floor(uy * static_cast<double>(side)));
    if (ix >= side) ix = side - 1;
    if (iy >= side) iy = side - 1;

    int cell = 0;
    for (int level = 0; level < depth; ++level) {
        const int shift = depth - level - 1;
        const int quadrant = ((ix >> shift) & 1) | (((iy >> shift) & 1) << 1);
        cell = cell * 4 + 1 + quadrant;
    }
    return cell < cell_count ? cell : -1;
}

__global__ void device_hierarchy_init_kernel(
    int cell_count, int depth, int node_count,
    double down_left_x, double down_left_y, double box_length,
    std::uint64_t position_epoch, std::uint64_t box_epoch,
    std::uint64_t tree_epoch,
    double* cell_down_left_x, double* cell_down_left_y,
    double* cell_box_length, double* cell_center_x, double* cell_center_y,
    int* cell_parent, int* cell_children4, int* cell_level,
    int* cell_compact_id, int* cell_particle_count,
    int* cell_count_device, int* root_count_device,
    std::uint64_t* tree_epoch_device,
    std::uint64_t* position_epoch_device,
    std::uint64_t* box_epoch_device, int* status_device) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cell_count) {
        const int level = hierarchy_level_for_index(i, depth);
        const int first = hierarchy_level_first(level);
        int ordinal = i - first;
        int ix = 0;
        int iy = 0;
        int tmp = ordinal;
        double scale = 1.0;
        for (int l = 0; l < level; ++l) {
            const int quadrant = tmp & 3;
            ix |= (quadrant & 1) << l;
            iy |= ((quadrant >> 1) & 1) << l;
            tmp >>= 2;
            scale *= 0.5;
        }
        cell_down_left_x[i] = down_left_x + box_length * scale * ix;
        cell_down_left_y[i] = down_left_y + box_length * scale * iy;
        cell_box_length[i] = box_length * scale;
        cell_center_x[i] = cell_down_left_x[i] + cell_box_length[i] * 0.5;
        cell_center_y[i] = cell_down_left_y[i] + cell_box_length[i] * 0.5;
        cell_parent[i] = level == 0 ? -1 : (i - 1) / 4;
        cell_level[i] = level;
        cell_compact_id[i] = i;
        cell_particle_count[i] = 0;
        for (int child = 0; child < 4; ++child) {
            const int child_index = i * 4 + 1 + child;
            cell_children4[i * 4 + child] =
                (level < depth && child_index < cell_count) ? child_index : -1;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *cell_count_device = node_count > 0 ? cell_count : 0;
        *root_count_device = node_count > 0 ? 1 : 0;
        *tree_epoch_device = tree_epoch;
        *position_epoch_device = position_epoch;
        *box_epoch_device = box_epoch;
        *status_device = (node_count >= 0 && box_length > 0.0) ? 0 : 1;
    }
}

__global__ void device_hierarchy_assign_particles_kernel(
    int node_count, int depth, int cell_count,
    const double* position_x, const double* position_y,
    double down_left_x, double down_left_y, double box_length,
    int* particle_to_leaf, int* cell_particle_count, int* status_device) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;
    const int leaf = hierarchy_leaf_for_position(
        position_x[i], position_y[i], down_left_x, down_left_y,
        box_length, depth, cell_count);
    particle_to_leaf[i] = leaf;
    if (leaf < 0) {
        atomicExch(status_device, 2);
        return;
    }
    atomicAdd(cell_particle_count + leaf, 1);
}

// ---- CUDA helpers ----------------------------------------------------------

inline void chk(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}

template <typename T>
T* dev_alloc(int n) {
    if (n <= 0) return nullptr;
    T* p = nullptr;
    chk(cudaMalloc(&p, static_cast<std::size_t>(n) * sizeof(T)), "cudaMalloc");
    chk(cudaMemset(p, 0, static_cast<std::size_t>(n) * sizeof(T)), "cudaMemset");
    return p;
}

template <typename T>
void dev_free(T*& p) { if (p) { cudaFree(p); p = nullptr; } }

template <typename T>
void dev_upload(const T* src, T* dst, int n) {
    if (n > 0 && src && dst)
        chk(cudaMemcpy(dst, src, static_cast<std::size_t>(n) * sizeof(T),
                       cudaMemcpyHostToDevice), "H2D");
}

template <typename T>
T dev_dl1(const T* p) {
    T v{};
    chk(cudaMemcpy(&v, p, sizeof(T), cudaMemcpyDeviceToHost), "D2H scalar");
    return v;
}

template <typename T>
void dev_d2d(const T* src, T* dst, int n) {
    if (n > 0 && src && dst)
        chk(cudaMemcpy(dst, src, static_cast<std::size_t>(n) * sizeof(T),
                       cudaMemcpyDeviceToDevice), "D2D");
}

// ---- Workspace factory -----------------------------------------------------

FmmmWorkspace make_ws(int nodes, int edges) {
    const int e = std::max(edges, 1);
    FmmmWorkspaceShape sh{};
    sh.node_count     = static_cast<std::size_t>(nodes);
    sh.edge_count     = static_cast<std::size_t>(e);
    sh.level_count    = 1;
    sh.cell_capacity  = static_cast<std::size_t>(nodes);
    sh.max_level_nodes = static_cast<std::size_t>(nodes);
    sh.max_level_edges = static_cast<std::size_t>(e);
    // Force scratch (rep_y) needs >= node_count; keep a floor for aux lists.
    sh.max_aux_entries = static_cast<std::size_t>(std::max(
        std::max(512, nodes), oracle::kFmmmAdvancedUncoarsenMaxAuxEntries));
    sh.low_adj_count   = static_cast<std::size_t>(e) * 2;
    sh.high_adj_count  = static_cast<std::size_t>(e) * 2;
    sh.lambda_count    = static_cast<std::size_t>(e) * 2;
    sh.neighbour_count = static_cast<std::size_t>(e) * 2;
    sh.moon_count      = static_cast<std::size_t>(e) * 2;
    FmmmWorkspace ws;
    if (FmmmWorkspace::create(sh, ws) != FmmmPipelineStatus::ok)
        throw std::runtime_error("fmmm_hierarchy: workspace allocation failed");
    return ws;
}

// ---- Plan / state alloc/free -----------------------------------------------

void alloc_plan(oracle::FmmmAdvancedUncoarsenPlanView& p,
                int ln, int le, int hn, int he,
                int lam, int nbr, int moon_sz) {
    p.lower_node_count        = ln;
    p.lower_edge_ordinal_span = le;
    p.high_node_count         = hn;
    p.high_edge_ordinal_span  = he;

    // Lower-level node SoA
    const_cast<int*&>(p.lower_type)                        = dev_alloc<int>(ln);
    const_cast<int*&>(p.lower_dedicated_sun)               = dev_alloc<int>(ln);
    const_cast<double*&>(p.lower_dedicated_sun_distance)   = dev_alloc<double>(ln);
    // High node map + positions
    const_cast<int*&>(p.high_lower_node)                   = dev_alloc<int>(hn);
    const_cast<double*&>(p.high_position_x)                = dev_alloc<double>(hn);
    const_cast<double*&>(p.high_position_y)                = dev_alloc<double>(hn);
    // Lower edge arrays
    const_cast<int*&>(p.lower_edge_source)                 = dev_alloc<int>(le);
    const_cast<int*&>(p.lower_edge_target)                 = dev_alloc<int>(le);
    const_cast<double*&>(p.lower_edge_length)              = dev_alloc<double>(le);
    const_cast<std::uint8_t*&>(p.lower_edge_is_moon)       = dev_alloc<std::uint8_t>(le);
    const_cast<int*&>(p.lower_adjacency_begin)             = dev_alloc<int>(ln + 1);
    const_cast<int*&>(p.lower_adjacency_edge)              = dev_alloc<int>(le * 2);
    // High edge arrays
    const_cast<int*&>(p.high_edge_source)                  = dev_alloc<int>(he);
    const_cast<int*&>(p.high_edge_target)                  = dev_alloc<int>(he);
    const_cast<std::uint8_t*&>(p.high_edge_is_extra)       = dev_alloc<std::uint8_t>(he);
    const_cast<int*&>(p.high_adjacency_begin)              = dev_alloc<int>(hn + 1);
    const_cast<int*&>(p.high_adjacency_edge)               = dev_alloc<int>(he * 2);
    // Auxiliary flat lists
    const_cast<int*&>(p.lambda_begin)                      = dev_alloc<int>(ln + 1);
    const_cast<double*&>(p.lambda_values)                  = dev_alloc<double>(lam);
    const_cast<int*&>(p.neighbour_sun_begin)               = dev_alloc<int>(ln + 1);
    const_cast<int*&>(p.neighbour_sun_nodes)               = dev_alloc<int>(nbr);
    const_cast<int*&>(p.moon_begin)                        = dev_alloc<int>(ln + 1);
    const_cast<int*&>(p.moon_nodes)                        = dev_alloc<int>(moon_sz);
}

void free_plan(oracle::FmmmAdvancedUncoarsenPlanView& p) {
    dev_free(const_cast<int*&>(p.lower_type));
    dev_free(const_cast<int*&>(p.lower_dedicated_sun));
    dev_free(const_cast<double*&>(p.lower_dedicated_sun_distance));
    dev_free(const_cast<int*&>(p.high_lower_node));
    dev_free(const_cast<double*&>(p.high_position_x));
    dev_free(const_cast<double*&>(p.high_position_y));
    dev_free(const_cast<int*&>(p.lower_edge_source));
    dev_free(const_cast<int*&>(p.lower_edge_target));
    dev_free(const_cast<double*&>(p.lower_edge_length));
    dev_free(const_cast<std::uint8_t*&>(p.lower_edge_is_moon));
    dev_free(const_cast<int*&>(p.lower_adjacency_begin));
    dev_free(const_cast<int*&>(p.lower_adjacency_edge));
    dev_free(const_cast<int*&>(p.high_edge_source));
    dev_free(const_cast<int*&>(p.high_edge_target));
    dev_free(const_cast<std::uint8_t*&>(p.high_edge_is_extra));
    dev_free(const_cast<int*&>(p.high_adjacency_begin));
    dev_free(const_cast<int*&>(p.high_adjacency_edge));
    dev_free(const_cast<int*&>(p.lambda_begin));
    dev_free(const_cast<double*&>(p.lambda_values));
    dev_free(const_cast<int*&>(p.neighbour_sun_begin));
    dev_free(const_cast<int*&>(p.neighbour_sun_nodes));
    dev_free(const_cast<int*&>(p.moon_begin));
    dev_free(const_cast<int*&>(p.moon_nodes));
}

void alloc_state(oracle::FmmmAdvancedUncoarsenStateView& s, int n) {
    s.position_x  = dev_alloc<double>(n);
    s.position_y  = dev_alloc<double>(n);
    s.placed      = dev_alloc<std::uint8_t>(n);
    s.angle_1     = dev_alloc<double>(n);
    s.angle_2     = dev_alloc<double>(n);
    s.box_length  = dev_alloc<double>(1);
    s.down_left_x = dev_alloc<double>(1);
    s.down_left_y = dev_alloc<double>(1);
    s.status      = dev_alloc<int>(1);
    s.consumed_draws = dev_alloc<int>(1);
}

void free_state(oracle::FmmmAdvancedUncoarsenStateView& s) {
    dev_free(s.position_x); dev_free(s.position_y); dev_free(s.placed);
    dev_free(s.angle_1);    dev_free(s.angle_2);
    dev_free(s.box_length); dev_free(s.down_left_x); dev_free(s.down_left_y);
    dev_free(s.status);     dev_free(s.consumed_draws);
}

// ---- OGDF policy helpers ---------------------------------------------------

// edgenumbersum_of_all_levels_is_linear from Multilevel.cpp
bool edge_linearity_ok(int prev_e, int cur_e, int& bad_ctr) {
    if (prev_e <= 0) return true;
    if (cur_e <= static_cast<int>(0.8 * prev_e)) return true;
    if (bad_ctr < 5) { ++bad_ctr; return true; }
    return false;
}

// get_max_mult_iter (LinearlyDecreasing branch).
// fixedIter = profile.normal_iterations, maxIterFactor = profile.max_iter_factor.
int calc_iterations(int act_level, int max_level,
                    int fixed_iter, int max_iter_factor) {
    if (max_level == 0)
        return max_iter_factor * fixed_iter;
    return fixed_iter
        + static_cast<int>(
            (static_cast<double>(act_level) / static_cast<double>(max_level))
            * static_cast<double>(max_iter_factor - 1)
            * static_cast<double>(fixed_iter));
}

// FMMMLayout::set_average_ideal_edgelength: mean of Graph::edges() ideal lengths.
double avg_length(const double* d_len, int edge_count) {
    if (edge_count <= 0) return 50.0;
    std::vector<double> h(static_cast<std::size_t>(edge_count));
    chk(cudaMemcpy(h.data(), d_len,
                   static_cast<std::size_t>(edge_count) * sizeof(double),
                   cudaMemcpyDeviceToHost), "avg_len D2H");
    double s = 0.0;
    for (double v : h) s += v;
    return s / static_cast<double>(edge_count);
}

// Run force loop on a given level. Uses G5 NMM repulsion when n >= 175.
// Returns false and fills *out_reason on failure.
bool run_force_loop(const oracle::FmmmAdvancedUncoarsenPlanView& plan,
                    oracle::FmmmAdvancedUncoarsenStateView& state,
                    int act_level, int max_level,
                    int fixed_iter, int max_iter_factor,
                    int force_phase = 0, int explicit_iters = -1,
                    std::uint64_t seed = 100, const char** out_reason = nullptr) {
    const int n = plan.lower_node_count;
    const int e = plan.lower_edge_ordinal_span;
    if (n <= 1) return true;

    const double avg = avg_length(plan.lower_edge_length, e);
    if (!(avg > 0.0)) return true;
    const double box = dev_dl1(state.box_length);
    const double dlx = dev_dl1(state.down_left_x);
    const double dly = dev_dl1(state.down_left_y);
    int iters = explicit_iters;
    if (iters < 0) {
        iters = std::max(1, calc_iterations(act_level, max_level, fixed_iter, max_iter_factor));
        if (n <= 500) iters = std::max(100, iters);
    }

    FmmmWorkspace fw = make_ws(n, e);
    int*    d_fs   = dev_alloc<int>(1);
    double* d_ob   = dev_alloc<double>(1);
    double* d_ox   = dev_alloc<double>(1);
    double* d_oy   = dev_alloc<double>(1);

    constexpr int kNmmMinNodes = 175;
    if (n >= kNmmMinNodes) {
        FmmmWorkspaceView wv = fw.view();
        if (wv.force_x == nullptr || wv.force_y == nullptr || wv.mass == nullptr ||
            wv.center_x == nullptr || wv.center_y == nullptr || wv.scratch_double == nullptr) {
            dev_free(d_fs); dev_free(d_ob); dev_free(d_ox); dev_free(d_oy);
            if (out_reason) *out_reason = "nmm force loop: null workspace scratch";
            return false;
        }
        FmmmStageResult nmm_loop = fmmm_run_nmm_force_loop(
            n, e, plan.lower_edge_source, plan.lower_edge_target, plan.lower_edge_length,
            state.position_x, state.position_y, box, dlx, dly, avg, iters, d_fs, d_ob, d_ox, d_oy,
            force_phase, wv.force_x, wv.force_y, wv.mass, wv.center_x, wv.center_y,
            wv.scratch_double, seed);
        if (nmm_loop.status != FmmmPipelineStatus::ok) {
            dev_free(d_fs); dev_free(d_ob); dev_free(d_ox); dev_free(d_oy);
            if (out_reason) {
                *out_reason = nmm_loop.reason ? nmm_loop.reason : "nmm force loop failed";
            }
            return false;
        }
    } else {
        launch_exact_force_loop_kernel(
            n, e,
            plan.lower_edge_source, plan.lower_edge_target, plan.lower_edge_length,
            state.position_x, state.position_y,
            box, dlx, dly, avg, iters,
            d_fs, d_ob, d_ox, d_oy,
            force_phase, fw.view());
        chk(cudaGetLastError(),      "force loop kernel launch");
        chk(cudaDeviceSynchronize(), "force loop kernel sync");
    }

    if (dev_dl1(d_fs) == 0) {
        dev_d2d(d_ob, state.box_length,  1);
        dev_d2d(d_ox, state.down_left_x, 1);
        dev_d2d(d_oy, state.down_left_y, 1);
    }
    dev_free(d_fs); dev_free(d_ob); dev_free(d_ox); dev_free(d_oy);
    return true;
}

void run_resize(const oracle::FmmmAdvancedUncoarsenPlanView& plan,
                oracle::FmmmAdvancedUncoarsenStateView& state,
                int refresh_box) {
    const int n = plan.lower_node_count;
    const int e = plan.lower_edge_ordinal_span;
    if (n <= 1) return;

    int*    d_fs   = dev_alloc<int>(1);
    double* d_ob   = dev_alloc<double>(1);
    double* d_ox   = dev_alloc<double>(1);
    double* d_oy   = dev_alloc<double>(1);

    oracle::launch_resize_to_ideal_length_kernel(
        n, e,
        plan.lower_edge_source, plan.lower_edge_target, plan.lower_edge_length,
        state.position_x, state.position_y,
        refresh_box, d_ob, d_ox, d_oy, d_fs);
    chk(cudaGetLastError(),      "resize kernel launch");
    chk(cudaDeviceSynchronize(), "resize kernel sync");

    if (refresh_box && dev_dl1(d_fs) == 0) {
        dev_d2d(d_ob, state.box_length,  1);
        dev_d2d(d_ox, state.down_left_x, 1);
        dev_d2d(d_oy, state.down_left_y, 1);
    }
    dev_free(d_fs); dev_free(d_ob); dev_free(d_ox); dev_free(d_oy);
}

} // anonymous namespace

cudaError_t fmmm_launch_device_hierarchy(
    const FmmmDeviceHierarchyInput& input,
    const FmmmDeviceHierarchyLaunchConfig& config,
    const FmmmDeviceHierarchyOutput& output,
    cudaStream_t stream) {
    if (input.node_count < 0 || config.cell_capacity < 1 ||
        config.leaf_capacity < 1 || config.max_depth < 0 ||
        input.box_length <= 0.0 || !std::isfinite(input.box_length)) {
        return cudaErrorInvalidValue;
    }
    if (!output.cell_count_device || !output.root_count_device ||
        !output.tree_epoch_device || !output.position_epoch_device ||
        !output.box_epoch_device || !output.status_device) {
        return cudaErrorInvalidValue;
    }
    if (input.node_count > 0 &&
        (!input.position_x || !input.position_y ||
         !output.cell_down_left_x || !output.cell_down_left_y ||
         !output.cell_box_length || !output.cell_center_x ||
         !output.cell_center_y || !output.cell_parent ||
         !output.cell_children4 || !output.cell_level ||
         !output.cell_compact_id || !output.cell_particle_count ||
         !output.particle_to_leaf)) {
        return cudaErrorInvalidValue;
    }

    // Select a deterministic depth without observing device state.  With no
    // explicit depth, stop once the nominal leaf load is within the requested
    // capacity; otherwise honor the caller's depth up to the capacity bound.
    const int requested_depth = std::min(
        config.max_depth > 0 ? config.max_depth : 15, 15);
    int depth = 0;
    long long cell_count = 1;
    long long level_width = 1;
    while (depth < requested_depth) {
        const long long next_width = level_width * 4LL;
        const long long next_count = cell_count + next_width;
        if (next_count > static_cast<long long>(config.cell_capacity) ||
            next_count > static_cast<long long>(std::numeric_limits<int>::max())) {
            break;
        }
        ++depth;
        level_width = next_width;
        cell_count = next_count;
        if (config.max_depth == 0 && depth >= 1 &&
            level_width * static_cast<long long>(config.leaf_capacity) >=
                static_cast<long long>(std::max(1, input.node_count))) {
            break;
        }
    }

    const int cells = input.node_count > 0 ? static_cast<int>(cell_count) : 0;
    const int init_blocks = std::max(1, (cells + 127) / 128);
    device_hierarchy_init_kernel<<<init_blocks, 128, 0, stream>>>(
        cells, depth, input.node_count,
        input.down_left_x, input.down_left_y, input.box_length,
        input.position_epoch, input.box_epoch,
        config.tree_epoch,
        output.cell_down_left_x, output.cell_down_left_y,
        output.cell_box_length, output.cell_center_x, output.cell_center_y,
        output.cell_parent, output.cell_children4, output.cell_level,
        output.cell_compact_id, output.cell_particle_count,
        output.cell_count_device, output.root_count_device,
        output.tree_epoch_device, output.position_epoch_device,
        output.box_epoch_device, output.status_device);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    if (input.node_count > 0) {
        const int blocks = (input.node_count + 127) / 128;
        device_hierarchy_assign_particles_kernel<<<blocks, 128, 0, stream>>>(
            input.node_count, depth, cells,
            input.position_x, input.position_y,
            input.down_left_x, input.down_left_y, input.box_length,
            output.particle_to_leaf, output.cell_particle_count,
            output.status_device);
        err = cudaGetLastError();
    }
    return err;
}

// ============================================================
// fmmm_build_hierarchy   (fine → coarse)
// ============================================================

FmmmStageResult fmmm_build_hierarchy(FmmmHierarchyState& state,
                                     const FmmmGraphView& graph,
                                     const FmmmProfile&   profile) {
    FmmmStageResult result{};
    result.completed_stage = FmmmPipelineStage::none;

    if (graph.node_count <= 0 || graph.edge_count < 0 ||
        state.levels == nullptr || state.capacity < 1) {
        result.status = FmmmPipelineStatus::invalid_input;
        result.reason = "fmmm_build_hierarchy: invalid input";
        return result;
    }

    // OGDF minGraphSize. Optional test override: when profile.max_level > 0 and
    // profile.act_level > 0, treat act_level as minGraphSize (diagnostic hook).
    const int min_graph_size =
        (profile.max_level > 0 && profile.act_level > 0)
            ? profile.act_level
            : kFmmmDefaultMinGraphSize;
    const int kMinNodes = std::max(1, min_graph_size);

    // G2 single-level OGDF branch: refuse hierarchy construction (use fmmm_run_single_level).
    {
        FmmmStageResult gate = fmmm_hierarchy_reject_if_single_level(graph.node_count, kMinNodes);
        if (gate.status != FmmmPipelineStatus::ok) {
            return gate;
        }
    }
    result.completed_stage = FmmmPipelineStage::hierarchy;

    state.level_count = 0;

    // ---- Level 0: upload original fine graph --------------------------------
    {
        const int n0 = graph.node_count;
        const int e0 = graph.edge_count;
        const int e1 = std::max(e0, 1);

        // Over-allocate high/lambda/nbr/moon to fine edge count (upper bound).
        alloc_plan(state.levels[0].plan,
                   n0, e0,
                   /*hn=*/n0, /*he=*/e1,
                   /*lam=*/e1 * 2, /*nbr=*/e1 * 2, /*moon_sz=*/n0);
        alloc_state(state.levels[0].state, n0);

        auto& p0 = state.levels[0].plan;
        dev_upload(graph.edge_source, const_cast<int*>(p0.lower_edge_source), e0);
        dev_upload(graph.edge_target, const_cast<int*>(p0.lower_edge_target), e0);
        dev_upload(graph.edge_length, const_cast<double*>(p0.lower_edge_length), e0);

        state.levels[0].level       = 0;
        state.levels[0].prng_before = profile.seed;
        state.levels[0].prng_after  = profile.seed;
        state.level_count = 1;
    }

    // ---- Coarsening loop: level k → k+1 -------------------------------------
    int  bad_edge_ctr = 0;
    int  prev_edges   = graph.edge_count;
    auto cur_seed     = profile.seed;

    // Pre-allocated scratch arena for coarsening stages.
    // Multilevel hierarchy sizes decrease monotonically (N_{k+1} < N_k, E_{k+1} <= E_k),
    // so sizing scratch buffers to the base fine graph (N_0, E_0) guarantees sufficient
    // capacity for all coarse tiers while eliminating repetitive driver allocations.
    const int max_scratch_n = std::max(graph.node_count, 1);
    const int max_scratch_e = std::max(graph.edge_count, 1);

    int*    d_scalars     = dev_alloc<int>(5); // status, coarse_n, coarse_e, high_e_span, draws
    int*    d_parent      = dev_alloc<int>(max_scratch_n);
    int*    d_type        = dev_alloc<int>(max_scratch_n);
    int*    d_sun         = dev_alloc<int>(max_scratch_n);
    double* d_dist        = dev_alloc<double>(max_scratch_n);
    int*    d_csrc        = dev_alloc<int>(max_scratch_e);
    int*    d_ctgt        = dev_alloc<int>(max_scratch_e);
    double* d_clen        = dev_alloc<double>(max_scratch_e);
    FmmmWorkspace ws      = make_ws(max_scratch_n, max_scratch_e);

    int* d_status      = d_scalars;
    int* d_coarse_n    = d_scalars + 1;
    int* d_coarse_e    = d_scalars + 2;
    int* d_high_e_span = d_scalars + 3;
    int* d_draws       = d_scalars + 4;

    while (state.level_count < state.capacity) {
        const int k   = state.level_count - 1;
        auto& cur     = state.levels[k];
        const int ln  = cur.plan.lower_node_count;
        const int le  = cur.plan.lower_edge_ordinal_span;
        const int le1 = std::max(le, 1);

        // OGDF stop: nodes <= minGraphSize
        if (ln <= kMinNodes) break;
        // OGDF stop: edge linearity (only after the first coarsening)
        if (k > 0 && !edge_linearity_ok(prev_edges, le, bad_edge_ctr)) break;

        cur.prng_before = cur_seed;

        // Launch the solar-hierarchy kernel.
        // Outputs written directly into cur.plan for adjacency/lambda/moon/high.
        launch_full_path_solar_hierarchy_kernel(
            ln, le,
            cur.plan.lower_edge_source,
            cur.plan.lower_edge_target,
            cur.plan.lower_edge_length,
            cur_seed,
            /*status*/           d_status,
            /*parent*/           d_parent,
            /*node_type*/        d_type,
            /*dedicated_sun*/    d_sun,
            /*sun_distance*/     d_dist,
            /*coarse_node_count*/d_coarse_n,
            /*coarse_edge_count*/d_coarse_e,
            /*coarse_source*/    d_csrc,
            /*coarse_target*/    d_ctgt,
            /*coarse_length*/    d_clen,
            // plan arrays (written directly by kernel)
            const_cast<std::uint8_t*>(cur.plan.lower_edge_is_moon),
            const_cast<int*>(cur.plan.lower_adjacency_begin),
            const_cast<int*>(cur.plan.lower_adjacency_edge),
            const_cast<int*>(cur.plan.lambda_begin),
            const_cast<double*>(cur.plan.lambda_values),
            const_cast<int*>(cur.plan.neighbour_sun_begin),
            const_cast<int*>(cur.plan.neighbour_sun_nodes),
            const_cast<int*>(cur.plan.moon_begin),
            const_cast<int*>(cur.plan.moon_nodes),
            const_cast<int*>(cur.plan.high_lower_node),
            /*plan_high_edge_span (output scalar)*/d_high_e_span,
            const_cast<int*>(cur.plan.high_edge_source),
            const_cast<int*>(cur.plan.high_edge_target),
            const_cast<std::uint8_t*>(cur.plan.high_edge_is_extra),
            const_cast<int*>(cur.plan.high_adjacency_begin),
            const_cast<int*>(cur.plan.high_adjacency_edge),
            d_draws,
            ws.view());
        chk(cudaGetLastError(),      "solar hierarchy kernel launch");
        chk(cudaDeviceSynchronize(), "solar hierarchy kernel sync");

        // --- Copy the kernel's node-SoA outputs into cur.plan lower fields ---
        dev_d2d(d_type, const_cast<int*>(cur.plan.lower_type),                   ln);
        dev_d2d(d_sun,  const_cast<int*>(cur.plan.lower_dedicated_sun),           ln);
        dev_d2d(d_dist, const_cast<double*>(cur.plan.lower_dedicated_sun_distance), ln);

        // --- Read scalar outputs ---
        const int status_val  = dev_dl1(d_status);
        const int high_n      = dev_dl1(d_coarse_n);
        const int high_e      = dev_dl1(d_coarse_e);
        const int high_e_span = dev_dl1(d_high_e_span);
        const int solar_draws = dev_dl1(d_draws);
        // Publish solar draw boundary: same seed engine (OGDF reseeds phases);
        // prng_after carries draws-consumed so AB can observe a real boundary.
        cur.prng_after = static_cast<std::uint64_t>(solar_draws);
        if (cur.state.consumed_draws != nullptr) {
            chk(cudaMemcpy(cur.state.consumed_draws, d_draws, sizeof(int),
                           cudaMemcpyDeviceToDevice), "solar draws D2D");
        }

        if (status_val != 0 || high_n <= 0) {
            // Kernel failed or graph is already minimal — stop coarsening.
            break;
        }

        // Update cur.plan's high-graph dimension fields.
        cur.plan.high_node_count       = high_n;
        cur.plan.high_edge_ordinal_span = high_e_span;

        // --- Allocate next level with the coarse graph as its lower graph ---
        auto& nxt = state.levels[state.level_count];
        nxt.level       = state.level_count;
        nxt.prng_before = cur_seed;
        nxt.prng_after  = 0;  // filled when this level is coarsened (or left 0 at coarsest)

        const int ne1 = std::max(high_e, 1);
        alloc_plan(nxt.plan,
                   high_n, high_e,
                   /*hn_ub=*/high_n, /*he_ub=*/ne1,
                   /*lam=*/ne1 * 2, /*nbr=*/ne1 * 2, /*moon_sz=*/high_n);
        alloc_state(nxt.state, high_n);
        nxt.plan.lower_node_count        = high_n;
        nxt.plan.lower_edge_ordinal_span = high_e;

        // Upload the coarse edges into the next level's lower_edge_*.
        dev_d2d(d_csrc, const_cast<int*>(nxt.plan.lower_edge_source), high_e);
        dev_d2d(d_ctgt, const_cast<int*>(nxt.plan.lower_edge_target), high_e);
        dev_d2d(d_clen, const_cast<double*>(nxt.plan.lower_edge_length), high_e);

        prev_edges = high_e;
        state.level_count++;
    }

    // Release pre-allocated scratch arena
    dev_free(d_scalars);
    dev_free(d_parent);
    dev_free(d_type);
    dev_free(d_sun);
    dev_free(d_dist);
    dev_free(d_csrc);
    dev_free(d_ctgt);
    dev_free(d_clen);
    ws.release();

        prev_edges = high_e;
        state.level_count++;
    }

    // Update profile-mirrored fields for execute.
    // max_level is the final level index (0-based), matching OGDF's max_level.
    // (stored in the caller's profile if needed; here we just track internally)

    result.status = FmmmPipelineStatus::ok;
    return result;
}

// ============================================================
// fmmm_execute_hierarchy   (coarsest → finest)
// ============================================================

FmmmStageResult fmmm_execute_hierarchy(FmmmHierarchyState& state) {
    FmmmStageResult result{};
    result.completed_stage = FmmmPipelineStage::uncoarsen;

    if (state.level_count <= 0 || state.levels == nullptr) {
        result.status = FmmmPipelineStatus::invalid_input;
        result.reason = "fmmm_execute_hierarchy: empty hierarchy";
        return result;
    }

    const int max_level = state.level_count - 1;   // OGDF max_level (0-based)
    auto& top = state.levels[max_level];
    const int top_n = top.plan.lower_node_count;

    // ---- Coarsest placement -------------------------------------------------
    // Mirrors FMMMLayout::create_initial_placement with RandomRandIterNr.
    // Node width/height match the frozen OGDF profile (20x20 by default).
    constexpr double kW = 20.0, kH = 20.0;
    int current_consumed_draws = 2 * top_n;

    {
        int*    d_ps  = dev_alloc<int>(1);
        double* d_ib  = dev_alloc<double>(1);  // initial_box (not stored)
        double* d_db  = dev_alloc<double>(1);  // dynamic_box
        double* d_dlx = dev_alloc<double>(1);
        double* d_dly = dev_alloc<double>(1);

        launch_full_path_coarsest_placement_kernel(
            top_n, kW, kH,
            top.prng_before,
            d_ps, d_ib, d_db, d_dlx, d_dly,
            top.state.position_x, top.state.position_y);
        chk(cudaGetLastError(),      "coarsest placement launch");
        chk(cudaDeviceSynchronize(), "coarsest placement sync");

        const int ps = dev_dl1(d_ps);
        dev_free(d_ps); dev_free(d_ib);

        if (ps != 0) {
            dev_free(d_db); dev_free(d_dlx); dev_free(d_dly);
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = "fmmm_execute_hierarchy: coarsest placement failed";
            return result;
        }

        dev_d2d(d_db,  top.state.box_length,  1);
        dev_d2d(d_dlx, top.state.down_left_x, 1);
        dev_d2d(d_dly, top.state.down_left_y, 1);
        dev_free(d_db); dev_free(d_dlx); dev_free(d_dly);
    }

    // Frozen OGDF profile constants (FmmmHierarchyState does not retain profile).
    // Matches tools/ogdf_fmmm_runner/hierarchy_oracle.cpp::configure_profile.
    constexpr int kFixed = 3;
    constexpr int kFactor = 10;
    constexpr int kPostIters = 10;
    constexpr int kFineIters = 20;
    // NMM tree builder consumes this seed; prefer the coarsest level's layout seed.
    const std::uint64_t layout_seed = top.prng_before ? top.prng_before : 100;

    // OGDF reseeds once in create_initial_placement (RandomRandIterNr), then
    // Advanced uncoarsen continues the same mt19937 across levels. Skip the
    // coarsest 2*n placement draws with randomNumber(0, BILLION); later levels
    // also replay prior-level waggle draws with randomNumber(1, BILLION).
    const int initial_placement_draws = current_consumed_draws;
    int current_total_draws = current_consumed_draws;

    auto run_level_force_and_maybe_post = [&](int act_level,
                                              oracle::FmmmAdvancedUncoarsenPlanView& plan,
                                              oracle::FmmmAdvancedUncoarsenStateView& st,
                                              const char** fail_reason) -> bool {
        // call_FORCE_CALCULATION_step: normal forces, then postprocess on act_level==0.
        if (!run_force_loop(plan, st, act_level, max_level, kFixed, kFactor, /*force_phase=*/0,
                            /*explicit_iters=*/-1, layout_seed, fail_reason)) {
            return false;
        }
        if (act_level == 0) {
            if (!run_force_loop(plan, st, act_level, max_level, kFixed, kFactor,
                                /*force_phase=*/1, /*explicit_iters=*/kPostIters, layout_seed,
                                fail_reason)) {
                return false;
            }
            run_resize(plan, st, /*refresh_box=*/1);
            if (!run_force_loop(plan, st, act_level, max_level, kFixed, kFactor,
                                /*force_phase=*/2, /*explicit_iters=*/kFineIters, layout_seed,
                                fail_reason)) {
                return false;
            }
            run_resize(plan, st, /*refresh_box=*/0);
        }
        return true;
    };

    // ---- Force (+ post if single-level table) on coarsest --------------------
    {
        const char* force_fail = nullptr;
        if (!run_level_force_and_maybe_post(max_level, top.plan, top.state, &force_fail)) {
            result.status = FmmmPipelineStatus::backend_error;
            result.reason = force_fail ? force_fail : "fmmm_execute_hierarchy: coarsest force failed";
            return result;
        }
    }

    // ---- Uncoarsening loop  (max_level - 1 down to 0) -----------------------
    for (int k = max_level - 1; k >= 0; --k) {
        auto& cur = state.levels[k];
        auto& parent = state.levels[k + 1];
        const int ln = cur.plan.lower_node_count;
        const int le = cur.plan.lower_edge_ordinal_span;

        // Publish parent force endpoint into this level's high_position plane,
        // then Advanced-place the child.
        dev_d2d(parent.state.position_x,
                const_cast<double*>(cur.plan.high_position_x),
                cur.plan.high_node_count);
        dev_d2d(parent.state.position_y,
                const_cast<double*>(cur.plan.high_position_y),
                cur.plan.high_node_count);

        FmmmWorkspace uncoarsen_ws = make_ws(ln, std::max(le, 1));
        oracle::FmmmAdvancedUncoarsenPrng prng;
        prng.seed = cur.prng_before;
        prng.consumed_draws = current_total_draws;
        prng.initial_placement_draws = initial_placement_draws;

        launch_advanced_uncoarsen_kernel(cur.plan, cur.state, prng, uncoarsen_ws.view(), nullptr);
        chk(cudaGetLastError(),      "uncoarsen launch");
        chk(cudaDeviceSynchronize(), "uncoarsen sync");

        int st = dev_dl1(cur.state.status);
        if (st != 0) {
            result.status = FmmmPipelineStatus::backend_error;
            if (st == 1) result.reason = "fmmm_execute_hierarchy: uncoarsen failed 1";
            else if (st == 2) result.reason = "fmmm_execute_hierarchy: uncoarsen failed 2";
            else result.reason = "fmmm_execute_hierarchy: uncoarsen failed (unknown code)";
            return result;
        }

        // Kernel reports total draws after this level's waggle; next uncoarsen
        // continues from that boundary. Diagnostic AB still compares the
        // oracle's recorded placement-only count separately.
        if (cur.state.consumed_draws != nullptr) {
            current_total_draws = dev_dl1(cur.state.consumed_draws);
        }

        // Preserve Advanced-placement endpoint in high_position for diagnostics.
        // Parent force coordinates were already consumed by the kernel above.
        dev_d2d(cur.state.position_x, const_cast<double*>(cur.plan.high_position_x), ln);
        dev_d2d(cur.state.position_y, const_cast<double*>(cur.plan.high_position_y), ln);

        // Release-after-consume: drop parent edge/adj/high-edge arrays. Keep
        // membership, lambda/neighbour, and high_position for diagnostic AB;
        // fmmm_release_hierarchy frees the retained pointers.
        {
            auto& pp = parent.plan;
            double* keep_hx = const_cast<double*>(pp.high_position_x);
            double* keep_hy = const_cast<double*>(pp.high_position_y);
            int* keep_type = const_cast<int*>(pp.lower_type);
            int* keep_sun = const_cast<int*>(pp.lower_dedicated_sun);
            double* keep_dist = const_cast<double*>(pp.lower_dedicated_sun_distance);
            int* keep_high_lower = const_cast<int*>(pp.high_lower_node);
            int* keep_lambda_begin = const_cast<int*>(pp.lambda_begin);
            double* keep_lambda_values = const_cast<double*>(pp.lambda_values);
            int* keep_nbr_begin = const_cast<int*>(pp.neighbour_sun_begin);
            int* keep_nbr_nodes = const_cast<int*>(pp.neighbour_sun_nodes);
            const_cast<double*&>(pp.high_position_x) = nullptr;
            const_cast<double*&>(pp.high_position_y) = nullptr;
            const_cast<int*&>(pp.lower_type) = nullptr;
            const_cast<int*&>(pp.lower_dedicated_sun) = nullptr;
            const_cast<double*&>(pp.lower_dedicated_sun_distance) = nullptr;
            const_cast<int*&>(pp.high_lower_node) = nullptr;
            const_cast<int*&>(pp.lambda_begin) = nullptr;
            const_cast<double*&>(pp.lambda_values) = nullptr;
            const_cast<int*&>(pp.neighbour_sun_begin) = nullptr;
            const_cast<int*&>(pp.neighbour_sun_nodes) = nullptr;
            free_plan(pp);
            const_cast<double*&>(pp.high_position_x) = keep_hx;
            const_cast<double*&>(pp.high_position_y) = keep_hy;
            const_cast<int*&>(pp.lower_type) = keep_type;
            const_cast<int*&>(pp.lower_dedicated_sun) = keep_sun;
            const_cast<double*&>(pp.lower_dedicated_sun_distance) = keep_dist;
            const_cast<int*&>(pp.high_lower_node) = keep_high_lower;
            const_cast<int*&>(pp.lambda_begin) = keep_lambda_begin;
            const_cast<double*&>(pp.lambda_values) = keep_lambda_values;
            const_cast<int*&>(pp.neighbour_sun_begin) = keep_nbr_begin;
            const_cast<int*&>(pp.neighbour_sun_nodes) = keep_nbr_nodes;
        }

        {
            const char* force_fail = nullptr;
            if (!run_level_force_and_maybe_post(k, cur.plan, cur.state, &force_fail)) {
                result.status = FmmmPipelineStatus::backend_error;
                result.first_bad_level = k;
                result.reason = force_fail ? force_fail : "fmmm_execute_hierarchy: level force failed";
                return result;
            }
        }
    }

    result.status = FmmmPipelineStatus::ok;
    return result;
}

void fmmm_release_hierarchy(FmmmHierarchyState& state) {
    if (state.levels == nullptr || state.level_count <= 0) {
        if (state.levels != nullptr) state.level_count = 0;
        return;
    }
    for (int i = 0; i < state.level_count; ++i) {
        free_plan(state.levels[i].plan);
        free_state(state.levels[i].state);
        state.levels[i].prng_before = 0;
        state.levels[i].prng_after = 0;
        state.levels[i].level = 0;
    }
    state.level_count = 0;
}

} // namespace lcn::release
