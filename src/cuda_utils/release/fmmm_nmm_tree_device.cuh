#pragma once
#include "fmmm_nmm_packing.h"
#include <cuComplex.h>
#include <cooperative_groups.h>

namespace lcn::release {

struct DeviceTreeBuilder {
    FmmmNmmState state;
    int particles_in_leaf;
    int precision;
    std::uint64_t seed;
    
    // Bump allocators (passed as pointers to global memory)
    int* p_next_cell;
    int* p_next_contained;
    int* p_next_I;
    int* p_next_D1;
    int* p_next_D2;
    int* p_next_M;
    int* p_status;
    int* p_overflow;
    int max_cells;
    int max_contained;
    int max_list;
    int queue_capacity;
    int frontier_queue_capacity;
    int traversal_queue_capacity;
    int max_I;
    int max_D1;
    int max_D2;
    int max_M;

    // Queue for BFS
    int* q_cells;
    int* traversal_q_cells;
    int q_head;
    int q_tail;

    __device__ void overflow(int category) {
        atomicAdd(&p_overflow[category], 1);
        atomicCAS(p_status, 0, category + 1);
    }
    __device__ void push_q(int cell) {
        if (q_tail >= frontier_queue_capacity) { overflow(kNmmOverflowFrontierQueue); return; }
        q_cells[q_tail++] = cell;
    }
    __device__ int pop_q() {
        return q_cells[q_head++];
    }
    __device__ bool q_empty() {
        return q_head == q_tail;
    }

    __device__ void push_contained(int cell, int particle) {
        int idx = *p_next_contained;
        if (idx >= max_contained) { overflow(kNmmOverflowContained); return; }
        (*p_next_contained)++;
        state.contained_nodes[idx] = particle;
    }
    __device__ void push_I(int cell, int neighbor) {
        int idx = *p_next_I;
        if (idx >= max_I) { overflow(kNmmOverflowIList); return; }
        (*p_next_I)++;
        state.I_nodes[idx] = neighbor;
    }
    __device__ void push_D1(int cell, int neighbor) {
        int idx = *p_next_D1;
        if (idx >= max_D1) { overflow(kNmmOverflowD1List); return; }
        (*p_next_D1)++;
        state.D1_nodes[idx] = neighbor;
    }
    __device__ void push_D2(int cell, int neighbor) {
        int idx = *p_next_D2;
        if (idx >= max_D2) { overflow(kNmmOverflowD2List); return; }
        (*p_next_D2)++;
        state.D2_nodes[idx] = neighbor;
    }
    __device__ void push_M(int cell, int neighbor) {
        int idx = *p_next_M;
        if (idx >= max_M) { overflow(kNmmOverflowMList); return; }
        (*p_next_M)++;
        state.M_nodes[idx] = neighbor;
    }

    // RNG for center
    __device__ double get_rand_y(int cell_id) {
        unsigned long long s = seed + cell_id;
        s = s * 6364136223846793005ULL + 1;
        int r = (s >> 33) & 0x7FFFFFFF;
        return (double)(r % 1000000000 + 1) / 1000000002.0;
    }
    
    __device__ void set_center(int cell) {
        double cx = state.cell_downleft_x[cell] + state.cell_box_length[cell] * 0.5;
        double cy = state.cell_downleft_y[cell] + state.cell_box_length[cell] * 0.5;
        cy += 0.001 * state.cell_box_length[cell] * get_rand_y(cell);
        state.cell_center_x[cell] = cx;
        state.cell_center_y[cell] = cy;
    }

    __device__ bool is_leaf(int cell) {
        return state.cell_child4[cell * 4] < 0;
    }

    __device__ bool nearly_equal(double a, double b) {
        double diff = a - b;
        return (diff >= -1e-8 && diff <= 1e-8);
    }

    __device__ bool well_separated(int a, int b) {
        double box1 = state.cell_box_length[a];
        double box2 = state.cell_box_length[b];
        double a_x = state.cell_downleft_x[a];
        double a_y = state.cell_downleft_y[a];
        double b_x = state.cell_downleft_x[b];
        double b_y = state.cell_downleft_y[b];
        
        double x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max;
        if (box1 <= box2) {
            x1_min = a_x; x1_max = a_x + box1;
            y1_min = a_y; y1_max = a_y + box1;
            x2_min = b_x - box2; x2_max = b_x + 2 * box2;
            y2_min = b_y - box2; y2_max = b_y + 2 * box2;
        } else {
            x1_min = a_x - box1; x1_max = a_x + 2 * box1;
            y1_min = a_y - box1; y1_max = a_y + 2 * box1;
            x2_min = b_x; x2_max = b_x + box2;
            y2_min = b_y; y2_max = b_y + box2;
        }
        
        bool x_overlap = !(x1_max <= x2_min || nearly_equal(x1_max, x2_min) ||
                           x2_max <= x1_min || nearly_equal(x2_max, x1_min));
        bool y_overlap = !(y1_max <= y2_min || nearly_equal(y1_max, y2_min) ||
                           y2_max <= y1_min || nearly_equal(y2_max, y1_min));
        return !(x_overlap && y_overlap);
    }

    __device__ bool bordering(int a, int b) {
        if (well_separated(a, b)) return false;
        double dx = state.cell_center_x[a] - state.cell_center_x[b];
        double dy = state.cell_center_y[a] - state.cell_center_y[b];
        if (dx < 0) dx = -dx;
        if (dy < 0) dy = -dy;
        double sum = (state.cell_box_length[a] + state.cell_box_length[b]) * 0.5 + 1e-4;
        return (dx <= sum && dy <= sum);
    }
    
    // Complex arithmetic
    __device__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b)); }
    __device__ cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b)); }
    __device__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
    __device__ cuDoubleComplex cdiv_real(cuDoubleComplex a, double b) { return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b); }
    __device__ cuDoubleComplex cneg(cuDoubleComplex a) { return make_cuDoubleComplex(-cuCreal(a), -cuCimag(a)); }

    __device__ void form_multipole_leaf(int cell) {
        int stride = state.expansion_stride;
        for (int k = 0; k < stride; ++k) {
            state.multipole_re[cell * stride + k] = 0;
            state.multipole_im[cell * stride + k] = 0;
        }
        state.multipole_re[cell * stride + 0] = state.cell_particle_count[cell];
        
        cuDoubleComplex z0 = make_cuDoubleComplex(state.cell_center_x[cell], state.cell_center_y[cell]);
        
        int b = state.contained_begin[cell];
        int e = state.contained_begin[cell + 1];
        for (int i = b; i < e; ++i) {
            int v = state.contained_nodes[i];
            cuDoubleComplex zv = make_cuDoubleComplex(state.position_x[v], state.position_y[v]);
            cuDoubleComplex z = csub(zv, z0);
            cuDoubleComplex pow = z;
            for (int k = 1; k <= precision; ++k) {
                cuDoubleComplex term = cdiv_real(cneg(pow), (double)k);
                state.multipole_re[cell * stride + k] += cuCreal(term);
                state.multipole_im[cell * stride + k] += cuCimag(term);
                pow = cmul(pow, z);
            }
        }
    }
    
    __device__ void shift_multipole_to_parent(int child, int father) {
        int stride = state.expansion_stride;
        cuDoubleComplex z0 = make_cuDoubleComplex(state.cell_center_x[child], state.cell_center_y[child]);
        cuDoubleComplex z1 = make_cuDoubleComplex(state.cell_center_x[father], state.cell_center_y[father]);
        cuDoubleComplex dz = csub(z0, z1);
        
        cuDoubleComplex zdiff[10]; // assuming max precision < 10
        zdiff[0] = make_cuDoubleComplex(1.0, 0.0);
        for (int i = 1; i <= precision; ++i) {
            zdiff[i] = cmul(zdiff[i - 1], dz);
        }
        
        state.multipole_re[father * stride + 0] += state.multipole_re[child * stride + 0];
        for (int k = 1; k <= precision; ++k) {
            cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
            for (int l = 1; l <= k; ++l) {
                // binko(k-1, l-1) can be computed
                long long b = 1;
                int n = k - 1; int c = l - 1;
                for (int i = 1; i <= c; ++i) {
                    b = (b * (n - i + 1)) / i;
                }
                cuDoubleComplex ml = make_cuDoubleComplex(state.multipole_re[child * stride + l], state.multipole_im[child * stride + l]);
                cuDoubleComplex term = cmul(ml, zdiff[k - l]);
                term = make_cuDoubleComplex(cuCreal(term) * b, cuCimag(term) * b);
                sum = cadd(sum, term);
            }
            state.multipole_re[father * stride + k] += cuCreal(sum);
            state.multipole_im[father * stride + k] += cuCimag(sum);
        }
    }

};

// Production topology pipeline.  Frontier entries are ordered by parent
// position and the fixed OGDF child order (2,3,0,1).  Each phase is launched
// over the whole capacity; the device frontier count gates inactive work so
// the host never polls or downloads a level counter.
__global__ void nmm_frontier_init_kernel(
        FmmmNmmState state, int* counters, int* overflow,
        int max_cells, int max_contained,
        double root_min_x, double root_min_y,
        double root_max_x, double root_max_y,
        int* frontier, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < state.node_count && state.temp_contained_nodes) state.temp_contained_nodes[i] = i;
    if (i != 0) return;
    if (!counters || !overflow || !frontier || !status || max_cells < 1 ||
            max_contained < state.node_count) {
        if (overflow) atomicAdd(&overflow[kNmmOverflowCell], 1);
        if (status) atomicCAS(status, 0, kNmmOverflowCell + 1);
        return;
    }
    counters[0] = 1;  // cell_count
    counters[1] = 1;  // current frontier count
    counters[2] = 0;  // next frontier count
    counters[7] = state.node_count;  // temporary membership append cursor
    state.cell_downleft_x[0] = root_min_x;
    state.cell_downleft_y[0] = root_min_y;
    double box = root_max_x - root_min_x;
    if (root_max_y - root_min_y > box) box = root_max_y - root_min_y;
    state.cell_box_length[0] = box;
    state.cell_level[0] = 0;
    state.cell_parent[0] = -1;
    state.cell_particle_count[0] = state.node_count;
    state.temp_contained_begin[0] = 0;
    state.temp_contained_begin[1] = state.node_count;
    for (int k = 0; k < 4; ++k) state.cell_child4[k] = -1;
    frontier[0] = 0;
}

__global__ void nmm_frontier_classify_count_kernel(
        FmmmNmmState state, const int* counters, int particles_in_leaf,
        const int* frontier, int max_cells, int* child_counts) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos >= frontier_count || pos >= max_cells) return;
    const int cell = frontier[pos];
    if (state.cell_particle_count[cell] <= particles_in_leaf) {
        for (int k = 0; k < 4; ++k) child_counts[pos * 4 + k] = 0;
        return;
    }
    const double half = state.cell_box_length[cell] * 0.5;
    const double mid_x = state.cell_downleft_x[cell] + half;
    const double mid_y = state.cell_downleft_y[cell] + half;
    int local[4] = {0, 0, 0, 0};
    for (int i = state.temp_contained_begin[cell];
             i < state.temp_contained_begin[cell + 1]; ++i) {
        const int p = state.temp_contained_nodes[i];
        const bool right = state.position_x[p] > mid_x;
        const bool upper = state.position_y[p] > mid_y;
        ++local[(right ? 1 : 0) | (upper ? 2 : 0)];
    }
    for (int k = 0; k < 4; ++k) child_counts[pos * 4 + k] = local[k];
}

__global__ void nmm_frontier_prepare_scan_kernel(
        const int* counters, const int* frontier, int max_cells,
        const int* child_counts, int* parent_counts, int* subdivide_flags) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos >= frontier_count || pos >= max_cells) return;
    const int cell = frontier[pos];
    parent_counts[pos] = child_counts[pos * 4] + child_counts[pos * 4 + 1] +
                         child_counts[pos * 4 + 2] + child_counts[pos * 4 + 3];
    subdivide_flags[pos] = parent_counts[pos] > 0 ? 1 : 0;
    (void)cell;
}

// Fused classify+prepare: one launch writes child_counts, parent_counts, and
// subdivide_flags for the current frontier.
__global__ void nmm_frontier_classify_prepare_kernel(
        FmmmNmmState state, const int* counters, int particles_in_leaf,
        const int* frontier, int max_cells, int* child_counts,
        int* parent_counts, int* subdivide_flags) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos >= frontier_count || pos >= max_cells) return;
    const int cell = frontier[pos];
    if (state.cell_particle_count[cell] <= particles_in_leaf) {
        for (int k = 0; k < 4; ++k) child_counts[pos * 4 + k] = 0;
        parent_counts[pos] = 0;
        subdivide_flags[pos] = 0;
        return;
    }
    const double half = state.cell_box_length[cell] * 0.5;
    const double mid_x = state.cell_downleft_x[cell] + half;
    const double mid_y = state.cell_downleft_y[cell] + half;
    int local[4] = {0, 0, 0, 0};
    for (int i = state.temp_contained_begin[cell];
             i < state.temp_contained_begin[cell + 1]; ++i) {
        const int p = state.temp_contained_nodes[i];
        const bool right = state.position_x[p] > mid_x;
        const bool upper = state.position_y[p] > mid_y;
        ++local[(right ? 1 : 0) | (upper ? 2 : 0)];
    }
    int sum = 0;
    for (int k = 0; k < 4; ++k) {
        child_counts[pos * 4 + k] = local[k];
        sum += local[k];
    }
    parent_counts[pos] = sum;
    subdivide_flags[pos] = sum > 0 ? 1 : 0;
}

__global__ void nmm_frontier_scatter_kernel(
        FmmmNmmState state, int* counters, int* overflow,
        int particles_in_leaf, const int* frontier, int max_cells,
        const int* child_counts, const int* parent_offsets,
        const int* child_offsets,
        int* next_frontier, int* active_counts, int* status) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos >= frontier_count || pos >= max_cells) return;
    const int parent = frontier[pos];
    if (state.cell_particle_count[parent] <= particles_in_leaf) {
        active_counts[pos] = 0;
        return;
    }
    const int old_cell_count = counters[0];
    if (old_cell_count + (child_offsets[frontier_count] * 4) > state.max_cells) {
        atomicAdd(&overflow[kNmmOverflowCell], 1);
        atomicCAS(status, 0, kNmmOverflowCell + 1);
        return;
    }
    if (child_offsets[pos + 1] == child_offsets[pos]) {
        active_counts[pos] = 0;
        return;
    }
    const int child_base = old_cell_count + child_offsets[pos] * 4;
    const int append_base = counters[7] + parent_offsets[pos];
    const double half = state.cell_box_length[parent] * 0.5;
    const double mid_x = state.cell_downleft_x[parent] + half;
    const double mid_y = state.cell_downleft_y[parent] + half;
    const int child_order[4] = {2, 3, 0, 1};
    int local[4] = {0, 0, 0, 0};
    int child_cursor[4] = {0, 0, 0, 0};
    for (int slot = 0; slot < 4; ++slot) {
        const int child = child_base + slot;
        const int q = child_order[slot];
        const int begin = append_base + local[0] + local[1] + local[2] + local[3];
        state.cell_parent[child] = parent;
        state.cell_level[child] = state.cell_level[parent] + 1;
        state.cell_box_length[child] = half;
        state.cell_downleft_x[child] = state.cell_downleft_x[parent] + ((q & 1) ? half : 0.0);
        state.cell_downleft_y[child] = state.cell_downleft_y[parent] + ((q & 2) ? half : 0.0);
        state.cell_particle_count[child] = child_counts[pos * 4 + q];
        state.cell_child4[parent * 4 + slot] = child;
        state.temp_contained_begin[child] = begin;
        state.temp_contained_begin[child + 1] = begin + child_counts[pos * 4 + q];
        for (int k = 0; k < 4; ++k) state.cell_child4[child * 4 + k] = -1;
        local[q] += child_counts[pos * 4 + q];
    }
    for (int i = state.temp_contained_begin[parent];
             i < state.temp_contained_begin[parent + 1]; ++i) {
        const int p = state.temp_contained_nodes[i];
        const bool right = state.position_x[p] > mid_x;
        const bool upper = state.position_y[p] > mid_y;
        const int q = (right ? 1 : 0) | (upper ? 2 : 0);
        const int slot = (q == 2 ? 0 : q == 3 ? 1 : q == 0 ? 2 : 3);
        const int dst = state.temp_contained_begin[child_base + slot] + child_cursor[q]++;
        if (dst >= state.max_contained) {
            atomicAdd(&overflow[kNmmOverflowContained], 1);
            atomicCAS(status, 0, kNmmOverflowContained + 1);
            return;
        }
        state.temp_contained_nodes[dst] = p;
    }
    int active = 0;
    for (int slot = 0; slot < 4; ++slot)
        if (state.cell_particle_count[child_base + slot] > particles_in_leaf) ++active;
    active_counts[pos] = active;
    (void)next_frontier;
    (void)counters;
}

__global__ void nmm_frontier_compact_kernel(
        const int* counters, const int* frontier, int max_cells,
        int particles_in_leaf, const int* child_counts,
        const int* child_offsets, const int* active_offsets,
        int* next_frontier) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos >= frontier_count || pos >= max_cells) return;
    const int parent = frontier[pos];
    if (active_offsets[pos + 1] == active_offsets[pos]) return;
    const int child_base = counters[0] + child_offsets[pos] * 4;
    int out = active_offsets[pos];
    for (int slot = 0; slot < 4; ++slot) {
        const int child = child_base + slot;
        const int q = slot == 0 ? 2 : slot == 1 ? 3 : slot == 2 ? 0 : 1;
        if (child < max_cells && child_counts[pos * 4 + q] > particles_in_leaf)
            next_frontier[out++] = child;
    }
    (void)parent;
}

// Compact next frontier and advance build counters in one launch.
__global__ void nmm_frontier_compact_update_kernel(
        int* counters, const int* frontier, int max_cells,
        int particles_in_leaf, const int* child_counts,
        const int* parent_offsets, const int* child_offsets,
        const int* active_offsets, int* next_frontier, int* status) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int frontier_count = counters ? counters[1] : 0;
    if (pos < frontier_count && pos < max_cells &&
            active_offsets[pos + 1] != active_offsets[pos]) {
        const int child_base = counters[0] + child_offsets[pos] * 4;
        int out = active_offsets[pos];
        for (int slot = 0; slot < 4; ++slot) {
            const int child = child_base + slot;
            const int q = slot == 0 ? 2 : slot == 1 ? 3 : slot == 2 ? 0 : 1;
            if (child < max_cells && child_counts[pos * 4 + q] > particles_in_leaf)
                next_frontier[out++] = child;
        }
    }
    if (pos != 0 || !counters || !status) return;
    const int old_count = counters[1];
    const int next_count = active_offsets[old_count];
    const int old_cells = counters[0];
    const int new_cells = old_cells + child_offsets[old_count] * 4;
    if (new_cells > max_cells) {
        atomicCAS(status, 0, kNmmOverflowCell + 1);
        return;
    }
    counters[0] = new_cells;
    counters[1] = next_count;
    counters[2] = 0;
    counters[7] += parent_offsets[old_count];
    (void)frontier;
}

// The frontier pipeline naturally numbers cells breadth-first.  OGDF's
// observable tree contract is preorder (and uses child order 2,3,0,1), so
// canonicalize the completed parallel tree with device-only rank passes.  The
// passes are independent per level and retain the exact topology while
// avoiding a serial DFS in the production path.
__global__ void nmm_tree_subtree_init_kernel(
        const int* counters, int max_cells, int* subtree) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counters ? counters[0] : 0;
    if (cell >= count || cell >= max_cells) return;
    subtree[cell] = 1;
}

__global__ void nmm_tree_subtree_accumulate_kernel(
        FmmmNmmState state, const int* counters, int max_cells,
        int level, int* subtree) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counters ? counters[0] : 0;
    if (cell >= count || cell >= max_cells || state.cell_level[cell] != level) return;
    const int parent = state.cell_parent[cell];
    if (parent >= 0) atomicAdd(&subtree[parent], subtree[cell]);
}

__global__ void nmm_tree_preorder_rank_kernel(
        const FmmmNmmState state, const int* counters, int max_cells,
        int level, int* rank, const int* subtree) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counters ? counters[0] : 0;
    if (cell >= count || cell >= max_cells || state.cell_level[cell] != level) return;
    if (state.cell_parent[cell] < 0) {
        rank[cell] = 0;
        return;
    }
    const int parent = state.cell_parent[cell];
    int slot = -1;
    for (int q = 0; q < 4; ++q)
        if (state.cell_child4[parent * 4 + q] == cell) { slot = q; break; }
    int offset = 1;
    for (int q = 0; q < slot; ++q) {
        const int sibling = state.cell_child4[parent * 4 + q];
        if (sibling >= 0) offset += subtree[sibling];
    }
    rank[cell] = rank[parent] + offset;
}

__global__ void nmm_tree_reorder_snapshot_kernel(
        FmmmNmmState state, const int* counters, int max_cells,
        const int* rank, int* subtree) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counters ? counters[0] : 0;
    if (cell >= count || cell >= max_cells) return;
    const int dst = rank[cell];
    state.cell_center_x[cell] = state.cell_downleft_x[cell];
    state.cell_center_y[cell] = state.cell_downleft_y[cell];
    state.local_re[cell] = state.cell_box_length[cell];
    subtree[2 * max_cells + cell] = state.cell_level[cell];
    subtree[3 * max_cells + cell] = state.cell_parent[cell];
    state.D2_begin[cell] = state.cell_particle_count[cell];
    state.D1_begin[cell] = state.temp_contained_begin[cell];
    for (int q = 0; q < 4; ++q)
        state.I_nodes[cell * 4 + q] = state.cell_child4[cell * 4 + q];
    __threadfence();
    (void)dst;
}

__global__ void nmm_tree_reorder_apply_kernel(
        FmmmNmmState state, const int* counters, int max_cells,
        const int* rank, const int* scratch) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = counters ? counters[0] : 0;
    if (cell >= count || cell >= max_cells) return;
    const int dst = rank[cell];
    state.cell_downleft_x[dst] = state.cell_center_x[cell];
    state.cell_downleft_y[dst] = state.cell_center_y[cell];
    state.cell_box_length[dst] = state.local_re[cell];
    state.cell_level[dst] = scratch[2 * max_cells + cell];
    const int old_parent = scratch[3 * max_cells + cell];
    state.cell_parent[dst] = old_parent < 0 ? -1 : rank[old_parent];
    state.cell_particle_count[dst] = state.D2_begin[cell];
    state.temp_contained_begin[dst] = state.D1_begin[cell];
    for (int q = 0; q < 4; ++q) {
        const int old_child = state.I_nodes[cell * 4 + q];
        state.cell_child4[dst * 4 + q] = old_child < 0 ? -1 : rank[old_child];
    }
}

__global__ void nmm_frontier_update_kernel(
        int* counters, int max_cells, const int* parent_offsets,
        const int* child_offsets, const int* active_offsets, int* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !counters || !status) return;
    const int old_count = counters[1];
    const int next_count = active_offsets[old_count];
    const int old_cells = counters[0];
    const int new_cells = old_cells + child_offsets[old_count] * 4;
    if (new_cells > max_cells) {
        atomicCAS(status, 0, kNmmOverflowCell + 1);
        return;
    }
    counters[0] = new_cells;
    counters[1] = next_count;
    counters[2] = 0;
    counters[7] += parent_offsets[old_count];
}

__global__ void nmm_leaf_count_kernel(
        FmmmNmmState state, const int* counters, int max_cells,
        int* leaf_counts) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = counters ? counters[0] : 0;
    if (cell >= cell_count || cell >= max_cells) return;
    const bool leaf = state.cell_child4[cell * 4] < 0;
    leaf_counts[cell] = leaf ? state.cell_particle_count[cell] : 0;
}

__global__ void nmm_leaf_scatter_kernel(
        FmmmNmmState state, const int* counters, int max_cells,
        const int* leaf_offsets) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = counters ? counters[0] : 0;
    if (cell >= cell_count || cell >= max_cells) return;
    if (state.cell_child4[cell * 4] >= 0) return;
    const int src = state.temp_contained_begin[cell];
    const int dst = leaf_offsets[cell];
    for (int i = 0; i < state.cell_particle_count[cell]; ++i)
        state.contained_nodes[dst + i] = state.temp_contained_nodes[src + i];
    state.contained_begin[cell] = dst;
}

__global__ void nmm_finalize_contained_kernel(
        FmmmNmmState state, int* counters, int max_cells,
        const int* leaf_offsets) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = counters ? counters[0] : 0;
    if (cell > cell_count || cell > max_cells) return;
    state.contained_begin[cell] = leaf_offsets[cell];
    if (cell == cell_count) counters[1] = leaf_offsets[cell_count];
}

__global__ void nmm_set_centers_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        std::uint64_t seed) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count || cell >= max_cells) return;
    DeviceTreeBuilder b{}; b.state = state; b.seed = seed;
    b.set_center(cell);
}

__global__ void nmm_multipole_level_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        int level) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count || cell >= max_cells || state.cell_level[cell] != level) return;
    DeviceTreeBuilder b{}; b.state = state; b.precision = state.expansion_stride - 1;
    if (b.is_leaf(cell)) {
        b.form_multipole_leaf(cell);
        return;
    }
    const int stride = state.expansion_stride;
    for (int k = 0; k < stride; ++k) {
        state.multipole_re[cell * stride + k] = 0;
        state.multipole_im[cell * stride + k] = 0;
    }
    for (int q = 0; q < 4; ++q) {
        const int child = state.cell_child4[cell * 4 + q];
        if (child >= 0) b.shift_multipole_to_parent(child, cell);
    }
}

// G2 minimal Dao-style experiment: keep the level dependency device-side and
// replace the 64 host-launched level kernels with one cooperative persistent
// work loop.  Counter slot 7 is no longer needed after frontier construction;
// it is reused as the device-side maximum-level telemetry scratch.
__global__ void nmm_multipole_persistent_kernel(
        FmmmNmmState state, int* counters, int max_cells) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) counters[7] = -1;
    grid.sync();

    const int cell_count = counters[0] < max_cells ? counters[0] : max_cells;
    for (int cell = tid; cell < cell_count; cell += stride) {
        atomicMax(&counters[7], state.cell_level[cell]);
    }
    grid.sync();

    const int max_level = counters[7];
    for (int level = max_level; level >= 0; --level) {
        for (int cell = tid; cell < cell_count; cell += stride) {
            if (state.cell_level[cell] != level) continue;
            DeviceTreeBuilder b{};
            b.state = state;
            b.precision = state.expansion_stride - 1;
            if (b.is_leaf(cell)) {
                b.form_multipole_leaf(cell);
                continue;
            }
            const int expansion_stride = state.expansion_stride;
            for (int k = 0; k < expansion_stride; ++k) {
                state.multipole_re[cell * expansion_stride + k] = 0;
                state.multipole_im[cell * expansion_stride + k] = 0;
            }
            for (int q = 0; q < 4; ++q) {
                const int child = state.cell_child4[cell * 4 + q];
                if (child >= 0) b.shift_multipole_to_parent(child, cell);
            }
        }
        grid.sync();
    }
}

// Deterministic WSPRL traversal state is independent for every target cell.
// A fixed local queue keeps the production path allocation-free; the queue is
// deliberately bounded by the ABI capacity and reports the same diagnostic
// category as the legacy path.  The accepted witnesses have a much smaller
// traversal frontier than this bound.
constexpr int kNmmParallelTraversalQueue = 2048;

__device__ inline void nmm_wsprl_counts_for_target(
        const FmmmNmmState& state, int act, int traversal_capacity,
        int& i_count, int& d1_count, int& d2_count, int& m_count,
        int* overflow, int* status) {
    DeviceTreeBuilder b{};
    b.state = state;
    b.p_status = status;
    b.p_overflow = overflow;
    b.traversal_queue_capacity = traversal_capacity;
    b.max_I = b.max_D1 = b.max_D2 = b.max_M = 0x7fffffff;
    int e_q[kNmmParallelTraversalQueue];
    int e_head = 0;
    int e_tail = 0;
    const int parent = state.cell_parent[act];
    auto enqueue = [&](int value) {
        if (e_tail >= traversal_capacity || e_tail >= kNmmParallelTraversalQueue) {
            b.overflow(kNmmOverflowTraversalQueue);
            return false;
        }
        e_q[e_tail++] = value;
        return true;
    };
    if (parent < 0) {
        for (int k = 0; k < 4; ++k) {
            const int ch = state.cell_child4[act * 4 + k];
            if (ch >= 0 && !enqueue(ch)) return;
        }
    } else {
        int pb = state.D1_begin[parent], pe = state.D1_begin[parent + 1];
        for (int x = pb; x < pe; ++x)
            if (!enqueue(state.D1_nodes[x])) return;
        pb = state.I_begin[parent]; pe = state.I_begin[parent + 1];
        for (int x = pb; x < pe; ++x)
            if (!enqueue(state.I_nodes[x])) return;
    }
    i_count = d1_count = d2_count = m_count = 0;
    while (e_head < e_tail) {
        const int sel = e_q[e_head++];
        if (b.well_separated(act, sel)) {
            ++m_count;
        } else if (state.cell_level[act] < state.cell_level[sel]) {
            ++i_count;
        } else if (!b.is_leaf(sel)) {
            for (int k = 0; k < 4; ++k) {
                const int ch = state.cell_child4[sel * 4 + k];
                if (ch >= 0 && !enqueue(ch)) return;
            }
        } else if (b.bordering(act, sel)) {
            ++d1_count;
        } else if (sel != act && b.is_leaf(act)) {
            ++d2_count;
        }
    }
}

// Count-scratch layout is four planes of (max_cells+1) ints so each plane can
// feed CUB ExclusiveSum without a host-side category loop.
__global__ void nmm_wsprl_count_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        int target_level, int traversal_capacity, int* count_scratch,
        int* overflow, int* status) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    const int stride = max_cells + 1;
    if (cell >= max_cells) return;
    if (cell >= cell_count || state.cell_level[cell] != target_level) {
        count_scratch[0 * stride + cell] = 0;
        count_scratch[1 * stride + cell] = 0;
        count_scratch[2 * stride + cell] = 0;
        count_scratch[3 * stride + cell] = 0;
        return;
    }
    int ic = 0, d1 = 0, d2 = 0, mc = 0;
    nmm_wsprl_counts_for_target(state, cell, traversal_capacity,
                                ic, d1, d2, mc, overflow, status);
    count_scratch[0 * stride + cell] = ic;
    count_scratch[1 * stride + cell] = d1;
    count_scratch[2 * stride + cell] = d2;
    count_scratch[3 * stride + cell] = mc;
}

// Parent seeds come from snapshots: after ExclusiveSum, begin[parent+1] is not
// necessarily the parent's list end.  Output bases still use scanned begin[].
__global__ void nmm_wsprl_scatter_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        int target_level, int traversal_capacity, const int* count_scratch,
        const int* I_snapshot, const int* D1_snapshot,
        int* overflow, int* status) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count || cell >= max_cells ||
            state.cell_level[cell] != target_level) return;
    const int stride = max_cells + 1;
    int io = state.I_begin[cell], d1o = state.D1_begin[cell];
    int d2o = state.D2_begin[cell], mo = state.M_begin[cell];
    int ie = io + count_scratch[0 * stride + cell];
    int d1e = d1o + count_scratch[1 * stride + cell];
    int d2e = d2o + count_scratch[2 * stride + cell];
    int me = mo + count_scratch[3 * stride + cell];
    DeviceTreeBuilder b{};
    b.state = state; b.p_status = status; b.p_overflow = overflow;
    b.traversal_queue_capacity = traversal_capacity;
    int e_q[kNmmParallelTraversalQueue];
    int e_head = 0, e_tail = 0;
    const int parent = state.cell_parent[cell];
    auto enqueue = [&](int value) {
        if (e_tail >= traversal_capacity || e_tail >= kNmmParallelTraversalQueue) {
            b.overflow(kNmmOverflowTraversalQueue); return false;
        }
        e_q[e_tail++] = value; return true;
    };
    if (parent < 0) {
        for (int k = 0; k < 4; ++k) {
            const int ch = state.cell_child4[cell * 4 + k];
            if (ch >= 0 && !enqueue(ch)) return;
        }
    } else {
        for (int x = D1_snapshot[parent]; x < D1_snapshot[parent + 1]; ++x)
            if (!enqueue(state.D1_nodes[x])) return;
        for (int x = I_snapshot[parent]; x < I_snapshot[parent + 1]; ++x)
            if (!enqueue(state.I_nodes[x])) return;
    }
    while (e_head < e_tail) {
        const int sel = e_q[e_head++];
        if (b.well_separated(cell, sel)) {
            if (mo >= me) { b.overflow(kNmmOverflowMList); return; }
            state.M_nodes[mo++] = sel;
        } else if (state.cell_level[cell] < state.cell_level[sel]) {
            if (io >= ie) { b.overflow(kNmmOverflowIList); return; }
            state.I_nodes[io++] = sel;
        } else if (!b.is_leaf(sel)) {
            for (int k = 0; k < 4; ++k) {
                const int ch = state.cell_child4[sel * 4 + k];
                if (ch >= 0 && !enqueue(ch)) return;
            }
        } else if (b.bordering(cell, sel)) {
            if (d1o >= d1e) { b.overflow(kNmmOverflowD1List); return; }
            state.D1_nodes[d1o++] = sel;
        } else if (sel != cell && b.is_leaf(cell)) {
            if (d2o >= d2e) { b.overflow(kNmmOverflowD2List); return; }
            state.D2_nodes[d2o++] = sel;
        }
    }
}

__global__ void nmm_wsprl_finalize_kernel(
        const int* cell_count_device, int max_cells, const int* I_begin,
        const int* D1_begin, const int* D2_begin, const int* M_begin,
        int* counters) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !counters) return;
    int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell_count < 0) cell_count = 0;
    if (cell_count > max_cells) cell_count = max_cells;
    counters[2] = I_begin[cell_count];
    counters[3] = D1_begin[cell_count];
    counters[4] = D2_begin[cell_count];
    counters[5] = M_begin[cell_count];
}

// The parallel WSPRL pass advances level by level because a target's
// traversal seed is the relation list of its parent.  A scan writes every
// begin entry, so completed levels must be restored from a session-owned
// snapshot before the next target level is evaluated.  The snapshot contains
// entries [0, max_cells); entry cell_count is only consumed by the final
// counter latch and is never a parent seed.
__global__ void nmm_wsprl_restore_completed_begin_kernel(
        const int* cell_level, const int* cell_count_device, int max_cells,
        int target_level,
        const int* snapshot, int* begin) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count) return;
    if (cell_level[cell] < target_level) begin[cell] = snapshot[cell];
}

__global__ void nmm_wsprl_restore_completed_begin_all_kernel(
        const int* cell_level, const int* cell_count_device, int max_cells,
        int target_level, const int* snapshots,
        int* I_begin, int* D1_begin, int* D2_begin, int* M_begin) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count) return;
    if (cell_level[cell] >= target_level) return;
    I_begin[cell] = snapshots[0 * max_cells + cell];
    D1_begin[cell] = snapshots[1 * max_cells + cell];
    D2_begin[cell] = snapshots[2 * max_cells + cell];
    M_begin[cell] = snapshots[3 * max_cells + cell];
}

__global__ void nmm_wsprl_snapshot_begin_kernel(
        const int* begin, int cell_count, int* snapshot) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count) return;
    snapshot[cell] = begin[cell];
}

__global__ void nmm_wsprl_snapshot_begin_all_kernel(
        const int* I_begin, const int* D1_begin, const int* D2_begin,
        const int* M_begin, int cell_count, int max_cells, int* snapshots) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cell_count || cell >= max_cells) return;
    snapshots[0 * max_cells + cell] = I_begin[cell];
    snapshots[1 * max_cells + cell] = D1_begin[cell];
    snapshots[2 * max_cells + cell] = D2_begin[cell];
    snapshots[3 * max_cells + cell] = M_begin[cell];
}

__global__ void nmm_wsprl_count_category_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        int target_level, int traversal_capacity, int category,
        int* counts, int* overflow, int* status) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count || cell >= max_cells ||
            state.cell_level[cell] != target_level) return;
    int ic = 0, d1 = 0, d2 = 0, mc = 0;
    nmm_wsprl_counts_for_target(state, cell, traversal_capacity,
                                ic, d1, d2, mc, overflow, status);
    counts[cell] = category == 0 ? ic : category == 1 ? d1 :
                   category == 2 ? d2 : mc;
}

__global__ void nmm_wsprl_scatter_category_kernel(
        FmmmNmmState state, const int* cell_count_device, int max_cells,
        int target_level, int traversal_capacity, int category,
        const int* counts, const int* I_snapshot, const int* D1_snapshot,
        const int* D2_snapshot, const int* M_snapshot,
        int* overflow, int* status) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = cell_count_device ? cell_count_device[0] : max_cells;
    if (cell >= cell_count || cell >= max_cells ||
            state.cell_level[cell] != target_level) return;

    DeviceTreeBuilder b{};
    b.state = state; b.p_status = status; b.p_overflow = overflow;
    b.traversal_queue_capacity = traversal_capacity;
    int e_q[kNmmParallelTraversalQueue];
    int e_head = 0, e_tail = 0;
    auto enqueue = [&](int value) {
        if (e_tail >= traversal_capacity || e_tail >= kNmmParallelTraversalQueue) {
            b.overflow(kNmmOverflowTraversalQueue); return false;
        }
        e_q[e_tail++] = value; return true;
    };
    const int parent = state.cell_parent[cell];
    if (parent < 0) {
        for (int k = 0; k < 4; ++k) {
            const int ch = state.cell_child4[cell * 4 + k];
            if (ch >= 0 && !enqueue(ch)) return;
        }
    } else {
        const int* d1_begin = D1_snapshot;
        const int* i_begin = I_snapshot;
        for (int x = d1_begin[parent]; x < d1_begin[parent + 1]; ++x)
            if (!enqueue(state.D1_nodes[x])) return;
        for (int x = i_begin[parent]; x < i_begin[parent + 1]; ++x)
            if (!enqueue(state.I_nodes[x])) return;
    }

    int out = 0;
    int base = category == 0 ? state.I_begin[cell] : category == 1 ? state.D1_begin[cell] :
               category == 2 ? state.D2_begin[cell] : state.M_begin[cell];
    const int limit = base + counts[cell];
    while (e_head < e_tail) {
        const int sel = e_q[e_head++];
        if (b.well_separated(cell, sel)) {
            if (category == 3) {
                if (base + out >= limit) { b.overflow(kNmmOverflowMList); return; }
                state.M_nodes[base + out++] = sel;
            }
        } else if (state.cell_level[cell] < state.cell_level[sel]) {
            if (category == 0) {
                if (base + out >= limit) { b.overflow(kNmmOverflowIList); return; }
                state.I_nodes[base + out++] = sel;
            }
        } else if (!b.is_leaf(sel)) {
            for (int k = 0; k < 4; ++k) {
                const int ch = state.cell_child4[sel * 4 + k];
                if (ch >= 0 && !enqueue(ch)) return;
            }
        } else if (b.bordering(cell, sel)) {
            if (category == 1) {
                if (base + out >= limit) { b.overflow(kNmmOverflowD1List); return; }
                state.D1_nodes[base + out++] = sel;
            }
        } else if (sel != cell && b.is_leaf(cell)) {
            if (category == 2) {
                if (base + out >= limit) { b.overflow(kNmmOverflowD2List); return; }
                state.D2_nodes[base + out++] = sel;
            }
        }
    }
}

} // namespace lcn::release
