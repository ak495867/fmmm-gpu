#include "fmmm_workspace.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <new>
#include <utility>

namespace lcn::release {

struct FmmmWorkspace::Impl {
    FmmmWorkspaceView view;
    void* block = nullptr;
    std::size_t block_size = 0;
};

FmmmPipelineStatus FmmmWorkspace::ensure_capacity(FmmmWorkspaceShape shape) {
    FmmmWorkspace& out = *this;
    if (shape.node_count == 0 || shape.edge_count == 0 || shape.level_count == 0) {
        return FmmmPipelineStatus::invalid_input;
    }
    
    std::size_t total_bytes = 0;
    bool overflow = false;
    auto add_bytes = [&total_bytes, &overflow](std::size_t count, std::size_t type_size) -> std::size_t {
        std::size_t align = 8;
        std::size_t pad = (align - (total_bytes % align)) % align;
        std::size_t offset = total_bytes + pad;
        if (offset < total_bytes) overflow = true;
        
        std::size_t size = count * type_size;
        if (count != 0 && size / count != type_size) overflow = true;
        
        total_bytes = offset + size;
        if (total_bytes < offset) overflow = true;
        
        return offset;
    };
    
    std::size_t off_pos_x = add_bytes(shape.node_count, sizeof(double));
    std::size_t off_pos_y = add_bytes(shape.node_count, sizeof(double));
    std::size_t off_force_x = add_bytes(shape.node_count, sizeof(double));
    std::size_t off_force_y = add_bytes(shape.node_count, sizeof(double));
    std::size_t off_scratch_d = add_bytes(shape.max_aux_entries, sizeof(double));
    std::size_t off_edge_src = add_bytes(shape.max_level_edges, sizeof(int));
    std::size_t off_edge_tgt = add_bytes(shape.max_level_edges, sizeof(int));
    std::size_t off_parent = add_bytes(shape.max_level_nodes, sizeof(int));
    std::size_t off_node_type = add_bytes(shape.max_level_nodes, sizeof(int));
    std::size_t off_ded_sun = add_bytes(shape.max_level_nodes, sizeof(int));
    std::size_t off_low_adj_b = add_bytes(shape.max_level_nodes + 1, sizeof(int));
    std::size_t off_low_adj_e = add_bytes(shape.low_adj_count, sizeof(int));
    std::size_t off_high_adj_b = add_bytes(shape.max_level_nodes + 1, sizeof(int));
    std::size_t off_high_adj_e = add_bytes(shape.high_adj_count, sizeof(int));
    std::size_t off_lambda_b = add_bytes(shape.max_level_nodes + 1, sizeof(int));
    std::size_t off_neigh_b = add_bytes(shape.max_level_nodes + 1, sizeof(int));
    std::size_t off_neigh_n = add_bytes(shape.neighbour_count, sizeof(int));
    std::size_t off_moon_b = add_bytes(shape.max_level_nodes + 1, sizeof(int));
    std::size_t off_moon_n = add_bytes(shape.moon_count, sizeof(int));
    std::size_t off_cell_c = add_bytes(shape.cell_capacity * 4, sizeof(int));
    std::size_t off_scratch_i = add_bytes(shape.max_aux_entries, sizeof(int));
    std::size_t off_edge_len = add_bytes(shape.max_level_edges, sizeof(double));
    std::size_t off_sun_dist = add_bytes(shape.max_level_nodes, sizeof(double));
    std::size_t off_lambda_v = add_bytes(shape.lambda_count, sizeof(double));
    std::size_t off_mass = add_bytes(shape.node_count, sizeof(double));
    std::size_t off_center_x = add_bytes(shape.cell_capacity, sizeof(double));
    std::size_t off_center_y = add_bytes(shape.cell_capacity, sizeof(double));
    
    if (overflow) {
        return FmmmPipelineStatus::invalid_input;
    }

    std::printf("[LCN-DIAGNOSTIC] FmmmWorkspace layout calculated: %zu bytes.\n", total_bytes);

    if (out.impl_ && out.impl_->block_size >= total_bytes) {
        // Reuse capacity
        out.impl_->view = FmmmWorkspaceView{};
        cudaMemset(out.impl_->block, 0, total_bytes);
    } else {
        if (out.impl_ && out.impl_->block) cudaFree(out.impl_->block);
        void* block = nullptr;
        if (cudaMalloc(&block, total_bytes) != cudaSuccess) return FmmmPipelineStatus::backend_error;
        if (cudaMemset(block, 0, total_bytes) != cudaSuccess) { cudaFree(block); return FmmmPipelineStatus::backend_error; }
        if (!out.impl_) out.impl_ = new Impl();
        out.impl_->block = block;
        out.impl_->block_size = total_bytes;
    }
    Impl* impl = out.impl_;
    void* block = impl->block;
    
    auto get_ptr = [block](std::size_t offset) {
        return reinterpret_cast<char*>(block) + offset;
    };
    
    impl->view.position_x = reinterpret_cast<double*>(get_ptr(off_pos_x));
    impl->view.position_y = reinterpret_cast<double*>(get_ptr(off_pos_y));
    impl->view.force_x = reinterpret_cast<double*>(get_ptr(off_force_x));
    impl->view.force_y = reinterpret_cast<double*>(get_ptr(off_force_y));
    impl->view.scratch_double = reinterpret_cast<double*>(get_ptr(off_scratch_d));
    impl->view.edge_source = reinterpret_cast<int*>(get_ptr(off_edge_src));
    impl->view.edge_target = reinterpret_cast<int*>(get_ptr(off_edge_tgt));
    impl->view.parent = reinterpret_cast<int*>(get_ptr(off_parent));
    impl->view.node_type = reinterpret_cast<int*>(get_ptr(off_node_type));
    impl->view.dedicated_sun = reinterpret_cast<int*>(get_ptr(off_ded_sun));
    impl->view.low_adj_begin = reinterpret_cast<int*>(get_ptr(off_low_adj_b));
    impl->view.low_adj_edge = reinterpret_cast<int*>(get_ptr(off_low_adj_e));
    impl->view.high_adj_begin = reinterpret_cast<int*>(get_ptr(off_high_adj_b));
    impl->view.high_adj_edge = reinterpret_cast<int*>(get_ptr(off_high_adj_e));
    impl->view.lambda_begin = reinterpret_cast<int*>(get_ptr(off_lambda_b));
    impl->view.neighbour_begin = reinterpret_cast<int*>(get_ptr(off_neigh_b));
    impl->view.neighbour_node = reinterpret_cast<int*>(get_ptr(off_neigh_n));
    impl->view.moon_begin = reinterpret_cast<int*>(get_ptr(off_moon_b));
    impl->view.moon_node = reinterpret_cast<int*>(get_ptr(off_moon_n));
    impl->view.cell_child4 = reinterpret_cast<int*>(get_ptr(off_cell_c));
    impl->view.scratch_int = reinterpret_cast<int*>(get_ptr(off_scratch_i));
    impl->view.edge_length = reinterpret_cast<double*>(get_ptr(off_edge_len));
    impl->view.sun_distance = reinterpret_cast<double*>(get_ptr(off_sun_dist));
    impl->view.lambda_value = reinterpret_cast<double*>(get_ptr(off_lambda_v));
    impl->view.mass = reinterpret_cast<double*>(get_ptr(off_mass));
    impl->view.center_x = reinterpret_cast<double*>(get_ptr(off_center_x));
    impl->view.center_y = reinterpret_cast<double*>(get_ptr(off_center_y));
    
    impl->view.node_count = shape.node_count;
    impl->view.edge_count = shape.edge_count;
    impl->view.level_count = shape.level_count;
    impl->view.cell_capacity = shape.cell_capacity;
    impl->view.max_level_nodes = shape.max_level_nodes;
    impl->view.max_level_edges = shape.max_level_edges;
    impl->view.max_aux_entries = shape.max_aux_entries;
    
    // out.release();
    // out.impl_ = impl; // already assigned
    return FmmmPipelineStatus::ok;
}

FmmmWorkspace::FmmmWorkspace(FmmmWorkspace&& other) noexcept {
    impl_ = other.impl_;
    other.impl_ = nullptr;
}

FmmmWorkspace& FmmmWorkspace::operator=(FmmmWorkspace&& other) noexcept {
    if (this != &other) {
        release();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

FmmmWorkspace::~FmmmWorkspace() {
    release();
}

FmmmWorkspaceView FmmmWorkspace::view() const {
    if (impl_) return impl_->view;
    return FmmmWorkspaceView{};
}

FmmmPipelineStatus FmmmWorkspace::reset() {
    if (impl_ && impl_->block) {
        if (cudaMemset(impl_->block, 0, impl_->block_size) != cudaSuccess) {
            return FmmmPipelineStatus::backend_error;
        }
    }
    return FmmmPipelineStatus::ok;
}

void FmmmWorkspace::release() noexcept {
    if (impl_) {
        if (impl_->block) {
            cudaFree(impl_->block);
        }
        delete impl_;
        impl_ = nullptr;
    }
}


FmmmPipelineStatus FmmmWorkspace::create(FmmmWorkspaceShape shape, FmmmWorkspace& out) {
    return out.ensure_capacity(shape);
}
} // namespace lcn::release
