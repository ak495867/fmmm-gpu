#pragma once

#include "fmmm_pipeline_contract.h"

#include <cstddef>

namespace lcn::release {

struct FmmmWorkspaceShape {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t level_count = 0;
    std::size_t cell_capacity = 0;
    std::size_t max_bytes = 0;
    std::size_t max_level_nodes = 0;
    std::size_t max_level_edges = 0;
    std::size_t max_aux_entries = 0;
    std::size_t low_adj_count = 0;
    std::size_t high_adj_count = 0;
    std::size_t lambda_count = 0;
    std::size_t neighbour_count = 0;
    std::size_t moon_count = 0;
};

struct FmmmWorkspaceView {
    double* position_x = nullptr;
    double* position_y = nullptr;
    double* force_x = nullptr;
    double* force_y = nullptr;
    double* scratch_double = nullptr;
    int* edge_source = nullptr;
    int* edge_target = nullptr;
    int* parent = nullptr;
    int* node_type = nullptr;
    int* dedicated_sun = nullptr;
    int* low_adj_begin = nullptr;
    int* low_adj_edge = nullptr;
    int* high_adj_begin = nullptr;
    int* high_adj_edge = nullptr;
    int* lambda_begin = nullptr;
    int* neighbour_begin = nullptr;
    int* neighbour_node = nullptr;
    int* moon_begin = nullptr;
    int* moon_node = nullptr;
    int* cell_child4 = nullptr;
    int* scratch_int = nullptr;
    double* edge_length = nullptr;
    double* sun_distance = nullptr;
    double* lambda_value = nullptr;
    double* mass = nullptr;
    double* center_x = nullptr;
    double* center_y = nullptr;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t level_count = 0;
    std::size_t cell_capacity = 0;
    std::size_t max_level_nodes = 0;
    std::size_t max_level_edges = 0;
    std::size_t max_aux_entries = 0;
};

class FmmmWorkspace {
public:
    static FmmmPipelineStatus create(FmmmWorkspaceShape shape, FmmmWorkspace& out);
    FmmmPipelineStatus ensure_capacity(FmmmWorkspaceShape shape);
    FmmmWorkspace() = default;
    FmmmWorkspace(FmmmWorkspace&&) noexcept;
    FmmmWorkspace& operator=(FmmmWorkspace&&) noexcept;
    FmmmWorkspace(const FmmmWorkspace&) = delete;
    FmmmWorkspace& operator=(const FmmmWorkspace&) = delete;
    ~FmmmWorkspace();

    FmmmWorkspaceView view() const;
    FmmmPipelineStatus reset();
    void release() noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace lcn::release
