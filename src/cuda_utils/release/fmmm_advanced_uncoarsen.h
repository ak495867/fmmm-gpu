#pragma once

#include "fmmm_workspace.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace lcn::oracle {

// Serial Advanced placement transition (level k+1 -> k). Device pointers use
// graph-local ordinals from the OGDF plan. Caps cover the Automatic corpus
// (n<=4704, e<=9200) for end-to-end coordinator routes; workspace-backed
// unbounded sizing remains a follow-up ENGINEERING_V1 migration.
constexpr int kFmmmAdvancedUncoarsenMaxNodes = 8192;
constexpr int kFmmmAdvancedUncoarsenMaxEdges = 16384;
constexpr int kFmmmAdvancedUncoarsenMaxAuxEntries = 32768;

struct FmmmAdvancedUncoarsenPlanView {
    int lower_node_count = 0;
    // Edge IDs in OGDF can have holes after coarsening/deletion. These are
    // exclusive ordinal spans, not `Graph::numberOfEdges()`; the arrays below
    // are indexed directly by the raw edge ordinal recorded in v3.
    int lower_edge_ordinal_span = 0;
    int high_node_count = 0;
    int high_edge_ordinal_span = 0;

    // Lower-level node SoA, each with lower_node_count entries.
    const int* lower_type = nullptr;
    const int* lower_dedicated_sun = nullptr;
    const double* lower_dedicated_sun_distance = nullptr;

    // High node h represents lower sun high_lower_node[h], and its position
    // is the completed coarse force state.
    const int* high_lower_node = nullptr;
    const double* high_position_x = nullptr;
    const double* high_position_y = nullptr;

    // Edge/order data. adjacency_begin has node_count + 1 entries; the
    // flattened values are edge ordinals and retain OGDF adjEntry order.
    const int* lower_edge_source = nullptr;
    const int* lower_edge_target = nullptr;
    const double* lower_edge_length = nullptr;
    const std::uint8_t* lower_edge_is_moon = nullptr;
    const int* lower_adjacency_begin = nullptr;
    const int* lower_adjacency_edge = nullptr;

    const int* high_edge_source = nullptr;
    const int* high_edge_target = nullptr;
    const std::uint8_t* high_edge_is_extra = nullptr;
    const int* high_adjacency_begin = nullptr;
    const int* high_adjacency_edge = nullptr;

    // Flat per-lower-node lists exported by the v3 oracle. begin has
    // lower_node_count + 1 entries and its last value bounds the flat array.
    const int* lambda_begin = nullptr;
    const double* lambda_values = nullptr;
    const int* neighbour_sun_begin = nullptr;
    const int* neighbour_sun_nodes = nullptr;
    const int* moon_begin = nullptr;
    const int* moon_nodes = nullptr;
};

struct FmmmAdvancedUncoarsenStateView {
    // Output lower-level double state; positions are overwritten in OGDF's
    // stage order. placed and angles make the intermediate state observable
    // for first-divergence diagnostics.
    double* position_x = nullptr;
    double* position_y = nullptr;
    std::uint8_t* placed = nullptr;
    double* angle_1 = nullptr;
    double* angle_2 = nullptr;
    double* box_length = nullptr;
    double* down_left_x = nullptr;
    double* down_left_y = nullptr;
    int* status = nullptr;
    int* consumed_draws = nullptr;
    void* nmm_workspace = nullptr;
};

struct FmmmAdvancedUncoarsenPrng {
    std::uint64_t seed = 0;
    // Total OGDF randomNumber draws to replay before Advanced placement.
    // For a single coarsest->fine step this equals 2 * coarsest_n (Auto2: 20).
    // For a later hierarchy level it also includes prior-level waggle draws.
    int consumed_draws = 0;
    // Of consumed_draws, how many used create_initial_placement_random's
    // randomNumber(0, BILLION). Remaining draws replay Multilevel waggle's
    // randomNumber(1, BILLION). -1 means all consumed_draws are placement-style
    // (standalone G3 / first uncoarsen callers).
    int initial_placement_draws = -1;
};

enum class FmmmAdvancedUncoarsenStatus : int {
    Ok = 0,
    Unsupported = 1,
    InvalidPlan = 2,
};

// Launches exactly one single-thread CUDA kernel. The caller owns uploads,
// downloads, and synchronization so this primitive never introduces a host
// layout/score fallback.
void fmmm_advanced_uncoarsen_reference(FmmmAdvancedUncoarsenPlanView plan,
                                       FmmmAdvancedUncoarsenStateView state,
                                       FmmmAdvancedUncoarsenPrng prng);

// Guard kernels run before the ordered placement kernel. They only initialize
// observable scratch state and validate read-only plan ranges; they never draw
// from PRNG or write positions.
void launch_advanced_uncoarsen_guards(FmmmAdvancedUncoarsenPlanView plan,
                                      FmmmAdvancedUncoarsenStateView state,
                                      FmmmAdvancedUncoarsenPrng prng,
                                      cudaStream_t stream = nullptr);

void launch_advanced_uncoarsen_transaction(
        FmmmAdvancedUncoarsenPlanView plan,
        FmmmAdvancedUncoarsenStateView state,
        FmmmAdvancedUncoarsenPrng prng,
        const lcn::release::FmmmWorkspaceView& workspace,
        cudaStream_t stream = nullptr);

} // namespace lcn::oracle

namespace lcn::release {
void launch_advanced_uncoarsen_kernel(
    lcn::oracle::FmmmAdvancedUncoarsenPlanView plan,
    lcn::oracle::FmmmAdvancedUncoarsenStateView state,
    lcn::oracle::FmmmAdvancedUncoarsenPrng prng,
    const lcn::release::FmmmWorkspaceView& workspace,
    cudaStream_t stream = nullptr);
} // namespace lcn::release

