#pragma once

#include "full_square.h"

#include <cstdint>

namespace lcn::oracle {

enum class DeviceContinuousScoreStatus : std::int32_t {
    ok = 0,
    invalid_input = 1,
    backend_error = 2,
    overflow = 3,
};

enum class DeviceContinuousFallbackReason : std::int32_t {
    none = 0,
    invalid_input = 1,
    overflow = 2,
};

struct DeviceAdmissionState {
    std::int32_t admitted = 0;
    std::int32_t status = static_cast<std::int32_t>(DeviceContinuousScoreStatus::invalid_input);
    std::int32_t reason = static_cast<std::int32_t>(DeviceContinuousFallbackReason::invalid_input);
    std::int32_t reserved = 0;
};

struct DeviceContinuousScoreState {
    DeviceObjective objective{};
    DeviceAdmissionState admission{};
    std::int32_t status = static_cast<std::int32_t>(DeviceContinuousScoreStatus::invalid_input);
    std::int32_t fallback_reason = static_cast<std::int32_t>(DeviceContinuousFallbackReason::invalid_input);
    std::int32_t overflow = 0;
    std::int32_t edge_count = 0;
};

struct DeviceContinuousScoreInput {
    std::int32_t node_count = 0;
    std::int32_t edge_count = 0;
    const double* position_x = nullptr;
    const double* position_y = nullptr;
    const std::int32_t* edge_source = nullptr;
    const std::int32_t* edge_target = nullptr;
    const std::int32_t* edge_layer = nullptr;
    std::int32_t layers_enabled = 0;
    std::int64_t* edge_crossings = nullptr;
    DeviceContinuousScoreState* state = nullptr;
};

struct DeviceFmmmPublicationRecord {
    std::int32_t status = static_cast<std::int32_t>(DeviceContinuousScoreStatus::invalid_input);
    std::int32_t admitted = 0;
    std::int32_t fallback_required = 1;
    std::int32_t fallback_reason = static_cast<std::int32_t>(DeviceContinuousFallbackReason::invalid_input);
    std::int32_t node_count = 0;
    std::int32_t edge_count = 0;
    std::uint64_t graph_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t publication_epoch = 0;
    DeviceObjective objective{};
    const double* position_x = nullptr;
    const double* position_y = nullptr;
    const std::int64_t* edge_crossings = nullptr;
};

struct DeviceFmmmPublicationInput {
    std::int32_t node_count = 0;
    std::int32_t edge_count = 0;
    const double* position_x = nullptr;
    const double* position_y = nullptr;
    const std::int64_t* edge_crossings = nullptr;
    const DeviceContinuousScoreState* score_state = nullptr;
    std::uint64_t graph_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t publication_epoch = 0;
};

DeviceContinuousScoreStatus fmmm_enqueue_device_continuous_score(
    const DeviceContinuousScoreInput& input,
    cudaStream_t stream = nullptr);

DeviceContinuousScoreStatus fmmm_enqueue_device_publication(
    const DeviceFmmmPublicationInput& input,
    DeviceFmmmPublicationRecord* device_publication,
    cudaStream_t stream = nullptr);

// Device-side K/nK/Phi/C aggregate. Layout matches the first four fields of
// DeviceObjective and search_v1::Objective64 (four contiguous int64 values).
struct ContinuousObjective {
    std::int64_t K = 0;
    std::int64_t nK = 0;
    std::int64_t Phi = 0;
    std::int64_t C = 0;
};

static_assert(sizeof(ContinuousObjective) == 32, "ContinuousObjective is four int64 fields");
static_assert(alignof(ContinuousObjective) == 8, "ContinuousObjective align");

// Continuous proper-segment crossing count. Does not mutate DeviceGraph.
// K/nK/Phi/C use the same aggregates as FULL_SQUARE.
// Host convenience wrapper: runs the device path on the default stream, then
// downloads counts + objective into ScoreResult.
ScoreResult evaluate_continuous_crossings(
    int node_count,
    int edge_count,
    const double* position_x,
    const double* position_y,
    const int* edge_source,
    const int* edge_target,
    const int* edge_layer,
    bool layers_enabled);

// Independent FP32 continuous oracle used by the G0 evidence lane.  This is
// deliberately separate from the production session and narrows coordinates
// to float before the independent CUDA upload.
ScoreResult evaluate_continuous_crossings_fp32(
    int node_count,
    int edge_count,
    const float* position_x,
    const float* position_y,
    const int* edge_source,
    const int* edge_target,
    const int* edge_layer,
    bool layers_enabled);

// Stream-ordered device-output path. Writes edge counts and ContinuousObjective
// directly on device; does not D2H. nullptr stream selects the default stream.
// edge_layer may be null when layers_enabled is false (kernel does not read it).
// d_edge_counts may be null only when edge_count == 0. d_objective is required.
cudaError_t evaluate_continuous_crossings_async(
    int node_count,
    int edge_count,
    const double* position_x,
    const double* position_y,
    const int* edge_source,
    const int* edge_target,
    const int* edge_layer,
    bool layers_enabled,
    std::int64_t* d_edge_counts,
    ContinuousObjective* d_objective,
    cudaStream_t stream);

} // namespace lcn::oracle
