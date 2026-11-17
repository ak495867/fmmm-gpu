#include "continuous_crossings.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lcn::oracle {
namespace {

constexpr unsigned long long kInt64MaxU = 0x7fffffffffffffffULL;

__device__ inline double orient_d(double ax, double ay, double bx, double by,
                                  double cx, double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline int sign_d(double value) { return (value > 0.0) - (value < 0.0); }

__device__ inline std::uint64_t pair_row_start(std::uint64_t row,
                                                std::uint64_t edge_count) {
    return row * (2ULL * edge_count - row - 1ULL) / 2ULL;
}

__device__ inline void pair_id_to_edges(std::uint64_t pair_id, int edge_count,
                                         int* first, int* second) {
    std::uint64_t lo = 0;
    std::uint64_t hi = static_cast<std::uint64_t>(edge_count - 1);
    const std::uint64_t edges = static_cast<std::uint64_t>(edge_count);
    while (lo + 1 < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2ULL;
        if (pair_row_start(mid, edges) <= pair_id) lo = mid;
        else hi = mid;
    }
    const std::uint64_t row = pair_row_start(hi, edges) <= pair_id ? hi : lo;
    const std::uint64_t offset = pair_id - pair_row_start(row, edges);
    *first = static_cast<int>(row);
    *second = static_cast<int>(row + 1ULL + offset);
}

__device__ inline bool valid_endpoint(int value, int node_count) {
    return value >= 0 && value < node_count;
}

__device__ inline void mark_invalid(DeviceContinuousScoreState* state) {
    atomicCAS(&state->status,
              static_cast<int>(DeviceContinuousScoreStatus::ok),
              static_cast<int>(DeviceContinuousScoreStatus::invalid_input));
}

__device__ inline void mark_overflow(DeviceContinuousScoreState* state) {
    atomicExch(&state->overflow, 1);
}

__device__ inline void checked_add_i64(std::int64_t* address,
                                       unsigned long long add,
                                       DeviceContinuousScoreState* state) {
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (true) {
        if (old > kInt64MaxU || add > kInt64MaxU - old) {
            mark_overflow(state);
            return;
        }
        const unsigned long long next = old + add;
        const unsigned long long observed = atomicCAS(bits, old, next);
        if (observed == old) return;
        old = observed;
    }
}

__device__ inline void checked_add_legacy(std::int64_t* address,
                                          unsigned long long add) {
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (true) {
        const unsigned long long next =
            old > kInt64MaxU || add > kInt64MaxU - old ? kInt64MaxU : old + add;
        const unsigned long long observed = atomicCAS(bits, old, next);
        if (observed == old) return;
        old = observed;
    }
}

__global__ void continuous_validate_edges_kernel(
    int node_count, int edge_count, const int* source, const int* target,
    DeviceContinuousScoreState* state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int edge = tid; edge < edge_count; edge += stride) {
        const int a = source[edge];
        const int b = target[edge];
        if (!valid_endpoint(a, node_count) || !valid_endpoint(b, node_count) || a == b) {
            mark_invalid(state);
        }
    }
}

__global__ void continuous_validate_positions_kernel(
    int node_count, const double* x, const double* y,
    DeviceContinuousScoreState* state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int node = tid; node < node_count; node += stride) {
        if (!isfinite(x[node]) || !isfinite(y[node])) mark_invalid(state);
    }
}

__global__ void continuous_validate_positions_fp32_kernel(
    int node_count, const float* x, const float* y,
    DeviceContinuousScoreState* state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int node = tid; node < node_count; node += stride) {
        if (!isfinite(x[node]) || !isfinite(y[node])) mark_invalid(state);
    }
}

__device__ inline float orient_f(float ax, float ay, float bx, float by,
                                 float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline int sign_f(float value) { return (value > 0.0F) - (value < 0.0F); }

__global__ void continuous_cross_fp32_kernel(
    int edge_count, const float* x, const float* y, const int* source,
    const int* target, const int* layer, int layers_enabled, int node_count,
    std::int64_t* counts, DeviceContinuousScoreState* state) {
    if (edge_count < 2 || state->status != static_cast<int>(DeviceContinuousScoreStatus::ok)) return;
    const std::uint64_t pair_count = static_cast<std::uint64_t>(edge_count) *
                                     static_cast<std::uint64_t>(edge_count - 1) / 2ULL;
    const std::uint64_t thread_id = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t pair_id = thread_id; pair_id < pair_count; pair_id += stride) {
        int i = 0, j = 0;
        pair_id_to_edges(pair_id, edge_count, &i, &j);
        const int ia = source[i], ib = target[i], ja = source[j], jb = target[j];
        if (!valid_endpoint(ia, node_count) || !valid_endpoint(ib, node_count) ||
            !valid_endpoint(ja, node_count) || !valid_endpoint(jb, node_count) ||
            ia == ib || ja == jb) {
            mark_invalid(state);
            continue;
        }
        if (ia == ja || ia == jb || ib == ja || ib == jb) continue;
        if (layers_enabled && layer[i] != layer[j]) continue;
        const float iax = x[ia], iay = y[ia], ibx = x[ib], iby = y[ib];
        const float jax = x[ja], jay = y[ja], jbx = x[jb], jby = y[jb];
        if (!isfinite(iax) || !isfinite(iay) || !isfinite(ibx) || !isfinite(iby) ||
            !isfinite(jax) || !isfinite(jay) || !isfinite(jbx) || !isfinite(jby)) {
            mark_invalid(state);
            continue;
        }
        const int s1 = sign_f(orient_f(iax, iay, ibx, iby, jax, jay));
        const int s2 = sign_f(orient_f(iax, iay, ibx, iby, jbx, jby));
        const int s3 = sign_f(orient_f(jax, jay, jbx, jby, iax, iay));
        const int s4 = sign_f(orient_f(jax, jay, jbx, jby, ibx, iby));
        if (s1 * s2 < 0 && s3 * s4 < 0) {
            checked_add_i64(counts + i, 1ULL, state);
            checked_add_i64(counts + j, 1ULL, state);
        }
    }
}

__global__ void continuous_cross_device_kernel(
    int edge_count, const double* x, const double* y, const int* source,
    const int* target, const int* layer, int layers_enabled, int node_count,
    std::int64_t* counts, DeviceContinuousScoreState* state) {
    if (edge_count < 2 || state->status != static_cast<int>(DeviceContinuousScoreStatus::ok)) return;
    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(edge_count) *
        static_cast<std::uint64_t>(edge_count - 1) / 2ULL;
    const std::uint64_t thread_id =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t pair_id = thread_id; pair_id < pair_count; pair_id += stride) {
        int i = 0, j = 0;
        pair_id_to_edges(pair_id, edge_count, &i, &j);
        const int ia = source[i], ib = target[i], ja = source[j], jb = target[j];
        if (!valid_endpoint(ia, node_count) || !valid_endpoint(ib, node_count) ||
            !valid_endpoint(ja, node_count) || !valid_endpoint(jb, node_count) ||
            ia == ib || ja == jb) {
            mark_invalid(state);
            continue;
        }
        if (ia == ja || ia == jb || ib == ja || ib == jb) continue;
        if (layers_enabled && layer[i] != layer[j]) continue;
        const double iax = x[ia], iay = y[ia], ibx = x[ib], iby = y[ib];
        const double jax = x[ja], jay = y[ja], jbx = x[jb], jby = y[jb];
        if (!isfinite(iax) || !isfinite(iay) || !isfinite(ibx) || !isfinite(iby) ||
            !isfinite(jax) || !isfinite(jay) || !isfinite(jbx) || !isfinite(jby)) {
            mark_invalid(state);
            continue;
        }
        const int s1 = sign_d(orient_d(iax, iay, ibx, iby, jax, jay));
        const int s2 = sign_d(orient_d(iax, iay, ibx, iby, jbx, jby));
        const int s3 = sign_d(orient_d(jax, jay, jbx, jby, iax, iay));
        const int s4 = sign_d(orient_d(jax, jay, jbx, jby, ibx, iby));
        if (s1 * s2 < 0 && s3 * s4 < 0) {
            checked_add_i64(counts + i, 1ULL, state);
            checked_add_i64(counts + j, 1ULL, state);
        }
    }
}

__global__ void continuous_cross_legacy_kernel(
    int edge_count, const double* x, const double* y, const int* source,
    const int* target, const int* layer, int layers_enabled, int node_count,
    std::int64_t* counts) {
    if (edge_count < 2) return;
    const std::uint64_t pair_count =
        static_cast<std::uint64_t>(edge_count) *
        static_cast<std::uint64_t>(edge_count - 1) / 2ULL;
    const std::uint64_t thread_id =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t pair_id = thread_id; pair_id < pair_count; pair_id += stride) {
        int i = 0, j = 0;
        pair_id_to_edges(pair_id, edge_count, &i, &j);
        const int ia = source[i], ib = target[i], ja = source[j], jb = target[j];
        if (!valid_endpoint(ia, node_count) || !valid_endpoint(ib, node_count) ||
            !valid_endpoint(ja, node_count) || !valid_endpoint(jb, node_count) ||
            ia == ib || ja == jb || ia == ja || ia == jb || ib == ja || ib == jb) continue;
        if (layers_enabled && layer[i] != layer[j]) continue;
        const double iax = x[ia], iay = y[ia], ibx = x[ib], iby = y[ib];
        const double jax = x[ja], jay = y[ja], jbx = x[jb], jby = y[jb];
        if (!isfinite(iax) || !isfinite(iay) || !isfinite(ibx) || !isfinite(iby) ||
            !isfinite(jax) || !isfinite(jay) || !isfinite(jbx) || !isfinite(jby)) continue;
        const int s1 = sign_d(orient_d(iax, iay, ibx, iby, jax, jay));
        const int s2 = sign_d(orient_d(iax, iay, ibx, iby, jbx, jby));
        const int s3 = sign_d(orient_d(jax, jay, jbx, jby, iax, iay));
        const int s4 = sign_d(orient_d(jax, jay, jbx, jby, ibx, iby));
        if (s1 * s2 < 0 && s3 * s4 < 0) {
            checked_add_legacy(counts + i, 1ULL);
            checked_add_legacy(counts + j, 1ULL);
        }
    }
}

__global__ void continuous_score_init_kernel(
    int edge_count, DeviceContinuousScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    state->objective.K = 0;
    state->objective.nK = 0;
    state->objective.Phi = 0;
    state->objective.C = 0;
    state->objective.valid = 0;
    state->objective.exact_valid = 0;
    state->objective.status = static_cast<int>(ProposalStatus::initial);
    state->objective.fallback_reason = static_cast<int>(FallbackReason::none);
    state->admission.admitted = 0;
    state->admission.status = static_cast<int>(DeviceContinuousScoreStatus::ok);
    state->admission.reason = static_cast<int>(DeviceContinuousFallbackReason::none);
    state->status = static_cast<int>(DeviceContinuousScoreStatus::ok);
    state->fallback_reason = static_cast<int>(DeviceContinuousFallbackReason::none);
    state->overflow = 0;
    state->edge_count = edge_count;
}

__global__ void continuous_score_max_kernel(
    int edge_count, const std::int64_t* counts, DeviceContinuousScoreState* state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int edge = tid; edge < edge_count; edge += stride) {
        const std::int64_t value = counts[edge];
        if (value < 0) mark_overflow(state);
        else atomicMax(reinterpret_cast<unsigned long long*>(&state->objective.K),
                       static_cast<unsigned long long>(value));
    }
}

__global__ void continuous_score_accumulate_kernel(
    int edge_count, const std::int64_t* counts, DeviceContinuousScoreState* state) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int edge = tid; edge < edge_count; edge += stride) {
        const std::int64_t value = counts[edge];
        if (value < 0) {
            mark_overflow(state);
            continue;
        }
        const unsigned long long uvalue = static_cast<unsigned long long>(value);
        if (value == state->objective.K) checked_add_i64(&state->objective.nK, 1ULL, state);
        if (uvalue != 0 && uvalue > kInt64MaxU / uvalue) mark_overflow(state);
        else checked_add_i64(&state->objective.Phi, uvalue * uvalue, state);
        checked_add_i64(&state->objective.C, uvalue, state);
    }
}

__global__ void continuous_score_finalize_kernel(DeviceContinuousScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (state->status == static_cast<int>(DeviceContinuousScoreStatus::invalid_input)) {
        state->objective.valid = 0;
        state->objective.exact_valid = 0;
        state->objective.status = static_cast<int>(ProposalStatus::rejected_invalid);
        state->objective.fallback_reason = static_cast<int>(FallbackReason::none);
        state->fallback_reason = static_cast<int>(DeviceContinuousFallbackReason::invalid_input);
        state->admission.admitted = 0;
        state->admission.status = state->status;
        state->admission.reason = state->fallback_reason;
        return;
    }
    if (state->overflow != 0) {
        state->status = static_cast<int>(DeviceContinuousScoreStatus::overflow);
        state->fallback_reason = static_cast<int>(DeviceContinuousFallbackReason::overflow);
        state->objective.valid = 0;
        state->objective.exact_valid = 0;
        state->objective.status = static_cast<int>(ProposalStatus::rejected_overflow);
        state->objective.fallback_reason = static_cast<int>(FallbackReason::overflow);
        state->admission.admitted = 0;
        state->admission.status = state->status;
        state->admission.reason = state->fallback_reason;
        return;
    }
    state->objective.C /= 2;
    state->objective.valid = 1;
    state->objective.exact_valid = 1;
    state->objective.status = static_cast<int>(ProposalStatus::initial);
    state->objective.fallback_reason = static_cast<int>(FallbackReason::none);
    state->admission.admitted = 1;
    state->admission.status = static_cast<int>(DeviceContinuousScoreStatus::ok);
    state->admission.reason = static_cast<int>(DeviceContinuousFallbackReason::none);
}

__global__ void legacy_score_init_kernel(ContinuousObjective* objective) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        objective->K = objective->nK = objective->Phi = objective->C = 0;
    }
}

__global__ void legacy_score_reduce_kernel(
    int edge_count, const std::int64_t* counts, ContinuousObjective* objective) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int edge = tid; edge < edge_count; edge += stride) {
        const std::int64_t value = counts[edge];
        if (value >= 0) atomicMax(reinterpret_cast<unsigned long long*>(&objective->K),
                                  static_cast<unsigned long long>(value));
    }
}

__global__ void legacy_score_accumulate_kernel(
    int edge_count, const std::int64_t* counts, ContinuousObjective* objective) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int edge = tid; edge < edge_count; edge += stride) {
        const std::int64_t value = counts[edge];
        if (value < 0) continue;
        const unsigned long long uvalue = static_cast<unsigned long long>(value);
        if (value == objective->K) checked_add_legacy(&objective->nK, 1ULL);
        const unsigned long long phi =
            uvalue != 0 && uvalue > kInt64MaxU / uvalue ? kInt64MaxU : uvalue * uvalue;
        checked_add_legacy(&objective->Phi, phi);
        checked_add_legacy(&objective->C, uvalue);
    }
}

__global__ void legacy_score_finalize_kernel(ContinuousObjective* objective) {
    if (blockIdx.x == 0 && threadIdx.x == 0) objective->C /= 2;
}

__global__ void continuous_publication_kernel(
    DeviceFmmmPublicationInput input, DeviceFmmmPublicationRecord* publication) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const DeviceContinuousScoreState& score = *input.score_state;
    publication->status = score.status;
    publication->admitted = score.admission.admitted;
    publication->fallback_required = score.status != static_cast<int>(DeviceContinuousScoreStatus::ok);
    publication->fallback_reason = score.fallback_reason;
    publication->node_count = input.node_count;
    publication->edge_count = input.edge_count;
    publication->graph_epoch = input.graph_epoch;
    publication->tree_epoch = input.tree_epoch;
    publication->publication_epoch = input.publication_epoch;
    publication->objective = score.objective;
    publication->position_x = input.position_x;
    publication->position_y = input.position_y;
    publication->edge_crossings = input.edge_crossings;
}

} // namespace

DeviceContinuousScoreStatus fmmm_enqueue_device_continuous_score(
    const DeviceContinuousScoreInput& input, cudaStream_t stream) {
    if (input.node_count < 0 || input.edge_count < 0 || input.state == nullptr ||
        (input.node_count > 0 && (input.position_x == nullptr || input.position_y == nullptr)) ||
        (input.edge_count > 0 && (input.edge_crossings == nullptr ||
                                  input.edge_source == nullptr || input.edge_target == nullptr)) ||
        (input.layers_enabled != 0 && input.edge_count > 0 && input.edge_layer == nullptr)) {
        return DeviceContinuousScoreStatus::invalid_input;
    }
    if (input.edge_count > 0) {
        cudaError_t error = cudaMemsetAsync(
            input.edge_crossings, 0,
            static_cast<std::size_t>(input.edge_count) * sizeof(std::int64_t), stream);
        if (error != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    }
    continuous_score_init_kernel<<<1, 1, 0, stream>>>(input.edge_count, input.state);
    if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    if (input.node_count > 0) {
        constexpr int threads = 256;
        const int blocks = std::max(1, std::min(65535,
            (input.node_count + threads - 1) / threads));
        continuous_validate_positions_kernel<<<blocks, threads, 0, stream>>>(
            input.node_count, input.position_x, input.position_y, input.state);
        if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    }
    if (input.edge_count > 0) {
        constexpr int threads = 256;
        const int blocks = std::max(1, std::min(65535,
            (input.edge_count + threads - 1) / threads));
        continuous_validate_edges_kernel<<<blocks, threads, 0, stream>>>(
            input.node_count, input.edge_count, input.edge_source, input.edge_target, input.state);
        if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    }
    if (input.edge_count > 1) {
        constexpr int threads = 256;
        const std::uint64_t pair_count =
            static_cast<std::uint64_t>(input.edge_count) *
            static_cast<std::uint64_t>(input.edge_count - 1) / 2ULL;
        const std::uint64_t needed_blocks =
            (pair_count + static_cast<std::uint64_t>(threads) - 1ULL) /
            static_cast<std::uint64_t>(threads);
        const int blocks = static_cast<int>(std::min<std::uint64_t>(needed_blocks, 65535ULL));
        continuous_cross_device_kernel<<<blocks, threads, 0, stream>>>(
            input.edge_count, input.position_x, input.position_y,
            input.edge_source, input.edge_target, input.edge_layer,
            input.layers_enabled, input.node_count, input.edge_crossings, input.state);
        if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    }
    if (input.edge_count > 0) {
        constexpr int threads = 256;
        const int blocks = std::max(1, std::min(65535,
            (input.edge_count + threads - 1) / threads));
        continuous_score_max_kernel<<<blocks, threads, 0, stream>>>(
            input.edge_count, input.edge_crossings, input.state);
        if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
        continuous_score_accumulate_kernel<<<blocks, threads, 0, stream>>>(
            input.edge_count, input.edge_crossings, input.state);
        if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    }
    continuous_score_finalize_kernel<<<1, 1, 0, stream>>>(input.state);
    if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    return DeviceContinuousScoreStatus::ok;
}

DeviceContinuousScoreStatus fmmm_enqueue_device_publication(
    const DeviceFmmmPublicationInput& input,
    DeviceFmmmPublicationRecord* device_publication,
    cudaStream_t stream) {
    if (device_publication == nullptr || input.score_state == nullptr ||
        input.node_count < 0 || input.edge_count < 0 ||
        (input.node_count > 0 && (input.position_x == nullptr || input.position_y == nullptr)) ||
        (input.edge_count > 0 && input.edge_crossings == nullptr)) {
        return DeviceContinuousScoreStatus::invalid_input;
    }
    continuous_publication_kernel<<<1, 1, 0, stream>>>(input, device_publication);
    if (cudaGetLastError() != cudaSuccess) return DeviceContinuousScoreStatus::backend_error;
    return DeviceContinuousScoreStatus::ok;
}

cudaError_t evaluate_continuous_crossings_async(
    int node_count, int edge_count, const double* position_x, const double* position_y,
    const int* edge_source, const int* edge_target, const int* edge_layer, bool layers_enabled,
    std::int64_t* d_edge_counts, ContinuousObjective* d_objective, cudaStream_t stream) {
    if (d_objective == nullptr || node_count < 0 || edge_count < 0 ||
        (node_count > 0 && (position_x == nullptr || position_y == nullptr)) ||
        (edge_count > 0 && (edge_source == nullptr || edge_target == nullptr || d_edge_counts == nullptr)) ||
        (layers_enabled && edge_count > 0 && edge_layer == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (edge_count > 0) {
        cudaError_t error = cudaMemsetAsync(
            d_edge_counts, 0, static_cast<std::size_t>(edge_count) * sizeof(std::int64_t), stream);
        if (error != cudaSuccess) return error;
        if (edge_count > 1) {
            constexpr int threads = 256;
            const std::uint64_t pair_count =
                static_cast<std::uint64_t>(edge_count) *
                static_cast<std::uint64_t>(edge_count - 1) / 2ULL;
            const int blocks = static_cast<int>(std::min<std::uint64_t>(
                (pair_count + threads - 1ULL) / threads, 65535ULL));
            continuous_cross_legacy_kernel<<<blocks, threads, 0, stream>>>(
                edge_count, position_x, position_y, edge_source, edge_target,
                edge_layer, layers_enabled ? 1 : 0, node_count, d_edge_counts);
            error = cudaGetLastError();
            if (error != cudaSuccess) return error;
        }
    }
    legacy_score_init_kernel<<<1, 1, 0, stream>>>(d_objective);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess || edge_count <= 0) return error;
    constexpr int threads = 256;
    const int blocks = std::max(1, std::min(65535, (edge_count + threads - 1) / threads));
    legacy_score_reduce_kernel<<<blocks, threads, 0, stream>>>(edge_count, d_edge_counts, d_objective);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    legacy_score_accumulate_kernel<<<blocks, threads, 0, stream>>>(edge_count, d_edge_counts, d_objective);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    legacy_score_finalize_kernel<<<1, 1, 0, stream>>>(d_objective);
    return cudaGetLastError();
}

ScoreResult evaluate_continuous_crossings(int node_count, int edge_count, const double* position_x,
                                          const double* position_y, const int* edge_source,
                                          const int* edge_target, const int* edge_layer,
                                          bool layers_enabled) {
    ScoreResult result;
    result.proposal_id = "continuous_v1";
    result.status = ProposalStatus::initial;
    if (node_count == 0 && edge_count == 0) {
        result.valid = true;
        result.exact_valid = true;
        return result;
    }
    if (node_count < 0 || edge_count < 0 ||
        (node_count > 0 && (position_x == nullptr || position_y == nullptr))) {
        result.valid = false;
        result.status = ProposalStatus::rejected_invalid;
        result.fallback_reason = "null positions";
        return result;
    }
    if (edge_count > 0 && (edge_source == nullptr || edge_target == nullptr)) {
        result.valid = false;
        result.status = ProposalStatus::rejected_invalid;
        result.fallback_reason = "null edges";
        return result;
    }

    DeviceBuffer<int> d_layer_owned(edge_count > 0 && edge_layer == nullptr
                                        ? static_cast<std::size_t>(edge_count) : 0);
    const int* layer_ptr = edge_layer;
    if (edge_count > 0 && layer_ptr == nullptr) {
        cudaMemset(d_layer_owned.get(), 0, static_cast<std::size_t>(edge_count) * sizeof(int));
        layer_ptr = d_layer_owned.get();
    }
    DeviceBuffer<std::int64_t> d_counts(static_cast<std::size_t>(std::max(edge_count, 0)));
    DeviceBuffer<DeviceContinuousScoreState> d_state(1);
    const DeviceContinuousScoreInput input{
        node_count, edge_count, position_x, position_y, edge_source, edge_target,
        layer_ptr, layers_enabled ? 1 : 0, d_counts.get(), d_state.get()};
    const DeviceContinuousScoreStatus status =
        fmmm_enqueue_device_continuous_score(input, nullptr);
    if (status != DeviceContinuousScoreStatus::ok || cudaDeviceSynchronize() != cudaSuccess) {
        result.valid = false;
        result.status = ProposalStatus::rejected_invalid;
        result.fallback_reason = "continuous score enqueue error";
        return result;
    }

    DeviceContinuousScoreState host_state{};
    d_state.download(&host_state, 1);
    result.edge_crossings.resize(static_cast<std::size_t>(std::max(edge_count, 0)));
    if (edge_count > 0) d_counts.download(result.edge_crossings.data(), result.edge_crossings.size());
    result.K = host_state.objective.K;
    result.nK = host_state.objective.nK;
    result.Phi = host_state.objective.Phi;
    result.C = host_state.objective.C;
    result.valid = host_state.objective.valid != 0;
    result.exact_valid = host_state.objective.exact_valid != 0;
    result.status = static_cast<ProposalStatus>(host_state.objective.status);
    result.fallback_required = host_state.objective.fallback_reason ==
                               static_cast<int>(FallbackReason::overflow);
    result.fallback_reason = result.fallback_required ? "overflow" : "continuous_v1";
    return result;
}

ScoreResult evaluate_continuous_crossings_fp32(int node_count, int edge_count,
                                               const float* position_x,
                                               const float* position_y,
                                               const int* edge_source,
                                               const int* edge_target,
                                               const int* edge_layer,
                                               bool layers_enabled) {
    ScoreResult result;
    result.proposal_id = "continuous_v1_fp32";
    result.status = ProposalStatus::initial;
    if (node_count < 0 || edge_count < 0 ||
        (node_count > 0 && (position_x == nullptr || position_y == nullptr)) ||
        (edge_count > 0 && (edge_source == nullptr || edge_target == nullptr)) ||
        (layers_enabled && edge_count > 0 && edge_layer == nullptr)) {
        result.valid = false;
        result.status = ProposalStatus::rejected_invalid;
        result.fallback_reason = "invalid fp32 recount input";
        return result;
    }

    DeviceBuffer<float> d_x(static_cast<std::size_t>(node_count));
    DeviceBuffer<float> d_y(static_cast<std::size_t>(node_count));
    DeviceBuffer<int> d_source(static_cast<std::size_t>(edge_count));
    DeviceBuffer<int> d_target(static_cast<std::size_t>(edge_count));
    DeviceBuffer<int> d_layer(static_cast<std::size_t>(edge_count));
    DeviceBuffer<std::int64_t> d_counts(static_cast<std::size_t>(edge_count));
    DeviceBuffer<DeviceContinuousScoreState> d_state(1);
    d_x.upload(position_x, static_cast<std::size_t>(node_count));
    d_y.upload(position_y, static_cast<std::size_t>(node_count));
    d_source.upload(edge_source, static_cast<std::size_t>(edge_count));
    d_target.upload(edge_target, static_cast<std::size_t>(edge_count));
    // The FP32 recount's initialization kernel owns only the objective state;
    // edge counts are produced by pair accumulation and therefore must be
    // explicitly zeroed before the first atomic add.  Without this clear,
    // repeated calls can publish stale 64-bit values (for example
    // 0x600000005) instead of an independent recount.
    if (edge_count > 0) {
        cuda_check(cudaMemset(
                       d_counts.get(), 0,
                       static_cast<std::size_t>(edge_count) * sizeof(std::int64_t)),
                   "cudaMemset fp32 edge counts");
    }
    if (edge_count > 0) {
        if (edge_layer != nullptr) d_layer.upload(edge_layer, static_cast<std::size_t>(edge_count));
        else cuda_check(cudaMemset(d_layer.get(), 0, static_cast<std::size_t>(edge_count) * sizeof(int)),
                        "cudaMemset fp32 layers");
    }

    continuous_score_init_kernel<<<1, 1>>>(edge_count, d_state.get());
    cuda_check(cudaGetLastError(), "fp32 score initialization");
    constexpr int threads = 256;
    const int node_blocks = std::max(1, std::min(65535, (node_count + threads - 1) / threads));
    const int edge_blocks = std::max(1, std::min(65535, (edge_count + threads - 1) / threads));
    if (node_count > 0) {
        continuous_validate_positions_fp32_kernel<<<node_blocks, threads>>>(
            node_count, d_x.get(), d_y.get(), d_state.get());
        cuda_check(cudaGetLastError(), "fp32 position validation");
    }
    if (edge_count > 0) {
        continuous_validate_edges_kernel<<<edge_blocks, threads>>>(
            node_count, edge_count, d_source.get(), d_target.get(), d_state.get());
        cuda_check(cudaGetLastError(), "fp32 edge validation");
        if (edge_count > 1) {
            const std::uint64_t pair_count = static_cast<std::uint64_t>(edge_count) *
                                             static_cast<std::uint64_t>(edge_count - 1) / 2ULL;
            const int pair_blocks = static_cast<int>(std::min<std::uint64_t>(
                std::max<std::uint64_t>(1, (pair_count + threads - 1ULL) / threads), 65535ULL));
            continuous_cross_fp32_kernel<<<pair_blocks, threads>>>(
                edge_count, d_x.get(), d_y.get(), d_source.get(), d_target.get(),
                d_layer.get(), layers_enabled ? 1 : 0, node_count, d_counts.get(), d_state.get());
            cuda_check(cudaGetLastError(), "fp32 continuous crossing recount");
        }
        continuous_score_max_kernel<<<edge_blocks, threads>>>(edge_count, d_counts.get(), d_state.get());
        cuda_check(cudaGetLastError(), "fp32 score maximum");
        continuous_score_accumulate_kernel<<<edge_blocks, threads>>>(edge_count, d_counts.get(), d_state.get());
        cuda_check(cudaGetLastError(), "fp32 score accumulation");
    }
    continuous_score_finalize_kernel<<<1, 1>>>(d_state.get());
    cuda_check(cudaGetLastError(), "fp32 score finalization");
    cuda_check(cudaDeviceSynchronize(), "fp32 continuous recount synchronization");

    DeviceContinuousScoreState state{};
    d_state.download(&state, 1);
    result.edge_crossings.resize(static_cast<std::size_t>(edge_count));
    d_counts.download(result.edge_crossings.data(), result.edge_crossings.size());
    result.K = state.objective.K;
    result.nK = state.objective.nK;
    result.Phi = state.objective.Phi;
    result.C = state.objective.C;
    result.valid = state.objective.valid != 0;
    result.exact_valid = state.objective.exact_valid != 0;
    result.status = static_cast<ProposalStatus>(state.objective.status);
    result.fallback_required = state.fallback_reason ==
                               static_cast<int>(DeviceContinuousFallbackReason::overflow);
    result.fallback_reason = result.fallback_required ? "overflow" : "continuous_v1_fp32";
    return result;
}

} // namespace lcn::oracle
