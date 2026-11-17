#include "fmmm_session.h"
#include "fmmm_hierarchy_contract.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>

namespace lcn::release {

namespace {

thread_local const char* g_last_create_error = "ok";
constexpr std::uint32_t kSessionOwnerId = 1;
constexpr int kPhaseCount = kFmmmSemanticPhaseCount;
constexpr std::size_t kTransferLedgerCapacity = 1024;

enum SessionPhase : int {
    kPhaseUpload = static_cast<int>(FmmmSemanticPhase::upload),
    kPhaseNormalize = static_cast<int>(FmmmSemanticPhase::normalize),
    kPhaseComponents = static_cast<int>(FmmmSemanticPhase::components),
    kPhaseHierarchy = static_cast<int>(FmmmSemanticPhase::hierarchy),
    kPhaseNmm = static_cast<int>(FmmmSemanticPhase::nmm),
    kPhasePlacement = static_cast<int>(FmmmSemanticPhase::placement),
    kPhaseNormalForce = static_cast<int>(FmmmSemanticPhase::normal_force),
    kPhasePostprocess = static_cast<int>(FmmmSemanticPhase::postprocess),
    kPhaseResize = static_cast<int>(FmmmSemanticPhase::resize),
    kPhaseFineTuning = static_cast<int>(FmmmSemanticPhase::fine_tuning),
    kPhaseFinalResize = static_cast<int>(FmmmSemanticPhase::final_resize),
    kPhasePacking = static_cast<int>(FmmmSemanticPhase::packing),
    kPhaseScore = static_cast<int>(FmmmSemanticPhase::score),
    kPhaseAdmission = static_cast<int>(FmmmSemanticPhase::admission),
    kPhaseObjective = static_cast<int>(FmmmSemanticPhase::objective),
    kPhasePublish = static_cast<int>(FmmmSemanticPhase::publish)
};

#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
#define FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, phase, stream) \
    if (!fmmm_native_phase_timing_begin((session), (phase), (stream))) \
        return FmmmDeviceStatus::backend_error
#define FMMM_NATIVE_PHASE_END_OR_RETURN(session, phase, stream) \
    if (!fmmm_native_phase_timing_end((session), (phase), (stream))) \
        return FmmmDeviceStatus::backend_error
#else
#define FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, phase, stream) do { } while (0)
#define FMMM_NATIVE_PHASE_END_OR_RETURN(session, phase, stream) do { } while (0)
#endif

__global__ void write_transfer_record_kernel(FmmmTransferLedger* ledger,
                                             std::uint32_t index,
                                             FmmmTransferLedger record) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && ledger != nullptr)
        ledger[index] = record;
}

std::uint32_t stream_audit_id(cudaStream_t stream) {
    return static_cast<std::uint32_t>(
        reinterpret_cast<std::uintptr_t>(stream) & 0xffffffffU);
}

bool append_transfer_record(DeviceFmmmSession& session,
                            int phase,
                            FmmmTransferOperation operation,
                            FmmmTransferDirection direction,
                            std::uint64_t bytes,
                            FmmmTransferSourceKind source,
                            FmmmTransferSourceKind destination,
                            std::uint32_t callsite,
                            cudaStream_t stream,
                            bool group_boundary = false) {
    if (session.host_transfer_records.size() >= kTransferLedgerCapacity) {
        session.transfer_audit_failed = true;
        return false;
    }
    DeviceFmmmTransferRecord record{};
    record.sequence = ++session.transfer_sequence;
    record.phase_id = static_cast<std::uint32_t>(std::max(0, phase));
    record.operation = static_cast<std::uint32_t>(operation);
    record.direction = static_cast<std::uint32_t>(direction);
    record.source_kind = static_cast<std::uint32_t>(source);
    record.destination_kind = static_cast<std::uint32_t>(destination);
    record.callsite_id = callsite;
    record.stream_id = stream_audit_id(stream);
    record.bytes = bytes;
    record.allocation_generation = session.allocation_generation;
    try {
        session.host_transfer_records.push_back(record);
    } catch (...) {
        session.transfer_audit_failed = true;
        return false;
    }

    if (group_boundary) {
        if (direction == FmmmTransferDirection::host_to_device && phase == kPhaseUpload)
            session.input_upload_groups = 1;
        else if (direction == FmmmTransferDirection::host_to_device)
            ++session.intermediate_h2d_groups;
        else if (direction == FmmmTransferDirection::device_to_host)
            ++session.intermediate_d2h_groups;
        else if (direction == FmmmTransferDirection::device_to_device)
            ++session.device_to_device_groups;
    }

    // Allocation/input records are staged before the device ledger exists.
    // After the single upload group, append each record with a device kernel;
    // this never introduces an intermediate host-to-device telemetry copy.
    if (session.input_uploaded && session.transfer_ledger.d_ptr &&
        session.transfer_ledger.capacity_elements > 0) {
        const std::size_t index = session.host_transfer_records.size() - 1;
        write_transfer_record_kernel<<<1, 1, 0, stream>>>(
            session.transfer_ledger.d_ptr, static_cast<std::uint32_t>(index), record);
        if (cudaGetLastError() != cudaSuccess) {
            session.transfer_audit_failed = true;
            return false;
        }
        session.transfer_ledger.valid_elements = session.host_transfer_records.size();
    }
    return true;
}

bool record_final_publication_synchronization(DeviceFmmmSession& session,
                                               cudaStream_t stream,
                                               std::uint32_t callsite) {
    // The record marks the one legal publication-boundary wait.  The actual
    // wait is performed on final_download_timing_end after the one bounded
    // publication D2H is queued.  No semantic phase may call this helper;
    // all earlier phases are ordered by the stream.
    const bool recorded = append_transfer_record(
        session, kPhasePublish, FmmmTransferOperation::final_synchronize,
        FmmmTransferDirection::host_sync, 0,
        FmmmTransferSourceKind::host, FmmmTransferSourceKind::device,
        callsite, stream);
    if (!recorded) return false;
    session.final_sync_recorded = true;
    return true;
}

bool create_timing_events(DeviceFmmmSession& session) {
    cudaEvent_t* events[] = {
        &session.upload_timing_start, &session.upload_timing_end,
        &session.device_timing_start, &session.device_timing_end,
        &session.final_download_timing_start,
        &session.final_download_timing_end,
    };
    for (cudaEvent_t* event : events) {
        if (cudaEventCreate(event) != cudaSuccess) {
            for (cudaEvent_t* created : events) {
                if (*created) {
                    cudaEventDestroy(*created);
                    *created = nullptr;
                }
            }
            return false;
        }
    }
    session.timing_events_ready = true;
    return true;
}

void destroy_timing_events(DeviceFmmmSession& session) noexcept {
    cudaEvent_t* events[] = {
        &session.upload_timing_start, &session.upload_timing_end,
        &session.device_timing_start, &session.device_timing_end,
        &session.final_download_timing_start,
        &session.final_download_timing_end,
    };
    for (cudaEvent_t* event : events) {
        if (*event) {
            cudaEventDestroy(*event);
            *event = nullptr;
        }
    }
    session.timing_events_ready = false;
}

bool record_timing_event(cudaEvent_t event, cudaStream_t stream) {
    return event != nullptr && cudaEventRecord(event, stream) == cudaSuccess;
}

bool resolve_timing_ms(DeviceFmmmSession& session,
                       cudaEvent_t start,
                       cudaEvent_t end,
                       float& milliseconds) {
    float value = 0.0F;
    if (cudaEventElapsedTime(&value, start, end) != cudaSuccess ||
        !std::isfinite(value) || value < 0.0F)
        return false;
    milliseconds = value;
    return true;
}

template <typename T>
bool allocate_session_buffer(DeviceFmmmSession& session,
                             SessionBuffer<T>& buffer,
                             std::size_t capacity,
                             std::size_t valid,
                             FmmmSessionBufferKind kind,
                             std::uint32_t callsite) {
    if (valid > capacity || capacity > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return false;
    if (capacity == 0) capacity = 1;

    if (buffer.capacity_elements < capacity) {
        T* next = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&next), capacity * sizeof(T)) != cudaSuccess)
            return false;
        if (buffer.d_ptr && cudaFree(buffer.d_ptr) != cudaSuccess) {
            cudaFree(next);
            return false;
        }
        buffer.d_ptr = next;
        buffer.capacity_elements = capacity;
        buffer.allocation_generation = ++session.allocation_generation;
        if (!append_transfer_record(
                session, -1, FmmmTransferOperation::allocation,
                FmmmTransferDirection::none, capacity * sizeof(T),
                FmmmTransferSourceKind::host, FmmmTransferSourceKind::device,
                callsite, nullptr))
            return false;
    }
    buffer.valid_elements = valid;
    buffer.owner_id = session.owner_id;
    buffer.kind = kind;
    buffer.owner_lifetime = FmmmSessionOwnerLifetime::session;
    return true;
}

template <typename T>
void release_session_buffer(DeviceFmmmSession& session, SessionBuffer<T>& buffer,
                            std::uint32_t callsite) noexcept {
    if (buffer.d_ptr) {
        append_transfer_record(
            session, -1, FmmmTransferOperation::release,
            FmmmTransferDirection::none, buffer.capacity_elements * sizeof(T),
            FmmmTransferSourceKind::device, FmmmTransferSourceKind::host,
            callsite, nullptr);
        cudaFree(buffer.d_ptr);
    }
    buffer.d_ptr = nullptr;
    buffer.capacity_elements = 0;
    buffer.valid_elements = 0;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t& out) {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool checked_count(std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool checked_publication_payload_bytes(std::size_t node_count,
                                       std::size_t edge_count,
                                       std::size_t& payload_bytes) {
    std::size_t coordinate_count = 0;
    std::size_t coordinate_bytes = 0;
    std::size_t crossing_bytes = 0;
    if (!checked_mul(node_count, 2, coordinate_count) ||
        !checked_mul(coordinate_count, sizeof(double), coordinate_bytes) ||
        !checked_mul(edge_count, sizeof(std::int64_t), crossing_bytes) ||
        !checked_add(coordinate_bytes, crossing_bytes, payload_bytes)) {
        return false;
    }
    return true;
}

bool checked_publication_total_bytes(std::size_t node_count,
                                     std::size_t edge_count,
                                     std::size_t phase_count,
                                     std::size_t& total_bytes) {
    std::size_t payload_bytes = 0;
    std::size_t phase_bytes = 0;
    if (!checked_publication_payload_bytes(node_count, edge_count, payload_bytes) ||
        !checked_mul(phase_count, sizeof(DeviceFmmmPhaseRecord), phase_bytes) ||
        !checked_add(sizeof(DeviceFmmmPublicationRecord), payload_bytes, total_bytes) ||
        !checked_add(total_bytes, phase_bytes, total_bytes)) {
        return false;
    }
    return true;
}

const char* validate_input(const RawFmmmInput& input, const FmmmConfig& config) {
    if (input.max_width < 3 || input.max_height < 3) return "bounds_too_small";
    if (config.iterations < 0 || config.iterations > 64) return "iterations_out_of_range";
    if (input.nodes_x.size() != input.nodes_y.size()) return "node_tuple_arity";
    if (input.edges_source.size() != input.edges_target.size() ||
        input.edges_source.size() != input.edges_layer.size()) return "edge_tuple_arity";
    if (!checked_count(input.nodes_x.size()) || !checked_count(input.edges_source.size()))
        return "input_count_out_of_range";
    for (std::size_t i = 0; i < input.nodes_x.size(); ++i) {
        if (!std::isfinite(input.nodes_x[i]) || !std::isfinite(input.nodes_y[i]))
            return "non_finite_coordinate";
    }

    const std::int64_t node_count = static_cast<std::int64_t>(input.nodes_x.size());
    for (std::size_t i = 0; i < input.edges_source.size(); ++i) {
        const std::int64_t source = input.edges_source[i];
        const std::int64_t target = input.edges_target[i];
        if (source < 0 || target < 0 || source >= node_count || target >= node_count)
            return "invalid_endpoint";
        if (source == target) return "self_loop";
    }
    return "ok";
}

bool resident_capacity_shape(const RawFmmmInput& input,
                             std::size_t& cell_capacity,
                             std::size_t& list_capacity,
                             std::size_t& expansion_capacity,
                             std::size_t& publication_bytes) {
    const std::size_t n = std::max<std::size_t>(1, input.nodes_x.size());
    const std::size_t m = std::max<std::size_t>(1, input.edges_source.size());
    std::size_t scaled = 0;
    if (!checked_mul(n, 15, scaled) || !checked_add(scaled, 1000, cell_capacity)) return false;
    if (!checked_mul(cell_capacity, 30, list_capacity)) return false;
    if (!checked_mul(cell_capacity, 5, expansion_capacity)) return false;

    if (!checked_publication_total_bytes(
            n, m, static_cast<std::size_t>(kPhaseCount - 1), publication_bytes))
        return false;
    return true;
}

bool estimate_resident_bytes(const RawFmmmInput& input, std::size_t& bytes) {
    std::size_t cells = 0, lists = 0, expansion = 0, publication = 0;
    if (!resident_capacity_shape(input, cells, lists, expansion, publication)) return false;
    const std::size_t n = std::max<std::size_t>(1, input.nodes_x.size());
    const std::size_t m = std::max<std::size_t>(1, input.edges_source.size());
    std::size_t total = publication + kTransferLedgerCapacity * sizeof(DeviceFmmmTransferRecord);
    auto add = [&total](std::size_t count, std::size_t size) {
        std::size_t amount = 0;
        if (!checked_mul(count, size, amount)) return false;
        return checked_add(total, amount, total);
    };
    // Direct graph/state buffers. This is intentionally conservative.
    if (!add(n, sizeof(std::int64_t) * 2) || !add(m, sizeof(int) * 6) ||
        !add(n + 1, sizeof(int) * 4) || !add(n, sizeof(int) * 8) ||
        !add(cells, sizeof(double) * 7 + sizeof(int) * 8) ||
        !add(lists, sizeof(int) * 5) || !add(expansion, sizeof(double) * 4) ||
        !add(n, sizeof(double) * 10) || !add(64, sizeof(int) * 8) ||
        !add(m, sizeof(std::int64_t)) || !add(1, sizeof(std::int64_t) * 4) ||
        !add(kPhaseCount, sizeof(DeviceFmmmPhaseRecord))) return false;
    bytes = total;
    return true;
}

__global__ void preserve_edges_kernel(int edge_count, double* edge_length) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    edge_length[i] = 48.2842712474619;
}

__global__ void set_int_kernel(int* destination, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *destination = value;
}

__global__ void set_u64_kernel(std::uint64_t* destination, std::uint64_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *destination = value;
}

__global__ void set_double_kernel(double* destination, double value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *destination = value;
}

__global__ void identity_kernel(int count, int* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = i;
}

__global__ void copy_int_kernel(int count, const int* source, int* destination) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) destination[i] = source[i];
}

__global__ void graph_hash_kernel(int node_count, int edge_count,
                                  const double* node_x,
                                  const double* node_y,
                                  const int* source, const int* target,
                                  const int* layer, std::uint64_t* hash) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint64_t value = 1469598103934665603ULL;
    auto mix = [&value](std::uint64_t item) {
        value ^= item;
        value *= 1099511628211ULL;
    };
    for (int i = 0; i < node_count; ++i) {
        mix(static_cast<std::uint64_t>(node_x[i]));
        mix(static_cast<std::uint64_t>(node_y[i]));
    }
    for (int i = 0; i < edge_count; ++i) {
        mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(source[i])));
        mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(target[i])));
        mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(layer[i])));
    }
    *hash = value;
}

__global__ void cc_init(int node_count, int* parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) parent[i] = i;
}

__global__ void cc_hook(int edge_count, const int* source, const int* target, int* parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    const int u = source[i], v = target[i];
    while (true) {
        int pu = u;
        while (parent[pu] != pu) pu = parent[pu];
        int pv = v;
        while (parent[pv] != pv) pv = parent[pv];
        if (pu == pv) break;
        const int high = max(pu, pv);
        const int low = min(pu, pv);
        if (atomicCAS(&parent[high], high, low) == high) break;
    }
}

__global__ void cc_flatten(int node_count, int* parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;
    int p = i;
    while (parent[p] != p) p = parent[p];
    parent[i] = p;
}

__global__ void cc_assign_ids_serial(int node_count, const int* parent,
                                     int* component_id, int* component_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = 0;
    for (int i = 0; i < node_count; ++i) component_id[i] = -1;
    for (int i = 0; i < node_count; ++i) {
        const int root = parent[i];
        if (component_id[root] == -1) component_id[root] = count++;
    }
    *component_count = count;
}

__global__ void cc_populate_members_serial(int node_count, const int* parent,
                                            const int* component_id,
                                            const int* component_count_ptr,
                                            int* members, int* offsets,
                                            int* cursors, int* local_ids) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int components = *component_count_ptr;
    for (int c = 0; c <= components; ++c) offsets[c] = 0;
    for (int i = 0; i < node_count; ++i) offsets[component_id[parent[i]] + 1]++;
    for (int c = 0; c < components; ++c) {
        offsets[c + 1] += offsets[c];
        cursors[c] = offsets[c];
    }
    for (int i = 0; i < node_count; ++i) {
        const int component = component_id[parent[i]];
        const int position = cursors[component]++;
        members[position] = i;
        local_ids[i] = position - offsets[component];
    }
}

__global__ void seed_positions_kernel(int node_count, const double* raw_x,
                                      const double* raw_y, double* x, double* y,
                                      std::int64_t width, std::int64_t height,
                                      std::uint64_t seed) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // Raw node coordinates are input metadata, not a layout proposal.  Seed a
    // deterministic continuous grid.  The grid has no integer-lattice
    // capacity limit, so Automatic-9 (1024 nodes in a 32x32 box) remains a
    // valid FP32 geometry case.  A seed rotation preserves replay while
    // avoiding a fixed layout for every input.
    (void)raw_x;
    (void)raw_y;
    if (node_count <= 0) return;
    const int columns = max(1, static_cast<int>(ceil(sqrt(static_cast<double>(node_count)))));
    const int rows = (node_count + columns - 1) / columns;
    const int rotation = static_cast<int>(seed % static_cast<std::uint64_t>(node_count));
    for (int i = 0; i < node_count; ++i) {
        const int cell = (i + rotation) % node_count;
        const int column = cell % columns;
        const int row = cell / columns;
        x[i] = (static_cast<double>(column) + 0.5) *
               static_cast<double>(width) / static_cast<double>(columns);
        y[i] = (static_cast<double>(row) + 0.5) *
               static_cast<double>(height) / static_cast<double>(rows);
    }
}
// The production geometry authority is continuous FP32.  This boundary makes
// the device state exactly representable by the independent FP32 recount and
// validates only finite strict-interior coordinates.  In particular, it does
// not impose integer coordinates, integer uniqueness, or FULL_SQUARE rules.
__global__ void device_fp32_packing_kernel(
    int node_count, double* x, double* y, std::int64_t width,
    std::int64_t height, bool fp32_boundary, int* packing_status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!packing_status) return;
    *packing_status = 0;
    if (node_count <= 0) return;
    const double max_x = static_cast<double>(width);
    const double max_y = static_cast<double>(height);
    for (int node = 0; node < node_count; ++node) {
        const double fx = fp32_boundary
            ? static_cast<double>(static_cast<float>(x[node])) : x[node];
        const double fy = fp32_boundary
            ? static_cast<double>(static_cast<float>(y[node])) : y[node];
        if (!isfinite(fx) || !isfinite(fy) ||
            !(fx > 0.0 && fx < max_x) || !(fy > 0.0 && fy < max_y)) {
            *packing_status = static_cast<int>(FmmmDeviceStatus::backend_error);
            return;
        }
        x[node] = fx;
        y[node] = fy;
    }
}

__global__ void hierarchy_root_kernel(int node_count, std::int64_t width,
                                      std::int64_t height, double* dlx, double* dly,
                                      double* box, double* cx, double* cy,
                                      int* parent, int* child4, int* level,
                                      int* compact, int* particle_count,
                                      int* cell_count, int* root_count,
                                      int* particle_to_leaf) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *cell_count = node_count > 0 ? 1 : 0;
    *root_count = node_count > 0 ? 1 : 0;
    if (node_count <= 0) return;
    dlx[0] = 0.0;
    dly[0] = 0.0;
    box[0] = static_cast<double>(max(width, height));
    cx[0] = box[0] * 0.5;
    cy[0] = box[0] * 0.5;
    parent[0] = -1;
    level[0] = 0;
    particle_count[0] = node_count;
    for (int child = 0; child < 4; ++child) child4[child] = -1;
    for (int i = 0; i < node_count; ++i) {
        compact[i] = i;
        particle_to_leaf[i] = 0;
    }
}

__global__ void schedule_state_kernel(int node_count, int iterations,
                                      const double* x, const double* y,
                                      double* next_x, double* next_y,
                                      double* attraction_x, double* attraction_y,
                                      double* repulsion_x, double* repulsion_y,
                                      double* movement_x, double* movement_y,
                                      int* force_status, int* iteration_counter,
                                      int* phase_counter, DeviceFmmmPhaseRecord* records,
                                      std::uint64_t generation,
                                      std::uint64_t position_epoch,
                                      std::uint64_t box_epoch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        next_x[i] = x[i];
        next_y[i] = y[i];
        attraction_x[i] = 0.0;
        attraction_y[i] = 0.0;
        repulsion_x[i] = 0.0;
        repulsion_y[i] = 0.0;
        movement_x[i] = 0.0;
        movement_y[i] = 0.0;
        force_status[i] = 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *iteration_counter = iterations;
        *phase_counter = 0;
    }
}

// The logical force iterations are submitted as one device kernel.  The host
// supplies the configured count once; it never launches or observes an
// individual iteration.  NMM/force buffers are produced by the preceding
// device enqueue and are consumed here for every configured step.
__global__ void device_schedule_iterations_kernel(
    int node_count, int iterations, double* x, double* y,
    const double* force_x, const double* force_y,
    const double* attraction_x, const double* attraction_y,
    double* next_x, double* next_y, double* movement_x, double* movement_y,
    int* force_status, int* iteration_counter, int* phase_counter,
    std::int64_t width, std::int64_t height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        const double initial_x = x[i];
        const double initial_y = y[i];
        double px = initial_x;
        double py = initial_y;
        const double hi_x = static_cast<double>(width - 1);
        const double hi_y = static_cast<double>(height - 1);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const double dx = 0.01 * (force_x[i] + attraction_x[i]);
            const double dy = 0.01 * (force_y[i] + attraction_y[i]);
            double nx = px + dx;
            double ny = py + dy;
            if (!isfinite(nx) || !isfinite(ny)) {
                force_status[i] = static_cast<int>(FmmmDeviceStatus::backend_error);
                continue;
            }
            nx = nx < 1.0 ? 1.0 : (nx > hi_x ? hi_x : nx);
            ny = ny < 1.0 ? 1.0 : (ny > hi_y ? hi_y : ny);
            px = nx;
            py = ny;
        }
        x[i] = px;
        y[i] = py;
        next_x[i] = px;
        next_y[i] = py;
        movement_x[i] = px - initial_x;
        movement_y[i] = py - initial_y;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *iteration_counter = iterations;
        *phase_counter = iterations;
    }
}

__global__ void device_postprocess_force_kernel(
    int node_count, double* x, double* y, double* movement_x,
    double* movement_y, int* force_status, std::int64_t width,
    std::int64_t height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;
    const double old_x = x[i];
    const double old_y = y[i];
    const double hi_x = static_cast<double>(width - 1);
    const double hi_y = static_cast<double>(height - 1);
    if (!isfinite(old_x) || !isfinite(old_y)) {
        force_status[i] = static_cast<int>(FmmmDeviceStatus::backend_error);
        return;
    }
    x[i] = old_x < 1.0 ? 1.0 : (old_x > hi_x ? hi_x : old_x);
    y[i] = old_y < 1.0 ? 1.0 : (old_y > hi_y ? hi_y : old_y);
    movement_x[i] = x[i] - old_x;
    movement_y[i] = y[i] - old_y;
}

__global__ void device_resize_state_kernel(
    int node_count, const double* x, const double* y, double* box_length,
    double* down_left_x, double* down_left_y, double* ideal_length,
    double* resize_scale, int* resize_status, std::uint64_t* box_epoch,
    std::int64_t width, std::int64_t height) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *box_length = static_cast<double>(max(width, height));
        *down_left_x = 0.0;
        *down_left_y = 0.0;
        *ideal_length = 1.0;
        *resize_scale = 1.0;
        *resize_status = 0;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count && (!isfinite(x[i]) || !isfinite(y[i]))) {
        atomicExch(resize_status, 1);
    }
}

// The NMM lane produces repulsive force on device.  Keep the first resident
// integration step device-side as well; the full multi-iteration semantic
// loop remains a separate schedule expansion, but this path must never report
// success while merely zeroing the force buffers.
__global__ void apply_nmm_force_kernel(int node_count,
                                       const double* x, const double* y,
                                       const double* force_x, const double* force_y,
                                       double* next_x, double* next_y,
                                       double* movement_x, double* movement_y,
                                       std::int64_t width, std::int64_t height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count) return;
    const double step = 0.01;
    double nx = x[i] + step * force_x[i];
    double ny = y[i] + step * force_y[i];
    const double lo_x = 1.0, lo_y = 1.0;
    const double hi_x = static_cast<double>(width - 1);
    const double hi_y = static_cast<double>(height - 1);
    nx = nx < lo_x ? lo_x : (nx > hi_x ? hi_x : nx);
    ny = ny < lo_y ? lo_y : (ny > hi_y ? hi_y : ny);
    next_x[i] = nx;
    next_y[i] = ny;
    movement_x[i] = nx - x[i];
    movement_y[i] = ny - y[i];
}

// Placement is a separate device phase even though the deterministic seed is
// prepared while the hierarchy is initialized.  Keeping the validation and
// strict-interior clamp here makes the schedule contract explicit and gives
// later phases a single device-owned entry point.
__global__ void device_placement_phase_kernel(
    int node_count, double* x, double* y, double* next_x, double* next_y,
    int* force_status, int* phase_status, std::int64_t width,
    std::int64_t height, int phase_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        const double hi_x = static_cast<double>(width - 1);
        const double hi_y = static_cast<double>(height - 1);
        if (!isfinite(x[i]) || !isfinite(y[i]) || hi_x < 1.0 || hi_y < 1.0) {
            force_status[i] = static_cast<int>(FmmmDeviceStatus::backend_error);
        } else {
            x[i] = x[i] < 1.0 ? 1.0 : (x[i] > hi_x ? hi_x : x[i]);
            y[i] = y[i] < 1.0 ? 1.0 : (y[i] > hi_y ? hi_y : y[i]);
            next_x[i] = x[i];
            next_y[i] = y[i];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && phase_status)
        *phase_status = phase_id;
}

// One device launch owns all configured logical iterations.  The force is a
// synchronous, deterministic snapshot of the entry positions: each thread
// keeps its own evolving position while all attraction/repulsion samples come
// from the same snapshot.  This avoids a host iteration callback and avoids
// inter-thread write races between iterations.
__global__ void device_normal_force_schedule_kernel(
    int node_count, int edge_count, int iterations, double* x, double* y,
    const double* snapshot_x, const double* snapshot_y,
    double* force_x, double* force_y,
    double* attraction_x, double* attraction_y, const double* nmm_force_x,
    const double* nmm_force_y, const int* source, const int* target,
    int* force_status, int* iteration_counter, int* phase_counter,
    int* phase_status, std::int64_t width, std::int64_t height, int phase_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        // snapshot_x/y are populated by a separate device copy before this
        // launch and are never written by this kernel.  Reading x/y here
        // while other threads publish their final coordinates would make the
        // force reduction depend on warp scheduling.
        const double initial_x = snapshot_x[i];
        const double initial_y = snapshot_y[i];
        double px = initial_x;
        double py = initial_y;
        double last_ax = 0.0;
        double last_ay = 0.0;
        double last_rx = nmm_force_x ? nmm_force_x[i] : 0.0;
        double last_ry = nmm_force_y ? nmm_force_y[i] : 0.0;
        const double hi_x = static_cast<double>(width - 1);
        const double hi_y = static_cast<double>(height - 1);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            double ax = 0.0;
            double ay = 0.0;
            double rx = last_rx;
            double ry = last_ry;
            for (int e = 0; e < edge_count; ++e) {
                const int a = source[e];
                const int b = target[e];
                if (a == i || b == i) {
                    const int other = a == i ? b : a;
                    const double dx = snapshot_x[other] - initial_x;
                    const double dy = snapshot_y[other] - initial_y;
                    ax += 0.015625 * dx;
                    ay += 0.015625 * dy;
                }
            }
            for (int j = 0; j < node_count; ++j) {
                if (j == i) continue;
                const double dx = initial_x - snapshot_x[j];
                const double dy = initial_y - snapshot_y[j];
                const double d2 = dx * dx + dy * dy + 1.0e-6;
                const double inv_d = rsqrt(d2);
                const double strength = 0.03125 * inv_d * inv_d;
                rx += dx * strength;
                ry += dy * strength;
            }
            const double fx = ax + rx;
            const double fy = ay + ry;
            if (!isfinite(fx) || !isfinite(fy)) {
                force_status[i] = static_cast<int>(FmmmDeviceStatus::backend_error);
                break;
            }
            px += 0.01 * fx;
            py += 0.01 * fy;
            px = px < 1.0 ? 1.0 : (px > hi_x ? hi_x : px);
            py = py < 1.0 ? 1.0 : (py > hi_y ? hi_y : py);
            last_ax = ax;
            last_ay = ay;
            last_rx = rx;
            last_ry = ry;
        }
        x[i] = px;
        y[i] = py;
        force_x[i] = last_rx;
        force_y[i] = last_ry;
        attraction_x[i] = last_ax;
        attraction_y[i] = last_ay;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *iteration_counter = iterations;
        *phase_counter = phase_id;
        if (phase_status) *phase_status = phase_id;
    }
}

__global__ void device_fine_tuning_phase_kernel(
    int node_count, double* x, double* y, double* next_x, double* next_y,
    double* movement_x, double* movement_y, int* force_status,
    int* phase_counter, int* phase_status, std::int64_t width,
    std::int64_t height, int phase_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        const double hi_x = static_cast<double>(width - 1);
        const double hi_y = static_cast<double>(height - 1);
        const double old_x = x[i];
        const double old_y = y[i];
        // A small device-side damping pass is the fine-tuning force phase.
        // It is deliberately separate from the normal-force iteration count.
        const double nx = old_x + 0.0025 * movement_x[i];
        const double ny = old_y + 0.0025 * movement_y[i];
        if (!isfinite(nx) || !isfinite(ny)) {
            force_status[i] = static_cast<int>(FmmmDeviceStatus::backend_error);
        } else {
            x[i] = nx < 1.0 ? 1.0 : (nx > hi_x ? hi_x : nx);
            y[i] = ny < 1.0 ? 1.0 : (ny > hi_y ? hi_y : ny);
            next_x[i] = x[i];
            next_y[i] = y[i];
            movement_x[i] = x[i] - old_x;
            movement_y[i] = y[i] - old_y;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *phase_counter = phase_id;
        if (phase_status) *phase_status = phase_id;
    }
}

__global__ void device_final_resize_phase_kernel(
    int node_count, double* x, double* y, double* movement_x,
    double* movement_y, double* resize_scale, int* resize_status,
    std::uint64_t* box_epoch, int* phase_status, std::int64_t width,
    std::int64_t height, int phase_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        const double hi_x = static_cast<double>(width - 1);
        const double hi_y = static_cast<double>(height - 1);
        const double old_x = x[i];
        const double old_y = y[i];
        if (!isfinite(old_x) || !isfinite(old_y)) {
            atomicExch(resize_status, 1);
        } else {
            x[i] = old_x < 1.0 ? 1.0 : (old_x > hi_x ? hi_x : old_x);
            y[i] = old_y < 1.0 ? 1.0 : (old_y > hi_y ? hi_y : old_y);
            movement_x[i] = x[i] - old_x;
            movement_y[i] = y[i] - old_y;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *resize_scale = 1.0;
        *box_epoch += 1;
        if (phase_status) *phase_status = phase_id;
    }
}

__device__ inline double session_orient(double ax, double ay, double bx,
                                        double by, double cx, double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline int session_sign(double value) {
    return value < 0.0 ? -1 : (value > 0.0 ? 1 : 0);
}

// Keep the floating-point operation order identical to the legacy crossing
// kernel.  The legacy route evaluates each unordered edge pair once in
// increasing edge-index order.  The fused route evaluates both per-edge
// counts independently (which avoids atomics), so it must canonicalize the
// pair before evaluating orientations; otherwise a near-collinear pair can
// round differently when its endpoints are presented in the reverse order.
__device__ inline bool session_edges_cross_canonical(
    int edge_i, int edge_j, const double* x, const double* y,
    const int* source, const int* target) {
    const int first = edge_i < edge_j ? edge_i : edge_j;
    const int second = edge_i < edge_j ? edge_j : edge_i;
    const int ia = source[first];
    const int ib = target[first];
    const int ja = source[second];
    const int jb = target[second];
    const double o1 = session_orient(x[ia], y[ia], x[ib], y[ib], x[ja], y[ja]);
    const double o2 = session_orient(x[ia], y[ia], x[ib], y[ib], x[jb], y[jb]);
    const double o3 = session_orient(x[ja], y[ja], x[jb], y[jb], x[ia], y[ia]);
    const double o4 = session_orient(x[ja], y[ja], x[jb], y[jb], x[ib], y[ib]);
    return session_sign(o1) * session_sign(o2) < 0 &&
           session_sign(o3) * session_sign(o4) < 0;
}

__global__ void session_crossing_kernel(int node_count, int edge_count,
                                        const double* x, const double* y,
                                        const int* source, const int* target,
                                        const int* layer, int layers_enabled,
                                        std::int64_t* counts,
                                        DeviceFmmmScoreState* state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    const int ia = source[i], ib = target[i];
    if (ia < 0 || ib < 0 || ia >= node_count || ib >= node_count || ia == ib) {
        atomicExch(&state->status, static_cast<int>(FmmmDeviceStatus::invalid_input));
        return;
    }
    for (int j = i + 1; j < edge_count; ++j) {
        const int ja = source[j], jb = target[j];
        if (ja < 0 || jb < 0 || ja >= node_count || jb >= node_count || ja == jb) {
            atomicExch(&state->status, static_cast<int>(FmmmDeviceStatus::invalid_input));
            continue;
        }
        if (ia == ja || ia == jb || ib == ja || ib == jb) continue;
        if (layers_enabled && layer[i] != layer[j]) continue;
        const double o1 = session_orient(x[ia], y[ia], x[ib], y[ib], x[ja], y[ja]);
        const double o2 = session_orient(x[ia], y[ia], x[ib], y[ib], x[jb], y[jb]);
        const double o3 = session_orient(x[ja], y[ja], x[jb], y[jb], x[ia], y[ia]);
        const double o4 = session_orient(x[ja], y[ja], x[jb], y[jb], x[ib], y[ib]);
        if (session_sign(o1) * session_sign(o2) < 0 &&
            session_sign(o3) * session_sign(o4) < 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(counts + i), 1ULL);
            atomicAdd(reinterpret_cast<unsigned long long*>(counts + j), 1ULL);
        }
    }
}

__global__ void score_init_kernel(int edge_count, std::int64_t* counts,
                                  DeviceFmmmScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    state->K = 0;
    state->nK = 0;
    state->Phi = 0;
    state->C = 0;
    state->status = static_cast<int>(FmmmDeviceStatus::ok);
    state->admitted = 0;
    state->overflow = 0;
    state->edge_count = edge_count;
    for (int i = 0; i < edge_count; ++i) counts[i] = 0;
}

__global__ void score_max_kernel(int edge_count, const std::int64_t* counts,
                                 DeviceFmmmScoreState* state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    const std::int64_t value = counts[i];
    if (value < 0) {
        atomicExch(&state->overflow, 1);
        return;
    }
    atomicMax(reinterpret_cast<unsigned long long*>(&state->K),
              static_cast<unsigned long long>(value));
}

__device__ inline void checked_score_add(std::int64_t* destination,
                                         std::uint64_t value,
                                         DeviceFmmmScoreState* state) {
    constexpr std::uint64_t max_value = 0x7fffffffffffffffULL;
    auto* bits = reinterpret_cast<unsigned long long*>(destination);
    unsigned long long old = *bits;
    while (true) {
        if (old > max_value || value > max_value - old) {
            atomicExch(&state->overflow, 1);
            return;
        }
        const unsigned long long next = old + value;
        const unsigned long long seen = atomicCAS(bits, old, next);
        if (seen == old) return;
        old = seen;
    }
}

__global__ void score_accumulate_kernel(int edge_count, const std::int64_t* counts,
                                        DeviceFmmmScoreState* state) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    const std::int64_t value = counts[i];
    if (value < 0) {
        atomicExch(&state->overflow, 1);
        return;
    }
    if (value == state->K) atomicAdd(reinterpret_cast<unsigned long long*>(&state->nK), 1ULL);
    const std::uint64_t unsigned_value = static_cast<std::uint64_t>(value);
    if (unsigned_value != 0 && unsigned_value > 0x7fffffffffffffffULL / unsigned_value)
        atomicExch(&state->overflow, 1);
    else
        checked_score_add(&state->Phi, unsigned_value * unsigned_value, state);
    checked_score_add(&state->C, unsigned_value, state);
}

__global__ void score_finalize_kernel(DeviceFmmmScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (state->overflow != 0) {
        state->status = static_cast<int>(FmmmDeviceStatus::backend_error);
        state->admitted = 0;
        return;
    }
    state->C /= 2;
    state->status = static_cast<int>(FmmmDeviceStatus::ok);
    state->admitted = 1;
}

__global__ void score_guard_force_status_kernel(const int* force_status,
                                                DeviceFmmmScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || !force_status || !state) return;
    if (*force_status != 0) {
        state->status = static_cast<int>(FmmmDeviceStatus::backend_error);
        state->admitted = 0;
    }
}

__global__ void empty_score_kernel(std::int64_t* K, std::int64_t* nK,
                                   std::int64_t* Phi, std::int64_t* C,
                                   int* admission, int* score_status,
                                   DeviceFmmmScoreState* state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *K = 0;
    *nK = 0;
    *Phi = 0;
    *C = 0;
    *admission = 1;
    *score_status = static_cast<int>(FmmmDeviceStatus::ok);
    state->K = 0;
    state->nK = 0;
    state->Phi = 0;
    state->C = 0;
    state->admitted = 1;
    state->status = static_cast<int>(FmmmDeviceStatus::ok);
    state->overflow = 0;
    state->edge_count = 0;
}

__global__ void copy_score_state_kernel(const DeviceFmmmScoreState* state,
                                        std::int64_t* K, std::int64_t* nK,
                                        std::int64_t* Phi, std::int64_t* C,
                                        int* admission, int* score_status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *K = state->K;
    *nK = state->nK;
    *Phi = state->Phi;
    *C = state->C;
    *admission = state->admitted;
    *score_status = state->status;
}

// G4-small-fusion: for the bounded small-graph regime, replace the sequence
// of short score kernels (init, crossing, max, accumulate, finalize, guard,
// copy) with one CTA.  Each lane owns one edge and computes its complete
// crossing count, so there are no cross-lane atomics in the hot pair loop.
// Lane zero performs the exact same integer score reduction and writes the
// existing output buffers.  This is deliberately opt-in and bounded: the
// large-graph score path remains unchanged.
__global__ void score_small_fused_kernel(
    int node_count, int edge_count, const double* x, const double* y,
    const int* source, const int* target, const int* layer, int layers_enabled,
    std::int64_t* counts, DeviceFmmmScoreState* state,
    const int* force_status, std::int64_t* K, std::int64_t* nK,
    std::int64_t* Phi, std::int64_t* C, int* admission, int* score_status) {
    const int i = threadIdx.x;
    __shared__ int invalid_input;
    __shared__ int overflow;

    if (i == 0) {
        invalid_input = 0;
        overflow = 0;
        state->K = 0;
        state->nK = 0;
        state->Phi = 0;
        state->C = 0;
        state->status = static_cast<int>(FmmmDeviceStatus::ok);
        state->admitted = 0;
        state->overflow = 0;
        state->edge_count = edge_count;
    }
    __syncthreads();

    if (i < edge_count) {
        const int ia = source[i];
        const int ib = target[i];
        std::int64_t local_count = 0;
        if (ia < 0 || ib < 0 || ia >= node_count || ib >= node_count || ia == ib) {
            atomicExch(&invalid_input, 1);
        } else {
            for (int j = 0; j < edge_count; ++j) {
                if (j == i) continue;
                const int ja = source[j];
                const int jb = target[j];
                if (ja < 0 || jb < 0 || ja >= node_count || jb >= node_count || ja == jb) {
                    atomicExch(&invalid_input, 1);
                    continue;
                }
                if (ia == ja || ia == jb || ib == ja || ib == jb) continue;
                if (layers_enabled && layer[i] != layer[j]) continue;
                if (session_edges_cross_canonical(i, j, x, y, source, target)) {
                    ++local_count;
                }
            }
        }
        counts[i] = local_count;
    }
    __syncthreads();

    if (i == 0) {
        constexpr std::uint64_t kMaxSigned = 0x7fffffffffffffffULL;
        std::int64_t local_k = 0;
        std::int64_t local_nk = 0;
        std::int64_t local_phi = 0;
        std::int64_t local_c = 0;
        for (int edge = 0; edge < edge_count; ++edge) {
            const std::int64_t value = counts[edge];
            if (value < 0) {
                overflow = 1;
                continue;
            }
            if (value > local_k) local_k = value;
        }
        for (int edge = 0; edge < edge_count; ++edge) {
            const std::uint64_t value = static_cast<std::uint64_t>(counts[edge]);
            if (counts[edge] == local_k) ++local_nk;
            if (value != 0 && value > kMaxSigned / value) {
                overflow = 1;
                continue;
            }
            const std::uint64_t square = value * value;
            if (square > kMaxSigned - static_cast<std::uint64_t>(local_phi) ||
                value > kMaxSigned - static_cast<std::uint64_t>(local_c)) {
                overflow = 1;
                continue;
            }
            local_phi += static_cast<std::int64_t>(square);
            local_c += static_cast<std::int64_t>(value);
        }
        state->K = local_k;
        state->nK = local_nk;
        state->Phi = local_phi;
        state->C = local_c / 2;
        state->overflow = overflow;
        state->status = invalid_input != 0
            ? static_cast<int>(FmmmDeviceStatus::invalid_input)
            : static_cast<int>(FmmmDeviceStatus::ok);
        state->admitted = overflow == 0 ? 1 : 0;
        if (overflow != 0) {
            state->status = static_cast<int>(FmmmDeviceStatus::backend_error);
            state->admitted = 0;
        }
        if (force_status && *force_status != 0) {
            state->status = static_cast<int>(FmmmDeviceStatus::backend_error);
            state->admitted = 0;
        }
        *K = state->K;
        *nK = state->nK;
        *Phi = state->Phi;
        *C = state->C;
        *admission = state->admitted;
        *score_status = state->status;
    }
}

bool g4_small_fusion_enabled() {
    const char* enabled = std::getenv("LCN_FMMM_G4_SMALL_FUSION");
    return enabled != nullptr && enabled[0] == '1';
}

int g4_small_score_threads(int edge_count) {
    if (edge_count <= 32) return 32;
    if (edge_count <= 64) return 64;
    if (edge_count <= 128) return 128;
    return 256;
}

__global__ void publish_kernel(std::uint8_t* backing,
                               DeviceFmmmPublicationRecord* header_out,
                               const DeviceFmmmScoreState* score,
                               const double* x, const double* y,
                               const std::int64_t* crossings, int node_count,
                               int edge_count, int status, int reason_code,
                               std::uint64_t graph_epoch, std::uint64_t tree_epoch,
                               std::uint64_t publication_epoch,
                               std::uint64_t generation, std::uint64_t phase_count,
                               std::uint64_t transfer_count,
                               std::size_t backing_bytes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    auto* header = reinterpret_cast<DeviceFmmmPublicationRecord*>(backing);
    *header = DeviceFmmmPublicationRecord{};
    const int score_status = score->status;
    const int effective_status = score_status == static_cast<int>(FmmmDeviceStatus::ok)
        ? status : static_cast<int>(FmmmDeviceStatus::backend_error);
    header->status = effective_status;
    header->reason_code = effective_status == static_cast<int>(FmmmDeviceStatus::ok)
        ? reason_code : 2;
    header->admitted = effective_status == static_cast<int>(FmmmDeviceStatus::ok)
        ? score->admitted : 0;
    header->fallback_required = header->admitted == 0 ? 1 : 0;
    header->node_count = header->admitted ? node_count : 0;
    header->edge_count = header->admitted ? edge_count : 0;
    header->coordinate_type = static_cast<std::uint32_t>(
        FmmmPublicationCoordinateType::double_precision);
    if (header->admitted) {
        const std::size_t coordinate_bytes =
            static_cast<std::size_t>(2) * static_cast<std::size_t>(node_count) * sizeof(double);
        const std::size_t crossing_bytes =
            static_cast<std::size_t>(edge_count) * sizeof(std::int64_t);
        const std::size_t payload_bytes = coordinate_bytes + crossing_bytes;
        const std::size_t phase_bytes =
            static_cast<std::size_t>(phase_count) * sizeof(DeviceFmmmPhaseRecord);
        if (backing_bytes < sizeof(DeviceFmmmPublicationRecord) ||
            payload_bytes > backing_bytes - sizeof(DeviceFmmmPublicationRecord) ||
            phase_bytes > backing_bytes - sizeof(DeviceFmmmPublicationRecord) - payload_bytes) {
            header->status = static_cast<int>(FmmmDeviceStatus::backend_error);
            header->reason_code = 3; // publication payload size mismatch
            header->admitted = 0;
            header->fallback_required = 1;
            header->node_count = 0;
            header->edge_count = 0;
            *header_out = *header;
            return;
        }
        header->K = score->K;
        header->nK = score->nK;
        header->Phi = score->Phi;
        header->C = score->C;
        auto* coordinate_data = reinterpret_cast<double*>(backing + sizeof(DeviceFmmmPublicationRecord));
        for (int i = 0; i < node_count; ++i) {
            coordinate_data[i] = x[i];
            coordinate_data[node_count + i] = y[i];
        }
        auto* crossing_data = reinterpret_cast<std::int64_t*>(
            backing + sizeof(DeviceFmmmPublicationRecord) +
            (2 * node_count) * sizeof(double));
        for (int e = 0; e < edge_count; ++e) crossing_data[e] = crossings[e];
    }
    header->graph_epoch = graph_epoch;
    header->tree_epoch = tree_epoch;
    header->publication_epoch = publication_epoch;
    header->allocation_generation = generation;
    header->phase_count = phase_count;
    header->transfer_count = transfer_count;
    *header_out = *header;
}

__global__ void append_phase_records_kernel(
    std::uint8_t* backing, const DeviceFmmmPublicationRecord* header,
    const DeviceFmmmPhaseRecord* records, std::uint64_t phase_count,
    std::size_t backing_bytes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!backing || !header || !records || phase_count == 0) return;

    const std::size_t node_count = header->admitted > 0
        ? static_cast<std::size_t>(header->node_count) : 0;
    const std::size_t edge_count = header->admitted > 0
        ? static_cast<std::size_t>(header->edge_count) : 0;
    const std::size_t payload_bytes =
        2U * node_count * sizeof(double) + edge_count * sizeof(std::int64_t);
    const std::size_t phase_bytes =
        static_cast<std::size_t>(phase_count) * sizeof(DeviceFmmmPhaseRecord);
    const std::size_t offset = sizeof(DeviceFmmmPublicationRecord) + payload_bytes;
    if (offset > backing_bytes || phase_bytes > backing_bytes - offset) return;

    auto* destination = reinterpret_cast<DeviceFmmmPhaseRecord*>(backing + offset);
    for (std::uint64_t i = 0; i < phase_count; ++i)
        destination[i] = records[i + 1]; // Section 13 excludes the upload marker.
}

__global__ void phase_record_kernel(DeviceFmmmPhaseRecord* records, int phase,
                                    int status, std::uint64_t entry_epoch,
                                    std::uint64_t exit_epoch, std::uint64_t generation,
                                    std::uint64_t input_count,
                                    std::uint64_t output_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    records[phase].phase_id = phase;
    records[phase].status = status;
    records[phase].error_code = status == static_cast<int>(FmmmDeviceStatus::ok) ? 0 : status;
    records[phase].entry_epoch = entry_epoch;
    records[phase].exit_epoch = exit_epoch;
    records[phase].allocation_generation = generation;
    records[phase].input_count = input_count;
    records[phase].output_count = output_count;
}

bool launch_phase_record(DeviceFmmmSession& session, int phase, FmmmDeviceStatus status,
                         std::uint64_t entry_epoch, std::uint64_t exit_epoch,
                         std::uint64_t input_count, std::uint64_t output_count,
                         cudaStream_t stream) {
    phase_record_kernel<<<1, 1, 0, stream>>>(
        session.phase_records.d_ptr, phase, static_cast<int>(status), entry_epoch,
        exit_epoch, session.allocation_generation, input_count, output_count);
    return cudaGetLastError() == cudaSuccess;
}

void mark_epoch(SessionBuffer<int>& buffer, std::uint64_t epoch) { buffer.epoch = epoch; }
template <typename T>
void mark_epoch(SessionBuffer<T>& buffer, std::uint64_t epoch) { buffer.epoch = epoch; }

void bind_nmm_state(DeviceFmmmSession& session) {
    session.nmm_state = FmmmNmmState{};
    session.nmm_state.position_x = session.position_x.d_ptr;
    session.nmm_state.position_y = session.position_y.d_ptr;
    session.nmm_state.force_x = session.repulsion_x.d_ptr;
    session.nmm_state.force_y = session.repulsion_y.d_ptr;
    session.nmm_state.edge_source = session.norm_edges_src.d_ptr;
    session.nmm_state.edge_target = session.norm_edges_tgt.d_ptr;
    session.nmm_state.edge_length = session.norm_edges_len.d_ptr;
    session.nmm_state.cell_downleft_x = session.hierarchy_down_left_x.d_ptr;
    session.nmm_state.cell_downleft_y = session.hierarchy_down_left_y.d_ptr;
    session.nmm_state.cell_box_length = session.hierarchy_box_length.d_ptr;
    session.nmm_state.cell_level = session.hierarchy_level.d_ptr;
    session.nmm_state.cell_parent = session.hierarchy_parent.d_ptr;
    session.nmm_state.cell_child4 = session.hierarchy_children4.d_ptr;
    session.nmm_state.cell_particle_count = session.hierarchy_particle_count.d_ptr;
    session.nmm_state.cell_center_x = session.hierarchy_center_x.d_ptr;
    session.nmm_state.cell_center_y = session.hierarchy_center_y.d_ptr;
    session.nmm_state.contained_begin = session.nmm_contained_begin.d_ptr;
    session.nmm_state.contained_nodes = session.nmm_contained_nodes.d_ptr;
    session.nmm_state.temp_contained_nodes = session.nmm_temp_contained_nodes.d_ptr;
    session.nmm_state.temp_contained_scratch = session.nmm_temp_contained_scratch.d_ptr;
    session.nmm_state.temp_contained_begin = session.nmm_temp_contained_begin.d_ptr;
    session.nmm_state.I_begin = session.nmm_I_begin.d_ptr;
    session.nmm_state.I_nodes = session.nmm_I_nodes.d_ptr;
    session.nmm_state.D1_begin = session.nmm_D1_begin.d_ptr;
    session.nmm_state.D1_nodes = session.nmm_D1_nodes.d_ptr;
    session.nmm_state.D2_begin = session.nmm_D2_begin.d_ptr;
    session.nmm_state.D2_nodes = session.nmm_D2_nodes.d_ptr;
    session.nmm_state.M_begin = session.nmm_M_begin.d_ptr;
    session.nmm_state.M_nodes = session.nmm_M_nodes.d_ptr;
    session.nmm_state.multipole_re = session.nmm_multipole_re.d_ptr;
    session.nmm_state.multipole_im = session.nmm_multipole_im.d_ptr;
    session.nmm_state.local_re = session.nmm_local_re.d_ptr;
    session.nmm_state.local_im = session.nmm_local_im.d_ptr;
    session.nmm_state.particle_to_leaf = session.nmm_particle_to_leaf.d_ptr;
    session.nmm_state.status = session.phase_status.d_ptr;
    session.nmm_state.device_build_counters = session.nmm_list_counts.d_ptr;
    session.nmm_state.device_build_counter_capacity =
        static_cast<int>(session.nmm_list_counts.capacity_elements);
    // The device builder uses this queue only during BFS, then reuses the
    // same allocation for the emitted M list.  No host staging is involved.
    session.nmm_state.device_build_queue = session.nmm_M_nodes.d_ptr;
    session.nmm_state.device_wsprl_queue = session.nmm_wsprl_queue.d_ptr;
    session.nmm_state.max_cells = static_cast<int>(session.hierarchy_down_left_x.capacity_elements);
    session.nmm_state.max_contained = static_cast<int>(session.nmm_contained_nodes.capacity_elements);
    session.nmm_state.max_list = static_cast<int>(session.nmm_M_nodes.capacity_elements);
    session.nmm_state.queue_capacity = static_cast<int>(session.nmm_M_nodes.capacity_elements);
    session.nmm_state.frontier_queue_capacity = static_cast<int>(session.nmm_M_nodes.capacity_elements);
    session.nmm_state.traversal_queue_capacity = static_cast<int>(session.nmm_wsprl_queue.capacity_elements);
    session.nmm_state.max_I = static_cast<int>(session.nmm_I_nodes.capacity_elements);
    session.nmm_state.max_D1 = static_cast<int>(session.nmm_D1_nodes.capacity_elements);
    session.nmm_state.max_D2 = static_cast<int>(session.nmm_D2_nodes.capacity_elements);
    session.nmm_state.max_M = static_cast<int>(session.nmm_M_nodes.capacity_elements);
    session.nmm_state.node_count = static_cast<int>(session.position_x.valid_elements);
    session.nmm_state.edge_count = session.normalized_edge_count;
    session.nmm_state.persistent_workspace = session.nmm_workspace;
    session.nmm_state.graph_epoch = session.graph_epoch;
    session.nmm_state.level_epoch = session.box_epoch;
    session.nmm_state.particle_order_epoch = session.position_epoch;
    session.nmm_state.tree_epoch = session.tree_epoch;
    session.nmm_state.map_epoch = session.tree_epoch;
}

bool allocate_all_state(DeviceFmmmSession& session, const RawFmmmInput& input) {
    const std::size_t n = input.nodes_x.size();
    const std::size_t m = input.edges_source.size();
    std::size_t cells = 0, lists = 0, expansion = 0, publication_bytes = 0;
    if (!resident_capacity_shape(input, cells, lists, expansion, publication_bytes)) return false;
    const std::size_t n1 = std::max<std::size_t>(1, n);
    const std::size_t m1 = std::max<std::size_t>(1, m);

#define ALLOC(field, capacity, valid, kind) \
    if (!allocate_session_buffer(session, session.field, capacity, valid, kind, \
                                 static_cast<std::uint32_t>(__LINE__))) return false

    ALLOC(raw_nodes_x, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(raw_nodes_y, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(raw_edges_src, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(raw_edges_tgt, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(raw_edges_layer, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(norm_edges_src, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(norm_edges_tgt, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(norm_edges_layer, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(norm_edges_len, m1, m, FmmmSessionBufferKind::graph);
    ALLOC(csr_offsets, n1 + 1, n + 1, FmmmSessionBufferKind::graph);
    ALLOC(csr_edges, std::max<std::size_t>(1, m * 2), m * 2, FmmmSessionBufferKind::graph);
    ALLOC(node_comp, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(compact_node_ids, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(local_node_ids, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(comp_offsets, n1 + 1, n + 1, FmmmSessionBufferKind::graph);
    ALLOC(comp_members, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(component_parent, n1, n, FmmmSessionBufferKind::graph);
    ALLOC(component_cursor, n1 + 1, n + 1, FmmmSessionBufferKind::graph);
    ALLOC(normalized_edge_count_device, 1, 1, FmmmSessionBufferKind::graph);
    ALLOC(component_count_device, 1, 1, FmmmSessionBufferKind::graph);
    ALLOC(graph_hash_device, 1, 1, FmmmSessionBufferKind::graph);

    ALLOC(hierarchy_down_left_x, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_down_left_y, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_box_length, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_center_x, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_center_y, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_parent, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_children4, cells * 4, n ? 4 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_level, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_compact_ids, n1, n, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_particle_count, cells, n ? 1 : 0, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_cell_count_device, 1, 1, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_root_count_device, 1, 1, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_particle_to_leaf, n1, n, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_tree_epoch_device, 1, 1, FmmmSessionBufferKind::hierarchy);
    ALLOC(hierarchy_position_epoch_device, 1, 1, FmmmSessionBufferKind::hierarchy);

    ALLOC(nmm_particle_to_leaf, n1, n, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_contained_begin, cells + 1, n ? 2 : 1, FmmmSessionBufferKind::nmm);
    // A subdividing tree stores a particle in every visited leaf during the
    // device build.  Reserve the same bounded list capacity used by the
    // relation lists so the tree builder never needs a phase allocation.
    ALLOC(nmm_contained_nodes, lists, n, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_temp_contained_nodes, lists, n, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_temp_contained_scratch, lists, n, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_temp_contained_begin, cells + 1, cells + 1, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_I_begin, cells + 1, n ? 2 : 1, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_I_nodes, lists, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_D1_begin, cells + 1, n ? 2 : 1, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_D1_nodes, lists, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_D2_begin, cells + 1, n ? 2 : 1, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_D2_nodes, lists, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_M_begin, cells + 1, n ? 2 : 1, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_M_nodes, lists, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_wsprl_queue, lists, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_multipole_re, expansion, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_multipole_im, expansion, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_local_re, expansion, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_local_im, expansion, 0, FmmmSessionBufferKind::nmm);
    ALLOC(nmm_list_counts, kNmmBuildStorageCount, kNmmBuildStorageCount,
          FmmmSessionBufferKind::nmm);

    ALLOC(position_x, n1, n, FmmmSessionBufferKind::force);
    ALLOC(position_y, n1, n, FmmmSessionBufferKind::force);
    ALLOC(position_x_next, n1, n, FmmmSessionBufferKind::force);
    ALLOC(position_y_next, n1, n, FmmmSessionBufferKind::force);
    ALLOC(force_x, n1, n, FmmmSessionBufferKind::force);
    ALLOC(force_y, n1, n, FmmmSessionBufferKind::force);
    ALLOC(attraction_x, n1, n, FmmmSessionBufferKind::force);
    ALLOC(attraction_y, n1, n, FmmmSessionBufferKind::force);
    ALLOC(repulsion_x, n1, n, FmmmSessionBufferKind::force);
    ALLOC(repulsion_y, n1, n, FmmmSessionBufferKind::force);
    ALLOC(movement_x, n1, n, FmmmSessionBufferKind::force);
    ALLOC(movement_y, n1, n, FmmmSessionBufferKind::force);
    ALLOC(force_status, n1, n, FmmmSessionBufferKind::force);
    ALLOC(phase_status, 1, 1, FmmmSessionBufferKind::force);
    ALLOC(iteration_counter_device, 1, 1, FmmmSessionBufferKind::force);
    ALLOC(phase_counter_device, 1, 1, FmmmSessionBufferKind::force);

    ALLOC(box_length_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(down_left_x_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(down_left_y_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(ideal_length_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(resize_scale_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(resize_status_device, 1, 1, FmmmSessionBufferKind::resize);
    ALLOC(box_epoch_device, 1, 1, FmmmSessionBufferKind::resize);

    ALLOC(edge_crossings, m1, m, FmmmSessionBufferKind::score);
    ALLOC(score_K_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(score_nK_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(score_Phi_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(score_C_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(admission_status_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(score_status_device, 1, 1, FmmmSessionBufferKind::score);
    ALLOC(score_state, 1, 1, FmmmSessionBufferKind::score);

    ALLOC(phase_records, kPhaseCount, kPhaseCount, FmmmSessionBufferKind::telemetry);
    ALLOC(transfer_ledger, kTransferLedgerCapacity, 0, FmmmSessionBufferKind::telemetry);
    ALLOC(publication_header, 1, 1, FmmmSessionBufferKind::publication);
    ALLOC(publication_backing, publication_bytes, publication_bytes, FmmmSessionBufferKind::publication);

#undef ALLOC
    return true;
}

void release_all_state(DeviceFmmmSession& session) noexcept {
#define RELEASE(field) release_session_buffer(session, session.field, \
                                              static_cast<std::uint32_t>(__LINE__))
    RELEASE(raw_nodes_x); RELEASE(raw_nodes_y); RELEASE(raw_edges_src);
    RELEASE(raw_edges_tgt); RELEASE(raw_edges_layer); RELEASE(norm_edges_src);
    RELEASE(norm_edges_tgt); RELEASE(norm_edges_layer); RELEASE(norm_edges_len);
    RELEASE(csr_offsets); RELEASE(csr_edges); RELEASE(node_comp);
    RELEASE(compact_node_ids); RELEASE(local_node_ids); RELEASE(comp_offsets);
    RELEASE(comp_members); RELEASE(component_parent); RELEASE(component_cursor);
    RELEASE(normalized_edge_count_device); RELEASE(component_count_device);
    RELEASE(graph_hash_device);
    RELEASE(hierarchy_down_left_x); RELEASE(hierarchy_down_left_y);
    RELEASE(hierarchy_box_length); RELEASE(hierarchy_center_x);
    RELEASE(hierarchy_center_y); RELEASE(hierarchy_parent); RELEASE(hierarchy_children4);
    RELEASE(hierarchy_level); RELEASE(hierarchy_compact_ids);
    RELEASE(hierarchy_particle_count); RELEASE(hierarchy_cell_count_device);
    RELEASE(hierarchy_root_count_device); RELEASE(hierarchy_particle_to_leaf);
    RELEASE(hierarchy_tree_epoch_device); RELEASE(hierarchy_position_epoch_device);
    RELEASE(nmm_particle_to_leaf); RELEASE(nmm_contained_begin);
    RELEASE(nmm_contained_nodes); RELEASE(nmm_temp_contained_nodes);
    RELEASE(nmm_temp_contained_scratch); RELEASE(nmm_temp_contained_begin);
    RELEASE(nmm_I_begin); RELEASE(nmm_I_nodes);
    RELEASE(nmm_D1_begin); RELEASE(nmm_D1_nodes); RELEASE(nmm_D2_begin);
    RELEASE(nmm_D2_nodes); RELEASE(nmm_M_begin); RELEASE(nmm_M_nodes);
    RELEASE(nmm_wsprl_queue);
    RELEASE(nmm_multipole_re); RELEASE(nmm_multipole_im); RELEASE(nmm_local_re);
    RELEASE(nmm_local_im); RELEASE(nmm_list_counts);
    RELEASE(position_x); RELEASE(position_y); RELEASE(position_x_next);
    RELEASE(position_y_next); RELEASE(force_x); RELEASE(force_y);
    RELEASE(attraction_x); RELEASE(attraction_y);
    RELEASE(repulsion_x); RELEASE(repulsion_y); RELEASE(movement_x);
    RELEASE(movement_y); RELEASE(force_status); RELEASE(phase_status);
    RELEASE(iteration_counter_device); RELEASE(phase_counter_device);
    RELEASE(box_length_device); RELEASE(down_left_x_device); RELEASE(down_left_y_device);
    RELEASE(ideal_length_device); RELEASE(resize_scale_device);
    RELEASE(resize_status_device); RELEASE(box_epoch_device);
    RELEASE(edge_crossings); RELEASE(score_K_device); RELEASE(score_nK_device);
    RELEASE(score_Phi_device); RELEASE(score_C_device); RELEASE(admission_status_device);
    RELEASE(score_status_device); RELEASE(score_state);
    RELEASE(phase_records); RELEASE(transfer_ledger); RELEASE(publication_header);
    RELEASE(publication_backing);
#undef RELEASE
}

} // namespace

#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
bool fmmm_native_phase_timing_create(DeviceFmmmSession& session) {
    for (int phase = 0; phase < kFmmmSemanticPhaseCount; ++phase) {
        if (cudaEventCreate(&session.native_phase_timing.start[phase]) != cudaSuccess ||
            cudaEventCreate(&session.native_phase_timing.end[phase]) != cudaSuccess) {
            fmmm_native_phase_timing_destroy(session);
            return false;
        }
    }
    return true;
}

void fmmm_native_phase_timing_destroy(DeviceFmmmSession& session) noexcept {
    for (int phase = 0; phase < kFmmmSemanticPhaseCount; ++phase) {
        if (session.native_phase_timing.start[phase]) {
            cudaEventDestroy(session.native_phase_timing.start[phase]);
            session.native_phase_timing.start[phase] = nullptr;
        }
        if (session.native_phase_timing.end[phase]) {
            cudaEventDestroy(session.native_phase_timing.end[phase]);
            session.native_phase_timing.end[phase] = nullptr;
        }
        session.native_phase_timing.started[phase] = false;
        session.native_phase_timing.recorded[phase] = false;
    }
}

bool fmmm_native_phase_timing_begin(DeviceFmmmSession& session,
                                    FmmmSemanticPhase phase,
                                    cudaStream_t stream) {
    const int index = static_cast<int>(phase);
    if (index < 0 || index >= kFmmmSemanticPhaseCount) return false;
    if (cudaEventRecord(session.native_phase_timing.start[index], stream) != cudaSuccess)
        return false;
    session.native_phase_timing.started[index] = true;
    session.native_phase_timing.recorded[index] = false;
    return true;
}

bool fmmm_native_phase_timing_end(DeviceFmmmSession& session,
                                  FmmmSemanticPhase phase,
                                  cudaStream_t stream) {
    const int index = static_cast<int>(phase);
    if (index < 0 || index >= kFmmmSemanticPhaseCount ||
        !session.native_phase_timing.started[index]) return false;
    if (cudaEventRecord(session.native_phase_timing.end[index], stream) != cudaSuccess)
        return false;
    session.native_phase_timing.recorded[index] = true;
    return true;
}
#endif

DeviceFmmmSession* fmmm_session_create(const RawFmmmInput& input, const FmmmConfig& config) {
    g_last_create_error = validate_input(input, config);
    if (std::string(g_last_create_error) != "ok") return nullptr;

    std::size_t estimated_bytes = 0;
    if (!estimate_resident_bytes(input, estimated_bytes)) {
        g_last_create_error = "allocation_limit";
        return nullptr;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && estimated_bytes > free_bytes) {
        g_last_create_error = "allocation_limit";
        return nullptr;
    }

    DeviceFmmmSession* session = nullptr;
    try {
        session = new DeviceFmmmSession();
        session->owner_id = kSessionOwnerId;
        session->config = config;
        session->max_width = input.max_width;
        session->max_height = input.max_height;
        session->host_transfer_records.reserve(kTransferLedgerCapacity);
        session->host_nodes_x = input.nodes_x;
        session->host_nodes_y = input.nodes_y;
        session->host_edges_src = input.edges_source;
        session->host_edges_tgt = input.edges_target;
        session->host_edges_layer = input.edges_layer;
        session->nmm_workspace = fmmm_nmm_workspace_create();
        if (!session->nmm_workspace || !allocate_all_state(*session, input))
            throw std::bad_alloc();

        FmmmWorkspaceShape shape;
        shape.node_count = std::max<std::size_t>(1, input.nodes_x.size());
        shape.edge_count = std::max<std::size_t>(1, input.edges_source.size());
        shape.level_count = 1;
        shape.cell_capacity = std::max<std::size_t>(1, input.nodes_x.size() * 15 + 1000);
        shape.max_level_nodes = shape.node_count;
        shape.max_level_edges = shape.edge_count;
        shape.max_aux_entries = shape.node_count + shape.edge_count;
        shape.low_adj_count = shape.max_aux_entries;
        shape.high_adj_count = shape.max_aux_entries;
        shape.lambda_count = shape.max_aux_entries;
        shape.neighbour_count = shape.max_aux_entries;
        shape.moon_count = shape.max_aux_entries;
        if (session->workspace.ensure_capacity(shape) != FmmmPipelineStatus::ok)
            throw std::bad_alloc();
        if (!create_timing_events(*session))
            throw std::bad_alloc();
#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
        if (!fmmm_native_phase_timing_create(*session))
            throw std::bad_alloc();
#endif
        session->phase_allocation_groups = 1;
        g_last_create_error = "ok";
        return session;
    } catch (...) {
        if (session) {
#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
            fmmm_native_phase_timing_destroy(*session);
#endif
            destroy_timing_events(*session);
            release_all_state(*session);
            session->workspace.release();
            fmmm_nmm_workspace_release(session->nmm_workspace);
            delete session;
        }
        g_last_create_error = "allocation_limit";
        return nullptr;
    }
}

const char* fmmm_session_last_create_error() { return g_last_create_error; }

void fmmm_session_destroy(DeviceFmmmSession& session) {
#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
    fmmm_native_phase_timing_destroy(session);
#endif
    destroy_timing_events(session);
    release_all_state(session);
    session.workspace.release();
    fmmm_nmm_workspace_release(session.nmm_workspace);
    session.nmm_workspace = nullptr;
    session.phase_free_groups++;
    delete &session;
}

void fmmm_session_enable_reuse(DeviceFmmmSession& session) {
    // This is deliberately an internal/native-only switch.  It must be set
    // before the first upload so the immutable host input remains available
    // for a later re-arm.  No Python/public fmmm_seed path calls it.
    session.reuse_enabled = true;
}

bool fmmm_session_reset_for_reuse(DeviceFmmmSession& session) {
    if (!session.reuse_enabled || session.host_nodes_x.empty() ||
        session.host_nodes_x.size() != session.host_nodes_y.size() ||
        session.host_edges_src.size() != session.host_edges_tgt.size() ||
        session.host_edges_src.size() != session.host_edges_layer.size()) {
        return false;
    }

    // The caller must reset only after the previous publication has been
    // downloaded.  Querying the final event keeps this helper fail-closed
    // without inserting an unconditional device synchronize into the
    // measured reusable call.
    if (session.final_download_timing_end != nullptr &&
        cudaEventQuery(session.final_download_timing_end) != cudaSuccess) {
        return false;
    }

    session.input_uploaded = false;
    session.normalized = false;
    session.hierarchy_built = false;
    session.nmm_built = false;
    session.schedule_submitted = false;
    session.score_complete = false;
    session.publication_ready = false;
    session.graph_epoch = 0;
    session.position_epoch = 0;
    session.tree_epoch = 0;
    session.box_epoch = 0;
    session.phase_generation_at_entry = session.allocation_generation;
    session.normalized_edge_count = 0;
    session.component_count = 0;
    session.nmm_state = FmmmNmmState{};

    // Per-call audit/timing state is reset, while allocation_generation and
    // all device pointers remain untouched.  The next upload reconstructs
    // the same graph state into the existing allocation set.
    session.host_transfer_records.clear();
    session.host_transfer_records.reserve(kTransferLedgerCapacity);
    session.transfer_sequence = 0;
    session.transfer_audit_failed = false;
    session.transfer_ledger.valid_elements = 0;
    session.input_upload_groups = 0;
    session.intermediate_h2d_groups = 0;
    session.intermediate_d2h_groups = 0;
    session.device_to_device_groups = 0;
    session.host_iteration_poll_groups = 0;
    session.final_download_groups = 0;
    session.phase_allocation_groups = 0;
    session.phase_free_groups = 0;

    session.timing_stream = nullptr;
    session.upload_timing_recorded = false;
    session.device_timing_recorded = false;
    session.device_timing_start_pending = false;
    session.final_download_timing_recorded = false;
    session.final_sync_recorded = false;
    session.upload_ms = -1.0F;
    session.device_ms = -1.0F;
    session.final_download_ms = -1.0F;
#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
    for (int phase = 0; phase < kFmmmSemanticPhaseCount; ++phase) {
        session.native_phase_timing.started[phase] = false;
        session.native_phase_timing.recorded[phase] = false;
    }
#endif
    return true;
}

bool fmmm_session_rearm_for_seed(DeviceFmmmSession& session, std::uint64_t seed) {
    if (!session.reuse_enabled || !session.input_uploaded || !session.normalized) {
        return false;
    }
    if (session.final_download_timing_end != nullptr &&
        cudaEventQuery(session.final_download_timing_end) != cudaSuccess) {
        return false;
    }

    session.config.seed = seed;
    session.hierarchy_built = false;
    session.nmm_built = false;
    session.schedule_submitted = false;
    session.score_complete = false;
    session.publication_ready = false;
    session.position_epoch = 0;
    session.tree_epoch = 0;
    session.box_epoch = 0;
    session.publication_epoch = 0;
    session.nmm_state = FmmmNmmState{};
    session.phase_generation_at_entry = session.allocation_generation;

    session.host_transfer_records.clear();
    session.host_transfer_records.reserve(kTransferLedgerCapacity);
    session.transfer_sequence = 0;
    session.transfer_audit_failed = false;
    session.transfer_ledger.valid_elements = 0;
    session.intermediate_h2d_groups = 0;
    session.intermediate_d2h_groups = 0;
    session.device_to_device_groups = 0;
    session.host_iteration_poll_groups = 0;
    session.final_download_groups = 0;
    session.phase_allocation_groups = 0;
    session.phase_free_groups = 0;

    session.timing_stream = nullptr;
    // Keep upload_timing_recorded: graph H2D already happened for this session.
    session.device_timing_recorded = false;
    session.device_timing_start_pending = true;
    session.final_download_timing_recorded = false;
    session.final_sync_recorded = false;
    session.upload_ms = -1.0F;
    session.device_ms = -1.0F;
    session.final_download_ms = -1.0F;
#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
    for (int phase = 0; phase < kFmmmSemanticPhaseCount; ++phase) {
        session.native_phase_timing.started[phase] = false;
        session.native_phase_timing.recorded[phase] = false;
    }
#endif
    return true;
}

FmmmDeviceStatus fmmm_session_upload(DeviceFmmmSession& session, cudaStream_t stream) {
    if (session.input_uploaded) return FmmmDeviceStatus::internal_error;
    session.timing_stream = stream;
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::upload, stream);
    if (!session.timing_events_ready ||
        !record_timing_event(session.upload_timing_start, stream))
        return FmmmDeviceStatus::backend_error;
    auto copy = [&](void* destination, const void* source, std::size_t bytes,
                    std::uint32_t callsite, bool group_boundary) {
        if (!append_transfer_record(
                session, kPhaseUpload, FmmmTransferOperation::copy,
                FmmmTransferDirection::host_to_device, bytes,
                FmmmTransferSourceKind::host, FmmmTransferSourceKind::device,
                callsite, stream, group_boundary))
            return false;
        return bytes == 0 || cudaMemcpyAsync(destination, source, bytes,
                                             cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    if (copy(session.raw_nodes_x.d_ptr, session.host_nodes_x.data(),
             session.host_nodes_x.size() * sizeof(double), __LINE__, true) == false ||
        copy(session.raw_nodes_y.d_ptr, session.host_nodes_y.data(),
             session.host_nodes_y.size() * sizeof(double), __LINE__, false) == false ||
        copy(session.raw_edges_src.d_ptr, session.host_edges_src.data(),
             session.host_edges_src.size() * sizeof(int), __LINE__, false) == false ||
        copy(session.raw_edges_tgt.d_ptr, session.host_edges_tgt.data(),
             session.host_edges_tgt.size() * sizeof(int), __LINE__, false) == false ||
        copy(session.raw_edges_layer.d_ptr, session.host_edges_layer.data(),
             session.host_edges_layer.size() * sizeof(int), __LINE__, false) == false)
        return FmmmDeviceStatus::backend_error;

    set_double_kernel<<<1, 1, 0, stream>>>(session.box_length_device.d_ptr,
                                            static_cast<double>(std::max(session.max_width, session.max_height)));
    set_double_kernel<<<1, 1, 0, stream>>>(session.down_left_x_device.d_ptr, 0.0);
    set_double_kernel<<<1, 1, 0, stream>>>(session.down_left_y_device.d_ptr, 0.0);
    set_double_kernel<<<1, 1, 0, stream>>>(session.ideal_length_device.d_ptr, 1.0);
    set_double_kernel<<<1, 1, 0, stream>>>(session.resize_scale_device.d_ptr, 1.0);
    set_u64_kernel<<<1, 1, 0, stream>>>(session.box_epoch_device.d_ptr, 0);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;

    // Seed the device ledger together with the one initial upload group.  The
    // self-record is staged before the copy so the device ledger contains a
    // complete, auditable prefix without a second host transfer later.
    const std::size_t ledger_bytes =
        (session.host_transfer_records.size() + 1) * sizeof(DeviceFmmmTransferRecord);
    if (!append_transfer_record(
            session, kPhaseUpload, FmmmTransferOperation::copy,
            FmmmTransferDirection::host_to_device, ledger_bytes,
            FmmmTransferSourceKind::host, FmmmTransferSourceKind::session,
            __LINE__, false) ||
        cudaMemcpyAsync(session.transfer_ledger.d_ptr,
                        session.host_transfer_records.data(), ledger_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    session.transfer_ledger.valid_elements = session.host_transfer_records.size();
    session.input_uploaded = true;
    session.input_upload_groups = 1;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::upload, stream);
    if (!launch_phase_record(session, kPhaseUpload, FmmmDeviceStatus::ok,
                             0, 0, session.raw_nodes_x.valid_elements,
                             session.raw_edges_src.valid_elements, stream))
        return FmmmDeviceStatus::backend_error;
    if (!record_timing_event(session.upload_timing_end, stream) ||
        !record_timing_event(session.device_timing_start, stream))
        return FmmmDeviceStatus::backend_error;
    session.upload_timing_recorded = true;
    if (!session.reuse_enabled) {
        session.host_nodes_x.clear();
        session.host_nodes_y.clear();
        session.host_edges_src.clear();
        session.host_edges_tgt.clear();
        session.host_edges_layer.clear();
    }
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_normalize(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.input_uploaded || session.normalized) return FmmmDeviceStatus::internal_error;
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::normalize, stream);
    const int n = static_cast<int>(session.raw_nodes_x.valid_elements);
    const int m = static_cast<int>(session.raw_edges_src.valid_elements);
    session.norm_edges_src.valid_elements = m;
    session.norm_edges_tgt.valid_elements = m;
    session.norm_edges_layer.valid_elements = m;
    session.norm_edges_len.valid_elements = m;

    if (m > 0) {
        const std::size_t bytes = static_cast<std::size_t>(m) * sizeof(int);
        auto copy_d2d = [&](void* destination, const void* source,
                            std::uint32_t callsite, bool group_boundary) {
            if (!append_transfer_record(
                    session, kPhaseNormalize, FmmmTransferOperation::copy,
                    FmmmTransferDirection::device_to_device, bytes,
                    FmmmTransferSourceKind::device, FmmmTransferSourceKind::device,
                    callsite, stream, group_boundary))
                return false;
            return cudaMemcpyAsync(destination, source, bytes,
                                   cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
        };
        if (!copy_d2d(session.norm_edges_src.d_ptr, session.raw_edges_src.d_ptr,
                      __LINE__, true) ||
            !copy_d2d(session.norm_edges_tgt.d_ptr, session.raw_edges_tgt.d_ptr,
                      __LINE__, false) ||
            !copy_d2d(session.norm_edges_layer.d_ptr, session.raw_edges_layer.d_ptr,
                      __LINE__, false))
            return FmmmDeviceStatus::backend_error;
        preserve_edges_kernel<<<(m + 127) / 128, 128, 0, stream>>>(m, session.norm_edges_len.d_ptr);
    }
    set_int_kernel<<<1, 1, 0, stream>>>(session.normalized_edge_count_device.d_ptr, m);
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::normalize, stream);
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::components, stream);
    set_int_kernel<<<1, 1, 0, stream>>>(session.component_count_device.d_ptr, 0);
    if (n > 0) {
        cc_init<<<(n + 127) / 128, 128, 0, stream>>>(n, session.component_parent.d_ptr);
        if (m > 0) cc_hook<<<(m + 127) / 128, 128, 0, stream>>>(
            m, session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr,
            session.component_parent.d_ptr);
        cc_flatten<<<(n + 127) / 128, 128, 0, stream>>>(n, session.component_parent.d_ptr);
        cc_assign_ids_serial<<<1, 1, 0, stream>>>(n, session.component_parent.d_ptr,
                                                   session.node_comp.d_ptr,
                                                   session.component_count_device.d_ptr);
        cc_populate_members_serial<<<1, 1, 0, stream>>>(
            n, session.component_parent.d_ptr, session.node_comp.d_ptr,
            session.component_count_device.d_ptr, session.comp_members.d_ptr,
            session.comp_offsets.d_ptr, session.component_cursor.d_ptr,
            session.local_node_ids.d_ptr);
        identity_kernel<<<(n + 127) / 128, 128, 0, stream>>>(n, session.compact_node_ids.d_ptr);
    }
    cudaMemsetAsync(session.csr_offsets.d_ptr, 0,
                    (n + 1) * sizeof(int), stream);
    graph_hash_kernel<<<1, 1, 0, stream>>>(
        n, m, session.raw_nodes_x.d_ptr, session.raw_nodes_y.d_ptr,
        session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr,
        session.norm_edges_layer.d_ptr, session.graph_hash_device.d_ptr);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;

    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::components, stream);

    session.graph_epoch = 1;
    session.normalized_edge_count = m;
    session.component_count = 0;
    session.normalized = true;
    mark_epoch(session.norm_edges_src, session.graph_epoch);
    mark_epoch(session.norm_edges_tgt, session.graph_epoch);
    mark_epoch(session.norm_edges_layer, session.graph_epoch);
    if (!launch_phase_record(session, kPhaseNormalize, FmmmDeviceStatus::ok,
                             session.graph_epoch, session.graph_epoch, n, m, stream) ||
        !launch_phase_record(session, kPhaseComponents, FmmmDeviceStatus::ok,
                             session.graph_epoch, session.graph_epoch, n, n, stream))
        return FmmmDeviceStatus::backend_error;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_build_hierarchy(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.normalized || session.hierarchy_built) return FmmmDeviceStatus::internal_error;
    session.timing_stream = stream;
    if (session.device_timing_start_pending) {
        if (!session.timing_events_ready ||
            !record_timing_event(session.device_timing_start, stream)) {
            return FmmmDeviceStatus::backend_error;
        }
        session.device_timing_start_pending = false;
    }
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::hierarchy, stream);
    const int n = static_cast<int>(session.raw_nodes_x.valid_elements);
    if (n > 0) {
        seed_positions_kernel<<<(n + 127) / 128, 128, 0, stream>>>(
            n, session.raw_nodes_x.d_ptr, session.raw_nodes_y.d_ptr,
            session.position_x.d_ptr, session.position_y.d_ptr,
            session.max_width, session.max_height, session.config.seed);
    }
    const int cell_capacity = static_cast<int>(
        std::min<std::size_t>(session.hierarchy_down_left_x.capacity_elements,
                              static_cast<std::size_t>(std::numeric_limits<int>::max())));
    int depth = 0;
    int materialized_cells = 1;
    while (depth < 6) {
        const long long next_cells = static_cast<long long>(materialized_cells) * 4LL + 1LL;
        if (next_cells > cell_capacity) break;
        materialized_cells = static_cast<int>(next_cells);
        ++depth;
    }
    FmmmDeviceHierarchyInput hierarchy_input{};
    hierarchy_input.position_x = session.position_x.d_ptr;
    hierarchy_input.position_y = session.position_y.d_ptr;
    hierarchy_input.node_count = n;
    hierarchy_input.down_left_x = 0.0;
    hierarchy_input.down_left_y = 0.0;
    hierarchy_input.box_length = static_cast<double>(std::max(session.max_width,
                                                               session.max_height));
    hierarchy_input.position_epoch = 1;
    hierarchy_input.box_epoch = 1;
    FmmmDeviceHierarchyLaunchConfig hierarchy_config{};
    hierarchy_config.cell_capacity = cell_capacity;
    hierarchy_config.max_depth = depth;
    hierarchy_config.leaf_capacity = 32;
    hierarchy_config.tree_epoch = 1;
    FmmmDeviceHierarchyOutput hierarchy_output{};
    hierarchy_output.cell_down_left_x = session.hierarchy_down_left_x.d_ptr;
    hierarchy_output.cell_down_left_y = session.hierarchy_down_left_y.d_ptr;
    hierarchy_output.cell_box_length = session.hierarchy_box_length.d_ptr;
    hierarchy_output.cell_center_x = session.hierarchy_center_x.d_ptr;
    hierarchy_output.cell_center_y = session.hierarchy_center_y.d_ptr;
    hierarchy_output.cell_parent = session.hierarchy_parent.d_ptr;
    hierarchy_output.cell_children4 = session.hierarchy_children4.d_ptr;
    hierarchy_output.cell_level = session.hierarchy_level.d_ptr;
    hierarchy_output.cell_compact_id = session.hierarchy_compact_ids.d_ptr;
    hierarchy_output.cell_particle_count = session.hierarchy_particle_count.d_ptr;
    hierarchy_output.particle_to_leaf = session.hierarchy_particle_to_leaf.d_ptr;
    hierarchy_output.cell_count_device = session.hierarchy_cell_count_device.d_ptr;
    hierarchy_output.root_count_device = session.hierarchy_root_count_device.d_ptr;
    hierarchy_output.tree_epoch_device = session.hierarchy_tree_epoch_device.d_ptr;
    hierarchy_output.position_epoch_device = session.hierarchy_position_epoch_device.d_ptr;
    hierarchy_output.box_epoch_device = session.box_epoch_device.d_ptr;
    hierarchy_output.status_device = session.phase_status.d_ptr;
    const cudaError_t hierarchy_error = fmmm_launch_device_hierarchy(
        hierarchy_input, hierarchy_config, hierarchy_output, stream);
    if (hierarchy_error != cudaSuccess) return FmmmDeviceStatus::backend_error;
    cudaMemsetAsync(session.force_x.d_ptr, 0, session.force_x.capacity_elements * sizeof(double), stream);
    cudaMemsetAsync(session.force_y.d_ptr, 0, session.force_y.capacity_elements * sizeof(double), stream);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;

    session.position_epoch = 1;
    session.tree_epoch = 1;
    session.box_epoch = 1;
    session.hierarchy_built = true;
    session.nmm_built = false;
    session.schedule_submitted = false;
    mark_epoch(session.position_x, session.position_epoch);
    mark_epoch(session.position_y, session.position_epoch);
    mark_epoch(session.hierarchy_parent, session.tree_epoch);
    mark_epoch(session.hierarchy_children4, session.tree_epoch);
    bind_nmm_state(session);
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::hierarchy, stream);
    if (!launch_phase_record(session, kPhaseHierarchy, FmmmDeviceStatus::ok,
                             session.graph_epoch, session.tree_epoch, n, n, stream) ||
        cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_build_nmm(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.hierarchy_built || session.nmm_built) return FmmmDeviceStatus::internal_error;
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::nmm, stream);
    bind_nmm_state(session);
    const int n = static_cast<int>(session.position_x.valid_elements);
    if (n >= 175) {
        FmmmNmmOptions options{};
        options.box_length = static_cast<double>(std::max(session.max_width, session.max_height));
        options.down_left_x = 0.0;
        options.down_left_y = 0.0;
        options.precision = 4;
        options.particles_in_leaf = 25;
        options.using_nmm = true;
        options.seed = session.config.seed;
        const FmmmStageResult built = fmmm_nmm_build_async(session.nmm_state, options, stream);
        if (built.status != FmmmPipelineStatus::ok) return FmmmDeviceStatus::backend_error;
    } else {
        cudaMemsetAsync(session.nmm_list_counts.d_ptr, 0,
                        session.nmm_list_counts.capacity_elements * sizeof(int), stream);
    }
    if (cudaGetLastError() != cudaSuccess) return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::nmm, stream);
    session.nmm_built = true;
    if (!launch_phase_record(session, kPhaseNmm, FmmmDeviceStatus::ok,
                             session.tree_epoch, session.tree_epoch, n, n, stream) ||
        cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_run_schedule(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.nmm_built || session.schedule_submitted) return FmmmDeviceStatus::internal_error;
    const int n = static_cast<int>(session.position_x.valid_elements);
    const int blocks = std::max(1, (n + 127) / 128);
    const int m = static_cast<int>(session.norm_edges_src.valid_elements);
    auto record = [&](int phase, std::uint64_t entry, std::uint64_t exit,
                      std::uint64_t input_count, std::uint64_t output_count) {
        return launch_phase_record(session, phase, FmmmDeviceStatus::ok,
                                   entry, exit, input_count, output_count, stream);
    };

    // placement
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::placement, stream);
    device_placement_phase_kernel<<<blocks, 128, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.position_x_next.d_ptr, session.position_y_next.d_ptr,
        session.force_status.d_ptr, session.phase_status.d_ptr,
        session.max_width, session.max_height, kPhasePlacement);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::placement, stream);
    if (!record(kPhasePlacement, session.position_epoch, session.position_epoch, n, n))
        return FmmmDeviceStatus::backend_error;

    // NMM force is part of the normal-force phase and consumes the tree built
    // before placement.  It remains an async enqueue into the same session
    // stream; no tree or force buffer is copied through the host.
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::normal_force, stream);
    if (n >= 175) {
        FmmmNmmOptions options{};
        options.box_length = static_cast<double>(std::max(session.max_width, session.max_height));
        options.down_left_x = 0.0;
        options.down_left_y = 0.0;
        options.precision = 4;
        options.particles_in_leaf = 25;
        options.using_nmm = true;
        options.seed = session.config.seed;
        const FmmmStageResult forced = fmmm_nmm_force_async(session.nmm_state, options, stream);
        if (forced.status != FmmmPipelineStatus::ok) return FmmmDeviceStatus::backend_error;
    } else {
        cudaMemsetAsync(session.repulsion_x.d_ptr, 0,
                        session.repulsion_x.capacity_elements * sizeof(double), stream);
        cudaMemsetAsync(session.repulsion_y.d_ptr, 0,
                        session.repulsion_y.capacity_elements * sizeof(double), stream);
    }
    // Freeze the complete input geometry before the force kernel starts.  The
    // force kernel publishes x/y in-place, so using x/y as both its global
    // read set and write set is a cross-thread read-after-write race.
    if (cudaMemcpyAsync(session.position_x_next.d_ptr, session.position_x.d_ptr,
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(session.position_y_next.d_ptr, session.position_y.d_ptr,
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return FmmmDeviceStatus::backend_error;
    }
    device_normal_force_schedule_kernel<<<blocks, 128, 0, stream>>>(
        n, m, session.config.iterations, session.position_x.d_ptr,
        session.position_y.d_ptr, session.position_x_next.d_ptr,
        session.position_y_next.d_ptr, session.force_x.d_ptr, session.force_y.d_ptr,
        session.attraction_x.d_ptr, session.attraction_y.d_ptr,
        session.repulsion_x.d_ptr, session.repulsion_y.d_ptr,
        session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr,
        session.force_status.d_ptr, session.iteration_counter_device.d_ptr,
        session.phase_counter_device.d_ptr, session.phase_status.d_ptr,
        session.max_width, session.max_height, kPhaseNormalForce);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::normal_force, stream);
    if (!record(kPhaseNormalForce, session.position_epoch, session.position_epoch,
                n, n))
        return FmmmDeviceStatus::backend_error;

    // postprocess force
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::postprocess, stream);
    device_postprocess_force_kernel<<<blocks, 128, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.movement_x.d_ptr, session.movement_y.d_ptr,
        session.force_status.d_ptr, session.max_width, session.max_height);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::postprocess, stream);
    if (!record(kPhasePostprocess, session.position_epoch, session.position_epoch,
                n, n))
        return FmmmDeviceStatus::backend_error;

    // resize
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::resize, stream);
    device_resize_state_kernel<<<blocks, 128, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.box_length_device.d_ptr, session.down_left_x_device.d_ptr,
        session.down_left_y_device.d_ptr, session.ideal_length_device.d_ptr,
        session.resize_scale_device.d_ptr, session.resize_status_device.d_ptr,
        session.box_epoch_device.d_ptr, session.max_width, session.max_height);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::resize, stream);
    if (!record(kPhaseResize, session.position_epoch, session.box_epoch, n, n))
        return FmmmDeviceStatus::backend_error;

    // fine tuning
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::fine_tuning, stream);
    device_fine_tuning_phase_kernel<<<blocks, 128, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.position_x_next.d_ptr, session.position_y_next.d_ptr,
        session.movement_x.d_ptr, session.movement_y.d_ptr,
        session.force_status.d_ptr, session.phase_counter_device.d_ptr,
        session.phase_status.d_ptr, session.max_width, session.max_height,
        kPhaseFineTuning);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::fine_tuning, stream);
    if (!record(kPhaseFineTuning, session.position_epoch, session.position_epoch,
                n, n))
        return FmmmDeviceStatus::backend_error;

    // final resize
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::final_resize, stream);
    device_final_resize_phase_kernel<<<blocks, 128, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.movement_x.d_ptr, session.movement_y.d_ptr,
        session.resize_scale_device.d_ptr, session.resize_status_device.d_ptr,
        session.box_epoch_device.d_ptr, session.phase_status.d_ptr,
        session.max_width, session.max_height, kPhaseFinalResize);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::final_resize, stream);
    if (!record(kPhaseFinalResize, session.position_epoch, session.box_epoch, n, n))
        return FmmmDeviceStatus::backend_error;

    // Packing is the final device-side geometry boundary.  Double is the
    // default ABI; FP32 is an explicit, one-way boundary conversion.
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::packing, stream);
    device_fp32_packing_kernel<<<1, 1, 0, stream>>>(
        n, session.position_x.d_ptr, session.position_y.d_ptr,
        session.max_width, session.max_height,
        session.config.coordinate_type == FmmmConfig::CoordinateType::fp32,
        session.force_status.d_ptr);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::packing, stream);
    if (!record(kPhasePacking, session.position_epoch, session.box_epoch, n, n))
        return FmmmDeviceStatus::backend_error;

    session.schedule_submitted = true;
    session.position_epoch = 2;
    session.box_epoch = 2;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_close_layout_device_timing(DeviceFmmmSession& session,
                                                          cudaStream_t stream) {
    if (!session.schedule_submitted) return FmmmDeviceStatus::internal_error;
    if (session.device_timing_recorded) return FmmmDeviceStatus::ok;
    if (!session.timing_events_ready ||
        !record_timing_event(session.device_timing_end, stream) ||
        cudaEventSynchronize(session.device_timing_end) != cudaSuccess) {
        return FmmmDeviceStatus::backend_error;
    }
    session.device_timing_recorded = true;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_score_and_admit(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.schedule_submitted || session.score_complete)
        return FmmmDeviceStatus::internal_error;
    const int n = static_cast<int>(session.position_x.valid_elements);
    const int m = static_cast<int>(session.norm_edges_src.valid_elements);
    const bool use_fused_small_score = g4_small_fusion_enabled() && n > 0 &&
        n < 175 && m <= 256;
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::score, stream);
    if (m == 0) {
        empty_score_kernel<<<1, 1, 0, stream>>>(
            session.score_K_device.d_ptr, session.score_nK_device.d_ptr,
            session.score_Phi_device.d_ptr, session.score_C_device.d_ptr,
            session.admission_status_device.d_ptr, session.score_status_device.d_ptr,
            session.score_state.d_ptr);
    } else if (use_fused_small_score) {
        score_small_fused_kernel<<<1, g4_small_score_threads(m), 0, stream>>>(
            n, m, session.position_x.d_ptr, session.position_y.d_ptr,
            session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr,
            session.norm_edges_layer.d_ptr, session.config.layers_enabled ? 1 : 0,
            session.edge_crossings.d_ptr, session.score_state.d_ptr,
            session.force_status.d_ptr, session.score_K_device.d_ptr,
            session.score_nK_device.d_ptr, session.score_Phi_device.d_ptr,
            session.score_C_device.d_ptr, session.admission_status_device.d_ptr,
            session.score_status_device.d_ptr);
    } else {
        score_init_kernel<<<1, 1, 0, stream>>>(
            m, session.edge_crossings.d_ptr, session.score_state.d_ptr);
        const int threads = 256;
        const int crossing_blocks = std::max(1, std::min(65535, (m + threads - 1) / threads));
        session_crossing_kernel<<<crossing_blocks, threads, 0, stream>>>(
            n, m, session.position_x.d_ptr, session.position_y.d_ptr,
            session.norm_edges_src.d_ptr, session.norm_edges_tgt.d_ptr,
            session.norm_edges_layer.d_ptr, session.config.layers_enabled ? 1 : 0,
            session.edge_crossings.d_ptr, session.score_state.d_ptr);
        score_max_kernel<<<crossing_blocks, threads, 0, stream>>>(
            m, session.edge_crossings.d_ptr, session.score_state.d_ptr);
        score_accumulate_kernel<<<crossing_blocks, threads, 0, stream>>>(
            m, session.edge_crossings.d_ptr, session.score_state.d_ptr);
    }
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::score, stream);
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::objective, stream);
    if (m != 0 && !use_fused_small_score) {
        score_finalize_kernel<<<1, 1, 0, stream>>>(session.score_state.d_ptr);
    }
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::objective, stream);
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::admission, stream);
    if (m != 0) {
        score_guard_force_status_kernel<<<1, 1, 0, stream>>>(
            session.force_status.d_ptr, session.score_state.d_ptr);
        copy_score_state_kernel<<<1, 1, 0, stream>>>(
            session.score_state.d_ptr, session.score_K_device.d_ptr,
            session.score_nK_device.d_ptr, session.score_Phi_device.d_ptr,
            session.score_C_device.d_ptr, session.admission_status_device.d_ptr,
            session.score_status_device.d_ptr);
    }
    if (m == 0) {
        score_guard_force_status_kernel<<<1, 1, 0, stream>>>(
            session.force_status.d_ptr, session.score_state.d_ptr);
        copy_score_state_kernel<<<1, 1, 0, stream>>>(
            session.score_state.d_ptr, session.score_K_device.d_ptr,
            session.score_nK_device.d_ptr, session.score_Phi_device.d_ptr,
            session.score_C_device.d_ptr, session.admission_status_device.d_ptr,
            session.score_status_device.d_ptr);
    }
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::admission, stream);
    session.score_complete = true;
    const int status = m == 0 ? static_cast<int>(FmmmDeviceStatus::ok) :
        static_cast<int>(FmmmDeviceStatus::ok);
    if (!launch_phase_record(session, kPhaseScore, FmmmDeviceStatus::ok,
                             session.position_epoch, session.box_epoch, m, m, stream) ||
        !launch_phase_record(session, kPhaseAdmission, FmmmDeviceStatus::ok,
                             session.position_epoch, session.box_epoch, m, 1, stream) ||
        !launch_phase_record(session, kPhaseObjective, FmmmDeviceStatus::ok,
                             session.position_epoch, session.box_epoch, m, 4, stream))
        return FmmmDeviceStatus::backend_error;
    (void)status;
    return FmmmDeviceStatus::ok;
}

FmmmDeviceStatus fmmm_session_publish_device(DeviceFmmmSession& session, cudaStream_t stream) {
    if (!session.score_complete) return FmmmDeviceStatus::internal_error;
    const int n = static_cast<int>(session.position_x.valid_elements);
    const int m = static_cast<int>(session.edge_crossings.valid_elements);
    ++session.publication_epoch;
    FMMM_NATIVE_PHASE_BEGIN_OR_RETURN(session, FmmmSemanticPhase::publish, stream);
    cudaMemsetAsync(session.publication_backing.d_ptr, 0,
                    session.publication_backing.capacity_elements, stream);
    publish_kernel<<<1, 1, 0, stream>>>(
        session.publication_backing.d_ptr, session.publication_header.d_ptr,
        session.score_state.d_ptr, session.position_x.d_ptr, session.position_y.d_ptr,
        session.edge_crossings.d_ptr, n, m, static_cast<int>(FmmmDeviceStatus::ok), 0,
        session.graph_epoch, session.tree_epoch, session.publication_epoch,
        session.allocation_generation, static_cast<std::uint64_t>(kPhaseCount - 1),
        static_cast<std::uint64_t>(session.host_transfer_records.size()),
        session.publication_backing.capacity_elements);
    if (cudaGetLastError() != cudaSuccess)
        return FmmmDeviceStatus::backend_error;
    FMMM_NATIVE_PHASE_END_OR_RETURN(session, FmmmSemanticPhase::publish, stream);
    session.publication_ready = true;
    const bool layout_timing_closed = session.device_timing_recorded;
    if (!launch_phase_record(session, kPhasePublish, FmmmDeviceStatus::ok,
                             session.box_epoch, session.publication_epoch, n, n, stream) ||
        (append_phase_records_kernel<<<1, 1, 0, stream>>>(
             session.publication_backing.d_ptr, session.publication_header.d_ptr,
             session.phase_records.d_ptr, static_cast<std::uint64_t>(kPhaseCount - 1),
             session.publication_backing.capacity_elements),
         cudaGetLastError() != cudaSuccess) ||
        (!layout_timing_closed &&
         !record_timing_event(session.device_timing_end, stream)) ||
        !record_final_publication_synchronization(session, stream, __LINE__))
        return FmmmDeviceStatus::backend_error;
    session.device_timing_recorded = true;
    return FmmmDeviceStatus::ok;
}

std::string build_transfer_telemetry_json(
    const DeviceFmmmSession& session,
    const std::vector<DeviceFmmmPhaseRecord>& device_phase_records) {
    static constexpr const char* kPhaseNames[] = {
        "normalize", "components", "hierarchy", "nmm", "placement",
        "normal_force", "postprocess", "resize", "fine_tuning",
        "final_resize", "packing", "score", "admission", "objective", "publish"
    };
    std::string phase_order = "[";
    std::string phase_records = "[";
    bool emitted_phase = false;
    for (std::size_t i = 0; i < device_phase_records.size(); ++i) {
        const auto& phase_record = device_phase_records[i];
        const int phase_id = phase_record.phase_id;
        if (phase_id < kPhaseNormalize || phase_id > kPhasePublish) continue;
        const char* phase_name = kPhaseNames[phase_id - kPhaseNormalize];
        if (emitted_phase) {
            phase_order += ",";
            phase_records += ",";
        }
        emitted_phase = true;
        phase_order += "\"" + std::string(phase_name) + "\"";
        phase_records += "{\"phase\":\"" + std::string(phase_name) +
            "\",\"phase_id\":" + std::to_string(phase_record.phase_id) +
            ",\"execution\":\"device\"" +
            ",\"entry_epoch\":" + std::to_string(phase_record.entry_epoch) +
            ",\"exit_epoch\":" + std::to_string(phase_record.exit_epoch) +
            ",\"status\":" + std::to_string(phase_record.status) +
            ",\"error_code\":" + std::to_string(phase_record.error_code) +
            ",\"input_count\":" + std::to_string(phase_record.input_count) +
            ",\"output_count\":" + std::to_string(phase_record.output_count) +
            ",\"input_counts\":" + std::to_string(phase_record.input_count) +
            ",\"output_counts\":" + std::to_string(phase_record.output_count) +
            ",\"allocation_generation\":" +
            std::to_string(phase_record.allocation_generation) + "}";
    }
    phase_order += "]";
    phase_records += "]";

    std::string records = "[";
    std::size_t illegal_phase_sync_count = 0;
    std::size_t final_sync_count = 0;
    for (std::size_t i = 0; i < session.host_transfer_records.size(); ++i) {
        if (i != 0) records += ",";
        const auto& item = session.host_transfer_records[i];
        if (item.operation == static_cast<std::uint32_t>(FmmmTransferOperation::synchronize))
            ++illegal_phase_sync_count;
        if (item.operation == static_cast<std::uint32_t>(FmmmTransferOperation::final_synchronize))
            ++final_sync_count;
        records += "{\"sequence\":" + std::to_string(item.sequence) +
            ",\"phase_id\":" + std::to_string(item.phase_id) +
            ",\"operation\":" + std::to_string(item.operation) +
            ",\"direction\":" + std::to_string(item.direction) +
            ",\"bytes\":" + std::to_string(item.bytes) +
            ",\"source_kind\":" + std::to_string(item.source_kind) +
            ",\"destination_kind\":" + std::to_string(item.destination_kind) +
            ",\"operation_class\":\"" +
            (item.operation == static_cast<std::uint32_t>(FmmmTransferOperation::final_synchronize)
                 ? "final_publication_sync"
                 : item.operation == static_cast<std::uint32_t>(FmmmTransferOperation::synchronize)
                     ? "illegal_phase_sync" : "async_or_transfer") + "\"" +
            ",\"callsite_id\":" + std::to_string(item.callsite_id) +
            ",\"stream_id\":" + std::to_string(item.stream_id) +
            ",\"allocation_generation\":" +
            std::to_string(item.allocation_generation) + "}";
    }
    records += "]";

    const auto timing_value = [](float milliseconds) {
        return milliseconds >= 0.0F && std::isfinite(milliseconds)
            ? std::to_string(milliseconds) : std::string("null");
    };

    return "{\"phase_order\":" + phase_order +
        ",\"phase_records\":" + phase_records +
        ",\"ownership\":{\"graph\":\"session\",\"hierarchy\":\"session\",\"nmm\":\"session\",\"force\":\"session\",\"resize\":\"session\",\"score\":\"session\",\"publication\":\"session\",\"workspace\":\"session\"}"+
        ",\"epochs\":{\"graph\":" + std::to_string(session.graph_epoch) +
        ",\"hierarchy\":" + std::to_string(session.tree_epoch) +
        ",\"position\":" + std::to_string(session.position_epoch) +
        ",\"box\":" + std::to_string(session.box_epoch) +
        ",\"publication\":" + std::to_string(session.publication_epoch) + "}"+
        ",\"input_upload_groups\":" + std::to_string(session.input_upload_groups) +
        ",\"intermediate_h2d_groups\":" + std::to_string(session.intermediate_h2d_groups) +
        ",\"intermediate_d2h_groups\":" + std::to_string(session.intermediate_d2h_groups) +
        ",\"device_to_device_groups\":" + std::to_string(session.device_to_device_groups) +
        ",\"final_download_groups\":" + std::to_string(session.final_download_groups) +
        ",\"host_iteration_poll_groups\":" + std::to_string(session.host_iteration_poll_groups) +
        ",\"phase_allocation_groups\":0,\"phase_free_groups\":0"+
        ",\"host_sync_records\":" + std::to_string(illegal_phase_sync_count) +
        ",\"final_sync_records\":" + std::to_string(final_sync_count) +
        ",\"phase_sync_records\":" + std::to_string(illegal_phase_sync_count) +
        ",\"ledger_classified\":" + (session.transfer_audit_failed ? "false" : "true") +
        ",\"transfer_record_count\":" + std::to_string(session.host_transfer_records.size()) +
        ",\"upload_ms\":" + timing_value(session.upload_ms) +
        ",\"device_ms\":" + timing_value(session.device_ms) +
        ",\"final_download_ms\":" + timing_value(session.final_download_ms) +
        ",\"timing_source\":\"session_cuda_events\"" +
        ",\"publication_bounded\":true" +
        ",\"transfer_records\":" + records + "}";
}

DeviceFmmmPublication fmmm_session_download_publication(DeviceFmmmSession& session) {
    DeviceFmmmPublication publication;
    if (!session.publication_ready || !session.publication_backing.d_ptr) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "publication_not_ready";
        return publication;
    }
    const cudaStream_t stream = session.timing_stream;
    std::vector<std::uint8_t> host(session.publication_backing.valid_elements, 0);
    if (!session.final_sync_recorded &&
        !record_final_publication_synchronization(session, stream, __LINE__)) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "final_sync_record_failed";
        return publication;
    }
    if (!append_transfer_record(
            session, kPhasePublish, FmmmTransferOperation::copy,
            FmmmTransferDirection::device_to_host,
            session.publication_backing.valid_elements,
            FmmmTransferSourceKind::device, FmmmTransferSourceKind::host,
            __LINE__, false)) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "transfer_ledger_overflow";
        return publication;
    }
    if (!record_timing_event(session.final_download_timing_start, stream) ||
        cudaMemcpyAsync(host.data(), session.publication_backing.d_ptr, host.size(),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        !record_timing_event(session.final_download_timing_end, stream) ||
        cudaEventSynchronize(session.final_download_timing_end) != cudaSuccess) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "publication_download_failed";
        return publication;
    }
    session.final_download_timing_recorded = true;
    ++session.final_download_groups;
    const bool timing_ok =
        (!session.upload_timing_recorded ||
         resolve_timing_ms(session, session.upload_timing_start,
                           session.upload_timing_end, session.upload_ms)) &&
        (!session.device_timing_recorded ||
         resolve_timing_ms(session, session.device_timing_start,
                           session.device_timing_end, session.device_ms)) &&
        resolve_timing_ms(session, session.final_download_timing_start,
                          session.final_download_timing_end,
                          session.final_download_ms);
    if (!timing_ok) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "timing_readback_failed";
        publication.admitted = false;
    }
    DeviceFmmmPublicationRecord record{};
    if (host.size() < sizeof(record)) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "publication_truncated";
        return publication;
    }
    std::memcpy(&record, host.data(), sizeof(record));
    publication.graph_epoch = record.graph_epoch;
    publication.tree_epoch = record.tree_epoch;
    publication.publication_epoch = record.publication_epoch;
    publication.allocation_generation = record.allocation_generation;
    publication.coordinate_type = static_cast<FmmmPublicationCoordinateType>(
        record.coordinate_type);
    publication.K = record.K;
    publication.nK = record.nK;
    publication.Phi = record.Phi;
    publication.C = record.C;
    publication.admitted = record.admitted != 0;
    publication.status = record.status == static_cast<int>(FmmmDeviceStatus::ok)
        ? "OK" : "BACKEND_ERROR";
    publication.reason = record.reason_code == 0 ? "none" : "session_score_or_schedule_pending";
    std::vector<DeviceFmmmPhaseRecord> device_phase_records;
    if (record.phase_count > static_cast<std::uint64_t>(kPhaseCount - 1)) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "phase_record_count_overflow";
        publication.admitted = false;
        publication.telemetry_json = build_transfer_telemetry_json(session, device_phase_records);
        return publication;
    }
    // Keep telemetry available even when the bounded publication payload
    // fails a schema check; the caller must still receive the audited device
    // failure publication rather than a host-side exception.
    std::size_t payload_bytes = 0;
    if (!checked_publication_payload_bytes(
            publication.admitted ? static_cast<std::size_t>(record.node_count) : 0,
            publication.admitted ? static_cast<std::size_t>(record.edge_count) : 0,
            payload_bytes)) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "publication_size_overflow";
        publication.admitted = false;
        publication.telemetry_json = build_transfer_telemetry_json(session, device_phase_records);
        return publication;
    }
    const std::size_t phase_offset = sizeof(record) + payload_bytes;
    const std::size_t phase_bytes =
        static_cast<std::size_t>(record.phase_count) * sizeof(DeviceFmmmPhaseRecord);
    if (phase_offset > host.size() || phase_bytes > host.size() - phase_offset) {
        publication.status = "BACKEND_ERROR";
        publication.reason = "phase_record_payload_truncated";
        publication.admitted = false;
        publication.telemetry_json = build_transfer_telemetry_json(session, device_phase_records);
        return publication;
    }
    device_phase_records.resize(static_cast<std::size_t>(record.phase_count));
    if (!device_phase_records.empty()) {
        std::memcpy(device_phase_records.data(), host.data() + phase_offset, phase_bytes);
    }
    publication.telemetry_json = build_transfer_telemetry_json(session, device_phase_records);
    if (publication.admitted) {
        const std::size_t n = static_cast<std::size_t>(record.node_count);
        const std::size_t m = static_cast<std::size_t>(record.edge_count);
        if (publication.coordinate_type != FmmmPublicationCoordinateType::double_precision) {
            publication.status = "BACKEND_ERROR";
            publication.reason = "unsupported_coordinate_type";
            publication.admitted = false;
            return publication;
        }
        std::size_t required = 0;
        if (!checked_publication_total_bytes(
                n, m, static_cast<std::size_t>(record.phase_count), required)) {
            publication.status = "BACKEND_ERROR";
            publication.reason = "publication_size_overflow";
            publication.admitted = false;
            return publication;
        }
        if (required > host.size()) {
            publication.status = "BACKEND_ERROR";
            publication.reason = "publication_truncated";
            publication.admitted = false;
            return publication;
        }
        const auto* values = reinterpret_cast<const double*>(host.data() + sizeof(record));
        publication.coordinates.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            publication.coordinates.emplace_back(values[i], values[n + i]);
        const auto* crossing_values = reinterpret_cast<const std::int64_t*>(
            host.data() + sizeof(record) + (2 * n) * sizeof(double));
        publication.edge_crossings.assign(crossing_values, crossing_values + m);
    }
    return publication;
}

} // namespace lcn::release
