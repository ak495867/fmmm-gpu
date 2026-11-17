#pragma once

#include "fmmm_workspace.h"
#include "fmmm_nmm_packing.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace lcn::oracle {
enum class FmmmFinalStatus;
}

namespace lcn::release {

// Compatibility alias for the legacy coordinator translation unit. The
// session ABI itself does not depend on the coordinator's final-status type.
using FmmmFinalStatus = ::lcn::oracle::FmmmFinalStatus;

struct RawFmmmInput {
    // Coordinates are continuous layout data.  Topology identifiers and
    // objective counters remain integral, but a seed must never be coerced
    // through an integer lattice at this ABI boundary.
    std::vector<double> nodes_x;
    std::vector<double> nodes_y;
    std::vector<int> edges_source;
    std::vector<int> edges_target;
    std::vector<int> edges_layer;
    std::int64_t max_width = 0;
    std::int64_t max_height = 0;
};

struct FmmmConfig {
    std::uint64_t seed = 0;
    int iterations = 3;
    bool layers_enabled = false;
    std::string config_string;
    enum class CoordinateType : std::uint32_t { double_precision = 0, fp32 = 1 };
    CoordinateType coordinate_type = CoordinateType::double_precision;
};

enum class FmmmDeviceStatus {
    ok = 0,
    invalid_input = 1,
    backend_error = 2,
    internal_error = 3
};

// The resident schedule exposes every semantic phase explicitly.  The
// records are device-owned; the host only submits the kernels in this fixed
// order and never chooses the next phase from an intermediate result.
enum class FmmmSemanticPhase : std::int32_t {
    upload = 0,
    normalize = 1,
    components = 2,
    hierarchy = 3,
    nmm = 4,
    placement = 5,
    normal_force = 6,
    postprocess = 7,
    resize = 8,
    fine_tuning = 9,
    final_resize = 10,
    packing = 11,
    score = 12,
    admission = 13,
    objective = 14,
    publish = 15,
};

constexpr int kFmmmSemanticPhaseCount = 16;

#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
// Native E2E profiling only.  This structure is compiled into the dedicated
// profiling target, never into the normal Python extension.  Events are
// recorded on the caller's session stream and are read after publication has
// synchronized that stream.
struct FmmmNativePhaseTiming {
    cudaEvent_t start[kFmmmSemanticPhaseCount]{};
    cudaEvent_t end[kFmmmSemanticPhaseCount]{};
    bool started[kFmmmSemanticPhaseCount]{};
    bool recorded[kFmmmSemanticPhaseCount]{};
};
#endif

// G0 production geometry is continuous.  Coordinates are quantized to this
// type at the device packing/publication boundary so the production scorer
// and the independent FP32 recount consume identical values.
enum class FmmmPublicationCoordinateType : std::uint32_t {
    double_precision = 0,
    fp32 = 1,
};

// Bounded host-facing publication ABI. The corresponding device record is
// trivially copyable and is stored in the session publication buffer.
struct DeviceFmmmPublication {
    std::string status;
    std::string reason;
    FmmmPublicationCoordinateType coordinate_type =
        FmmmPublicationCoordinateType::double_precision;
    std::vector<std::pair<double, double>> coordinates;
    std::vector<std::int64_t> edge_crossings;
    std::int64_t K = 0;
    std::int64_t nK = 0;
    std::int64_t Phi = 0;
    std::int64_t C = 0;
    bool admitted = false;
    std::uint64_t graph_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t publication_epoch = 0;
    std::uint64_t allocation_generation = 0;
    std::string telemetry_json;
};

enum class FmmmSessionBufferKind : std::uint16_t {
    unknown = 0,
    graph = 1,
    hierarchy = 2,
    nmm = 3,
    force = 4,
    resize = 5,
    score = 6,
    publication = 7,
    telemetry = 8
};

enum class FmmmSessionOwnerLifetime : std::uint8_t {
    session = 1
};

// Direct session allocations carry owner and generation metadata. A generation
// increments only when capacity grows, which lets the residency tests prove
// that execution phases reuse the session allocation set.
template <typename T>
struct SessionBuffer {
    T* d_ptr = nullptr;
    std::size_t capacity_elements = 0;
    std::size_t valid_elements = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t epoch = 0;
    std::uint32_t owner_id = 0;
    FmmmSessionBufferKind kind = FmmmSessionBufferKind::unknown;
    FmmmSessionOwnerLifetime owner_lifetime = FmmmSessionOwnerLifetime::session;
};

struct DeviceFmmmPhaseRecord {
    std::int32_t phase_id = 0;
    std::int32_t status = static_cast<std::int32_t>(FmmmDeviceStatus::internal_error);
    std::int32_t error_code = 0;
    std::int32_t reserved = 0;
    std::uint64_t entry_epoch = 0;
    std::uint64_t exit_epoch = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t input_count = 0;
    std::uint64_t output_count = 0;
};

struct DeviceFmmmTransferRecord {
    std::uint64_t sequence = 0;
    std::uint32_t phase_id = 0;
    std::uint32_t operation = 0;
    std::uint32_t direction = 0;
    std::uint32_t source_kind = 0;
    std::uint32_t destination_kind = 0;
    std::uint32_t callsite_id = 0;
    std::uint32_t stream_id = 0;
    std::uint64_t bytes = 0;
    std::uint64_t allocation_generation = 0;
};

using FmmmTransferLedger = DeviceFmmmTransferRecord;

// Stable audit vocabulary.  These values are part of the bounded telemetry
// contract; do not infer residency from the aggregate counters alone.
enum class FmmmTransferOperation : std::uint32_t {
    allocation = 1,
    release = 2,
    copy = 3,
    synchronize = 4,        // forbidden semantic-phase host barrier
    final_synchronize = 5   // one legal publication-boundary barrier
};

enum class FmmmTransferDirection : std::uint32_t {
    none = 0,
    host_to_device = 1,
    device_to_host = 2,
    device_to_device = 3,
    host_sync = 4
};

enum class FmmmTransferSourceKind : std::uint32_t {
    unknown = 0,
    host = 1,
    device = 2,
    session = 3
};

// Device-side fixed header. Coordinates and edge crossings are stored in the
// adjacent publication backing allocation; no STL object crosses the CUDA ABI.
struct DeviceFmmmPublicationRecord {
    std::int32_t status = static_cast<std::int32_t>(FmmmDeviceStatus::internal_error);
    std::int32_t reason_code = 0;
    std::int32_t admitted = 0;
    std::int32_t fallback_required = 0;
    std::int32_t node_count = 0;
    std::int32_t edge_count = 0;
    std::uint32_t coordinate_type =
        static_cast<std::uint32_t>(FmmmPublicationCoordinateType::double_precision);
    std::uint32_t reserved_coordinate = 0;
    std::int64_t K = 0;
    std::int64_t nK = 0;
    std::int64_t Phi = 0;
    std::int64_t C = 0;
    std::uint64_t graph_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t publication_epoch = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t phase_count = 0;
    std::uint64_t transfer_count = 0;
};

static_assert(std::is_standard_layout<DeviceFmmmPublicationRecord>::value,
              "DeviceFmmmPublicationRecord must have a stable byte layout");
static_assert(std::is_trivially_copyable<DeviceFmmmPublicationRecord>::value,
              "DeviceFmmmPublicationRecord crosses the CUDA publication ABI");
static_assert(sizeof(DeviceFmmmPublicationRecord) % alignof(std::int64_t) == 0,
              "publication record must preserve int64 payload alignment");

struct DeviceFmmmScoreState {
    std::int64_t K = 0;
    std::int64_t nK = 0;
    std::int64_t Phi = 0;
    std::int64_t C = 0;
    std::int32_t status = static_cast<std::int32_t>(FmmmDeviceStatus::internal_error);
    std::int32_t admitted = 0;
    std::int32_t overflow = 0;
    std::int32_t edge_count = 0;
};

struct DeviceFmmmSession {
    FmmmConfig config;
    std::int64_t max_width = 0;
    std::int64_t max_height = 0;
    std::uint32_t owner_id = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t phase_generation_at_entry = 0;

    // Host parsing staging. These vectors are cleared after upload and never
    // become authoritative device state.
    std::vector<double> host_nodes_x;
    std::vector<double> host_nodes_y;
    std::vector<int> host_edges_src;
    std::vector<int> host_edges_tgt;
    std::vector<int> host_edges_layer;

    // Device epochs.
    std::uint64_t graph_epoch = 0;
    std::uint64_t position_epoch = 0;
    std::uint64_t tree_epoch = 0;
    std::uint64_t box_epoch = 0;
    std::uint64_t publication_epoch = 0;

    // One reusable timing set owned by the session.  The events are recorded
    // on the caller-provided session stream and are only waited on at the
    // publication boundary, after the single bounded final D2H is queued.
    cudaEvent_t upload_timing_start = nullptr;
    cudaEvent_t upload_timing_end = nullptr;
    cudaEvent_t device_timing_start = nullptr;
    cudaEvent_t device_timing_end = nullptr;
    cudaEvent_t final_download_timing_start = nullptr;
    cudaEvent_t final_download_timing_end = nullptr;
    cudaStream_t timing_stream = nullptr;
    bool timing_events_ready = false;
    bool upload_timing_recorded = false;
    bool device_timing_recorded = false;
    // Set by fmmm_session_rearm_for_seed so hierarchy can restart device_ms
    // without clobbering the normal upload→device_timing_start handoff.
    bool device_timing_start_pending = false;
    bool final_download_timing_recorded = false;
    bool final_sync_recorded = false;
    float upload_ms = -1.0F;
    float device_ms = -1.0F;
    float final_download_ms = -1.0F;

    // Graph and canonical representation.
    SessionBuffer<double> raw_nodes_x;
    SessionBuffer<double> raw_nodes_y;
    SessionBuffer<int> raw_edges_src;
    SessionBuffer<int> raw_edges_tgt;
    SessionBuffer<int> raw_edges_layer;
    SessionBuffer<int> norm_edges_src;
    SessionBuffer<int> norm_edges_tgt;
    SessionBuffer<int> norm_edges_layer;
    SessionBuffer<double> norm_edges_len;
    SessionBuffer<int> csr_offsets;
    SessionBuffer<int> csr_edges;
    SessionBuffer<int> node_comp;
    SessionBuffer<int> compact_node_ids;
    SessionBuffer<int> local_node_ids;
    SessionBuffer<int> comp_offsets;
    SessionBuffer<int> comp_members;
    SessionBuffer<int> component_parent;
    SessionBuffer<int> component_cursor;
    SessionBuffer<int> normalized_edge_count_device;
    SessionBuffer<int> component_count_device;
    SessionBuffer<std::uint64_t> graph_hash_device;
    int normalized_edge_count = 0; // compatibility mirror, never authoritative
    int component_count = 0;       // compatibility mirror, never authoritative

    // Hierarchy state and device rebuild metadata.
    SessionBuffer<double> hierarchy_down_left_x;
    SessionBuffer<double> hierarchy_down_left_y;
    SessionBuffer<double> hierarchy_box_length;
    SessionBuffer<double> hierarchy_center_x;
    SessionBuffer<double> hierarchy_center_y;
    SessionBuffer<int> hierarchy_parent;
    SessionBuffer<int> hierarchy_children4;
    SessionBuffer<int> hierarchy_level;
    SessionBuffer<int> hierarchy_compact_ids;
    SessionBuffer<int> hierarchy_particle_count;
    SessionBuffer<int> hierarchy_cell_count_device;
    SessionBuffer<int> hierarchy_root_count_device;
    SessionBuffer<int> hierarchy_particle_to_leaf;
    SessionBuffer<std::uint64_t> hierarchy_tree_epoch_device;
    SessionBuffer<std::uint64_t> hierarchy_position_epoch_device;

    // NMM relation/list and expansion state. These are session-owned even
    // while the numerical NMM implementation is wired to consume them.
    SessionBuffer<int> nmm_particle_to_leaf;
    SessionBuffer<int> nmm_contained_begin;
    SessionBuffer<int> nmm_contained_nodes;
    SessionBuffer<int> nmm_temp_contained_nodes;
    SessionBuffer<int> nmm_temp_contained_scratch;
    SessionBuffer<int> nmm_temp_contained_begin;
    SessionBuffer<int> nmm_I_begin;
    SessionBuffer<int> nmm_I_nodes;
    SessionBuffer<int> nmm_D1_begin;
    SessionBuffer<int> nmm_D1_nodes;
    SessionBuffer<int> nmm_D2_begin;
    SessionBuffer<int> nmm_D2_nodes;
    SessionBuffer<int> nmm_M_begin;
    SessionBuffer<int> nmm_M_nodes;
    SessionBuffer<int> nmm_wsprl_queue;
    SessionBuffer<double> nmm_multipole_re;
    SessionBuffer<double> nmm_multipole_im;
    SessionBuffer<double> nmm_local_re;
    SessionBuffer<double> nmm_local_im;
    SessionBuffer<int> nmm_list_counts;

    // Force/placement schedule state.
    SessionBuffer<double> position_x;
    SessionBuffer<double> position_y;
    SessionBuffer<double> position_x_next;
    SessionBuffer<double> position_y_next;
    // Compatibility force view retained for the existing level-engine ABI;
    // the richer force lanes below remain session-owned as well.
    SessionBuffer<double> force_x;
    SessionBuffer<double> force_y;
    SessionBuffer<double> attraction_x;
    SessionBuffer<double> attraction_y;
    SessionBuffer<double> repulsion_x;
    SessionBuffer<double> repulsion_y;
    SessionBuffer<double> movement_x;
    SessionBuffer<double> movement_y;
    SessionBuffer<int> force_status;
    SessionBuffer<int> phase_status;
    SessionBuffer<int> iteration_counter_device;
    SessionBuffer<int> phase_counter_device;

    // Resize/box state.
    SessionBuffer<double> box_length_device;
    SessionBuffer<double> down_left_x_device;
    SessionBuffer<double> down_left_y_device;
    SessionBuffer<double> ideal_length_device;
    SessionBuffer<double> resize_scale_device;
    SessionBuffer<int> resize_status_device;
    SessionBuffer<std::uint64_t> box_epoch_device;

    // Score/admission/objective state.
    SessionBuffer<std::int64_t> edge_crossings;
    SessionBuffer<std::int64_t> score_K_device;
    SessionBuffer<std::int64_t> score_nK_device;
    SessionBuffer<std::int64_t> score_Phi_device;
    SessionBuffer<std::int64_t> score_C_device;
    SessionBuffer<int> admission_status_device;
    SessionBuffer<int> score_status_device;
    SessionBuffer<DeviceFmmmScoreState> score_state;

    // Device publication and phase/transfer telemetry.  The single bounded
    // publication backing layout is: fixed header, double coordinates, int64
    // edge crossings, then the Section 13 phase-record array.  The phase
    // array is copied from this session-owned buffer by the publish schedule
    // and is read back together with the publication payload.
    SessionBuffer<DeviceFmmmPhaseRecord> phase_records;
    SessionBuffer<FmmmTransferLedger> transfer_ledger;
    SessionBuffer<DeviceFmmmPublicationRecord> publication_header;
    SessionBuffer<std::uint8_t> publication_backing;

    // Host mirror is audit metadata only.  The device ledger is initialized
    // during the single input-upload group and subsequently appended by a
    // device kernel; this vector is never used as algorithmic state.
    std::vector<DeviceFmmmTransferRecord> host_transfer_records;
    std::uint64_t transfer_sequence = 0;
    bool transfer_audit_failed = false;

    FmmmNmmState nmm_state{};
    bool input_uploaded = false;
    // Internal G4-session experiment only.  Public fmmm_seed remains
    // session-per-call; this flag preserves host input so an explicitly
    // re-armed native benchmark session can reuse its allocations.
    bool reuse_enabled = false;
    bool normalized = false;
    bool hierarchy_built = false;
    bool nmm_built = false;
    bool schedule_submitted = false;
    bool score_complete = false;
    bool publication_ready = false;

#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
    FmmmNativePhaseTiming native_phase_timing{};
#endif

    // Created once and released once for the complete session lifetime.
    FmmmWorkspace workspace;
    void* nmm_workspace = nullptr;

    // Host-visible audit mirrors.
    int phase_allocation_groups = 0;
    int phase_free_groups = 0;
    int input_upload_groups = 0;
    int intermediate_h2d_groups = 0;
    int intermediate_d2h_groups = 0;
    int device_to_device_groups = 0;
    int host_iteration_poll_groups = 0;
    int final_download_groups = 0;
};

DeviceFmmmSession* fmmm_session_create(const RawFmmmInput& input, const FmmmConfig& config);
const char* fmmm_session_last_create_error();
// Opt-in lifecycle experiment.  These functions are intentionally not
// exported through the Python ABI.  They only permit a native benchmark to
// re-arm the same graph/config session after publication has completed.
void fmmm_session_enable_reuse(DeviceFmmmSession& session);
bool fmmm_session_reset_for_reuse(DeviceFmmmSession& session);
// Soft re-arm after a completed download: keep uploaded+normalized graph state,
// clear seed-dependent phases, and install a new seed.  Requires reuse_enabled.
bool fmmm_session_rearm_for_seed(DeviceFmmmSession& session, std::uint64_t seed);
FmmmDeviceStatus fmmm_session_upload(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_normalize(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_build_hierarchy(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_build_nmm(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_run_schedule(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
// Paper Table-1 seam: close device_ms after run_schedule (excludes score/admit/publish).
// No-op if device timing was already closed. Publish records the end only when still open.
FmmmDeviceStatus fmmm_session_close_layout_device_timing(DeviceFmmmSession& session,
                                                         cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_score_and_admit(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
FmmmDeviceStatus fmmm_session_publish_device(DeviceFmmmSession& session, cudaStream_t stream = nullptr);
DeviceFmmmPublication fmmm_session_download_publication(DeviceFmmmSession& session);
void fmmm_session_destroy(DeviceFmmmSession& session);

#if defined(LCN_FMMM_NATIVE_PHASE_EVENTS)
bool fmmm_native_phase_timing_create(DeviceFmmmSession& session);
void fmmm_native_phase_timing_destroy(DeviceFmmmSession& session) noexcept;
bool fmmm_native_phase_timing_begin(DeviceFmmmSession& session,
                                    FmmmSemanticPhase phase,
                                    cudaStream_t stream);
bool fmmm_native_phase_timing_end(DeviceFmmmSession& session,
                                  FmmmSemanticPhase phase,
                                  cudaStream_t stream);
#endif

} // namespace lcn::release
