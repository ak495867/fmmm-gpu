#pragma once

#include "device_memory.h"

#include <cstdint>
#include <vector>

namespace lcn::oracle {

struct HostNode { std::int64_t x; std::int64_t y; };
struct HostEdge { std::int32_t source; std::int32_t target; std::int32_t layer; };

struct GraphSpec {
    std::vector<HostNode> nodes;
    std::vector<HostEdge> edges;
    std::int64_t max_width = 0;
    std::int64_t max_height = 0;
};

// The exact ABI passed to CUDA kernels.  It intentionally contains only POD
// pointers and dimensions; ownership remains in DeviceGraph.
struct DeviceGraphView {
    std::int64_t* x;
    std::int64_t* y;
    const std::int32_t* source;
    const std::int32_t* target;
    const std::int32_t* layer;
    std::int32_t node_count;
    std::int32_t edge_count;
    std::int64_t max_width;
    std::int64_t max_height;
};

class DeviceGraph {
public:
    explicit DeviceGraph(GraphSpec spec);
    ~DeviceGraph() = default;
    DeviceGraph(const DeviceGraph&) = delete;
    DeviceGraph& operator=(const DeviceGraph&) = delete;
    DeviceGraph(DeviceGraph&&) noexcept = default;
    DeviceGraph& operator=(DeviceGraph&&) noexcept = default;

    // Public ABI upload validates coordinate admission.  The oracle evaluator
    // uses validate_coordinates=false so malformed proposals reach the device
    // predicate and produce a structured device result rather than a host
    // scorer decision.
    void upload(bool validate_coordinates = true);
    void close() noexcept;
    void commit_device_coordinates(const DeviceBuffer<std::int64_t>& coordinates);
    bool uploaded() const { return uploaded_; }
    bool closed() const { return closed_; }
    std::int32_t node_count() const { return static_cast<std::int32_t>(spec_.nodes.size()); }
    std::int32_t edge_count() const { return static_cast<std::int32_t>(spec_.edges.size()); }
    const GraphSpec& spec() const { return spec_; }
    DeviceGraphView view() const;

private:
    GraphSpec spec_;
    DeviceBuffer<std::int64_t> x_;
    DeviceBuffer<std::int64_t> y_;
    DeviceBuffer<std::int32_t> source_;
    DeviceBuffer<std::int32_t> target_;
    DeviceBuffer<std::int32_t> layer_;
    bool uploaded_ = false;
    bool closed_ = false;
};

void validate_graph_structure(const GraphSpec& spec);

} // namespace lcn::oracle
