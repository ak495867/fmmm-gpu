#include "graph.h"

#include <limits>
#include <stdexcept>

namespace lcn::oracle {

void validate_graph_structure(const GraphSpec& spec) {
    if (spec.nodes.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        spec.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("graph exceeds device ABI capacity");
    }
    if (spec.max_width < 2 || spec.max_height < 2) {
        throw std::invalid_argument("bounds must be at least two");
    }
    for (const auto& edge : spec.edges) {
        if (edge.source < 0 || edge.target < 0 ||
            edge.source >= static_cast<std::int32_t>(spec.nodes.size()) ||
            edge.target >= static_cast<std::int32_t>(spec.nodes.size())) {
            throw std::invalid_argument("edge endpoint index is outside node range");
        }
        if (edge.source == edge.target) throw std::invalid_argument("self-loop edge is not admitted");
    }
}

DeviceGraph::DeviceGraph(GraphSpec spec) : spec_(std::move(spec)) {
    validate_graph_structure(spec_);
}

void DeviceGraph::upload(bool validate_coordinates) {
    if (closed_) throw std::runtime_error("DeviceGraph is closed");
    if (uploaded_) return;

    // Upload never claims ownership until all host admission checks and all
    // allocations/copies have completed.  This keeps failed upload retry-safe.
    for (std::size_t i = 0; validate_coordinates && i < spec_.nodes.size(); ++i) {
        const auto& node = spec_.nodes[i];
        if (node.x <= 0 || node.x >= spec_.max_width || node.y <= 0 || node.y >= spec_.max_height) {
            throw std::invalid_argument("node coordinate is outside strictly interior bounds");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (node.x == spec_.nodes[j].x && node.y == spec_.nodes[j].y) {
                throw std::invalid_argument("duplicate node coordinates are not admitted");
            }
        }
    }
    std::vector<std::int64_t> hx, hy;
    std::vector<std::int32_t> hs, ht, hl;
    hx.reserve(spec_.nodes.size()); hy.reserve(spec_.nodes.size());
    hs.reserve(spec_.edges.size()); ht.reserve(spec_.edges.size()); hl.reserve(spec_.edges.size());
    for (const auto& node : spec_.nodes) { hx.push_back(node.x); hy.push_back(node.y); }
    for (const auto& edge : spec_.edges) { hs.push_back(edge.source); ht.push_back(edge.target); hl.push_back(edge.layer); }

    DeviceBuffer<std::int64_t> new_x(hx.size()), new_y(hy.size());
    DeviceBuffer<std::int32_t> new_s(hs.size()), new_t(ht.size()), new_l(hl.size());
    new_x.upload(hx.data(), hx.size()); new_y.upload(hy.data(), hy.size());
    new_s.upload(hs.data(), hs.size()); new_t.upload(ht.data(), ht.size()); new_l.upload(hl.data(), hl.size());
    x_ = std::move(new_x); y_ = std::move(new_y);
    source_ = std::move(new_s); target_ = std::move(new_t); layer_ = std::move(new_l);
    uploaded_ = true;
}

void DeviceGraph::close() noexcept {
    x_.reset(); y_.reset(); source_.reset(); target_.reset(); layer_.reset();
    uploaded_ = false;
    closed_ = true;
}

__global__ void commit_device_coordinates_kernel(int node_count, const std::int64_t* coords, std::int64_t* x, std::int64_t* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < node_count) {
        x[i] = coords[2 * i];
        y[i] = coords[2 * i + 1];
    }
}

void DeviceGraph::commit_device_coordinates(const DeviceBuffer<std::int64_t>& coordinates) {
    if (!uploaded_) throw std::runtime_error("DeviceGraph is not uploaded");
    if (coordinates.size() != spec_.nodes.size() * 2) {
        throw std::invalid_argument("commit size mismatch");
    }
    const int nodes = static_cast<int>(spec_.nodes.size());
    if (nodes > 0) {
        int threads = 256;
        int blocks = (nodes + threads - 1) / threads;
        commit_device_coordinates_kernel<<<blocks, threads>>>(nodes, coordinates.get(), x_.get(), y_.get());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error("commit kernel failed");
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) throw std::runtime_error("commit sync failed");
    }
}

DeviceGraphView DeviceGraph::view() const {
    if (!uploaded_) throw std::runtime_error("DeviceGraph is not uploaded");
    return {x_.get(), y_.get(), source_.get(), target_.get(), layer_.get(),
            node_count(), edge_count(), spec_.max_width, spec_.max_height};
}

} // namespace lcn::oracle
