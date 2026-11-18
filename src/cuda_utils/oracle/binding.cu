#include "full_square.h"
#include "continuous_crossings.h"
#include "incremental.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

namespace py = pybind11;
namespace lcn::oracle {
#ifdef LCN_ENABLE_RESIDENT_ENGINES
void register_resident_api(py::module_& module);
#endif
namespace {

GraphSpec parse_graph(const py::iterable& nodes, const py::iterable& edges,
                      std::int64_t max_width, std::int64_t max_height) {
    GraphSpec spec;
    spec.max_width = max_width;
    spec.max_height = max_height;
    for (const py::handle item : nodes) {
        const py::sequence point = py::reinterpret_borrow<py::sequence>(item);
        if (point.size() != 2) throw py::value_error("each node must be an (x, y) pair");
        spec.nodes.push_back({py::cast<std::int64_t>(point[0]), py::cast<std::int64_t>(point[1])});
    }
    for (const py::handle item : edges) {
        const py::sequence edge = py::reinterpret_borrow<py::sequence>(item);
        if (edge.size() != 3) throw py::value_error("each edge must be a (source, target, layer) triple");
        const auto source64 = py::cast<std::int64_t>(edge[0]);
        const auto target64 = py::cast<std::int64_t>(edge[1]);
        const auto layer64 = py::cast<std::int64_t>(edge[2]);
        if (source64 < INT32_MIN || source64 > INT32_MAX || target64 < INT32_MIN || target64 > INT32_MAX ||
            layer64 < INT32_MIN || layer64 > INT32_MAX) throw py::value_error("edge ABI integer is out of range");
        spec.edges.push_back({static_cast<std::int32_t>(source64), static_cast<std::int32_t>(target64), static_cast<std::int32_t>(layer64)});
    }
    return spec;
}

struct ContinuousInput {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<std::int32_t> source;
    std::vector<std::int32_t> target;
    std::vector<std::int32_t> layer;
};

ContinuousInput parse_continuous_input(const py::iterable& nodes, const py::iterable& edges) {
    ContinuousInput input;
    for (const py::handle item : nodes) {
        const py::sequence point = py::reinterpret_borrow<py::sequence>(item);
        if (point.size() != 2) throw py::value_error("each continuous node must be an (x, y) pair");
        input.x.push_back(py::cast<float>(point[0]));
        input.y.push_back(py::cast<float>(point[1]));
    }
    for (const py::handle item : edges) {
        const py::sequence edge = py::reinterpret_borrow<py::sequence>(item);
        if (edge.size() != 3) throw py::value_error("each edge must be a (source, target, layer) triple");
        const auto source = py::cast<std::int64_t>(edge[0]);
        const auto target = py::cast<std::int64_t>(edge[1]);
        const auto layer = py::cast<std::int64_t>(edge[2]);
        if (source < INT32_MIN || source > INT32_MAX || target < INT32_MIN || target > INT32_MAX ||
            layer < INT32_MIN || layer > INT32_MAX) throw py::value_error("edge ABI integer is out of range");
        input.source.push_back(static_cast<std::int32_t>(source));
        input.target.push_back(static_cast<std::int32_t>(target));
        input.layer.push_back(static_cast<std::int32_t>(layer));
    }
    return input;
}

py::dict result_dict(const ScoreResult& result, bool is_initial) {
    py::dict out;
    if (is_initial) out["proposal_id"] = py::none();
    else out["proposal_id"] = py::str(result.proposal_id);
    out["proposal_status"] = status_name(result.status);
    out["valid"] = result.valid;
    out["exact_valid"] = result.exact_valid;
    out["edge_count"] = result.edge_crossings.size();
    out["edge_crossings"] = result.edge_crossings;
    out["K"] = result.K;
    out["nK"] = result.nK;
    out["Phi"] = result.Phi;
    out["C"] = result.C;
    out["fallback_required"] = result.fallback_required;
    out["fallback_reason"] = result.fallback_reason;
    return out;
}

py::dict audit_dict(const IncrementalAudit& audit) {
    py::dict out;
    out["lane"] = audit.lane;
    out["compared"] = audit.compared;
    out["counts_match"] = audit.counts_match;
    out["objective_match"] = audit.objective_match;
    out["state_preserved"] = audit.state_preserved;
    out["fallback"] = audit.fallback;
    out["fallback_reason"] = audit.fallback_reason;
    out["candidate_count"] = audit.candidate_count;
    return out;
}

Proposal parse_proposal(const py::handle& item) {
    const py::dict proposal = py::reinterpret_borrow<py::dict>(item);
    Proposal out;
    out.id = py::cast<std::string>(proposal["proposal_id"]);
    out.node_id = py::cast<std::int32_t>(proposal["node_id"]);
    out.from_x = py::cast<std::int64_t>(proposal["from_x"]);
    out.from_y = py::cast<std::int64_t>(proposal["from_y"]);
    out.to_x = py::cast<std::int64_t>(proposal["to_x"]);
    out.to_y = py::cast<std::int64_t>(proposal["to_y"]);
    // acceptance_draw is intentionally not read: replay is evaluator-only and
    // consumes zero RNG draws.  It remains caller-owned fixture metadata.
    return out;
}

} // namespace
} // namespace lcn::oracle

PYBIND11_MODULE(planar_cuda_refactored, m) {
    using namespace lcn::oracle;
    m.doc() = "LCN exact GPU full-square oracle (no CPU scorer or fallback)";

    py::class_<DeviceGraph, std::shared_ptr<DeviceGraph>>(m, "DeviceGraph")
        .def(py::init([](const py::iterable& nodes, const py::iterable& edges,
                         std::int64_t max_width, std::int64_t max_height) {
                 return std::make_shared<DeviceGraph>(parse_graph(nodes, edges, max_width, max_height));
             }), py::arg("nodes"), py::arg("edges"), py::kw_only(),
                 py::arg("max_width"), py::arg("max_height"))
        .def("upload", [](DeviceGraph& graph) { graph.upload(true); })
        .def("close", &DeviceGraph::close)
        .def_property_readonly("uploaded", &DeviceGraph::uploaded)
        .def_property_readonly("node_count", &DeviceGraph::node_count)
        .def_property_readonly("edge_count", &DeviceGraph::edge_count)
        .def("evaluate", [](DeviceGraph& graph, bool layers_enabled) {
            if (!graph.uploaded()) graph.upload(true);
            return result_dict(evaluate_full_square(graph, layers_enabled), true);
        }, py::kw_only(), py::arg("layers_enabled") = true);

    m.def("evaluate_full_square", [](const py::iterable& nodes, const py::iterable& edges,
                                      std::int64_t max_width, std::int64_t max_height, bool layers_enabled) {
        DeviceGraph graph(parse_graph(nodes, edges, max_width, max_height));
        // Geometry admission happens in the CUDA kernel, which is why invalid
        // direct calls yield structured results rather than host-computed ones.
        graph.upload(false);
        return result_dict(evaluate_full_square(graph, layers_enabled), true);
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(), py::arg("max_width"),
       py::arg("max_height"), py::arg("layers_enabled") = true);

    m.def("evaluate_continuous_crossings_fp32", [](const py::iterable& nodes,
                                                     const py::iterable& edges,
                                                     bool layers_enabled) {
        const auto input = parse_continuous_input(nodes, edges);
        const auto result = evaluate_continuous_crossings_fp32(
            static_cast<int>(input.x.size()), static_cast<int>(input.source.size()),
            input.x.data(), input.y.data(), input.source.data(), input.target.data(),
            input.layer.data(), layers_enabled);
        auto out = result_dict(result, true);
        out["authority"] = "CONTINUOUS_CROSSINGS_V1_FP32";
        out["evaluator_mode"] = "CONTINUOUS_CROSSINGS_V1_FP32";
        out["coordinate_type"] = "float32";
        return out;
    }, py::arg("nodes"), py::arg("edges"), py::kw_only(),
       py::arg("layers_enabled") = true);

    m.def("replay_full_square", [](const py::iterable& nodes, const py::iterable& edges,
                                    const py::iterable& proposals, std::int64_t max_width,
                                    std::int64_t max_height, bool layers_enabled) {
        DeviceGraph graph(parse_graph(nodes, edges, max_width, max_height));
        graph.upload(false);
        FullSquareSession session(graph, layers_enabled);
        const ScoreResult initial = session.initial();
        py::list results;
        for (const py::handle proposal : proposals) results.append(result_dict(session.apply(parse_proposal(proposal)), false));
        py::dict replay;
        replay["rng_draws_consumed"] = 0;
        replay["initial_result"] = result_dict(initial, true);
        replay["results"] = results;
        return replay;
    }, py::arg("nodes"), py::arg("edges"), py::arg("proposals"), py::kw_only(),
       py::arg("max_width"), py::arg("max_height"), py::arg("layers_enabled") = true);

    m.def("replay_incremental", [](const py::iterable& nodes, const py::iterable& edges,
                                     const py::iterable& proposals, std::int64_t max_width,
                                     std::int64_t max_height, bool layers_enabled,
                                     const std::string& lane) {
        std::vector<Proposal> parsed;
        for (const py::handle proposal : proposals) parsed.push_back(parse_proposal(proposal));
        const auto replay = replay_incremental_exact(
            parse_graph(nodes, edges, max_width, max_height), parsed, layers_enabled, lane);
        py::dict out;
        out["evaluator_mode"] = "INCREMENTAL_" + lane;
        out["authority"] = "FULL_SQUARE_GPU";
        out["rng_draws_consumed"] = 0;
        out["initial_result"] = result_dict(replay.initial, true);
        py::list results, audit;
        for (const auto& result : replay.results) results.append(result_dict(result, false));
        for (const auto& item : replay.audit) audit.append(audit_dict(item));
        out["results"] = results;
        out["audit"] = audit;
        out["csr_offsets"] = replay.csr_offsets;
        out["csr_edges"] = replay.csr_edges;
        return out;
    }, py::arg("nodes"), py::arg("edges"), py::arg("proposals"), py::kw_only(),
       py::arg("max_width"), py::arg("max_height"), py::arg("layers_enabled") = true,
       py::arg("lane") = "C1");

    m.attr("cuda_enabled") = true;
    m.attr("evaluator_mode") = "FULL_SQUARE_GPU";
#ifdef LCN_ENABLE_RESIDENT_ENGINES
    register_resident_api(m);
#endif
}

// Dual entry point export: lets the extension load under its clean standalone name
// `_fmmm_cuda` or its legacy ABI name `planar_cuda_refactored` without two separate builds.
#if defined(_WIN32)
#  define FMMM_EXPORT __declspec(dllexport)
#else
#  define FMMM_EXPORT __attribute__((visibility("default")))
#endif

extern "C" FMMM_EXPORT PyObject* PyInit__fmmm_cuda(void) {
    return PyInit_planar_cuda_refactored();
}
