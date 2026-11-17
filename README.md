# fmmm-gpu

Standalone CUDA implementation of an FMMM-inspired graph-layout seed engine
with a small Python API. The project accepts a graph and initial coordinates,
runs the GPU layout path, performs GPU-side geometry admission and scoring,
and returns a provenance-carrying publication record.

This repository contains the layout stage only. PTS, PDS, SharedSA, campaign
scheduling, LCN application orchestration, seminar material, and experiment
artifacts remain outside this repository.

## Acknowledgements and provenance

This project is grateful to the authors of the Fast Multipole Multilevel
Method (FMMM/FM3), especially:

> Stefan Hachul and Michael Jünger, “Drawing Large Graphs with a
> Potential-Field-Based Multilevel Algorithm,” Graph Drawing 2004.

- [Paper and publication record](https://doi.org/10.1007/978-3-540-31843-9_29)
- [Open-access paper copy](https://kups.ub.uni-koeln.de/54877/1/zaik2004-472.pdf)
- [OGDF repository](https://github.com/ogdf/ogdf)
- [OGDF FMMMLayout documentation](https://ogdf.github.io/doc/ogdf/classogdf_1_1_f_m_m_m_layout.html)

The paper provides the mathematical and algorithmic foundation for the
multilevel force-directed method and its potential-field/multipole ideas. The
OGDF implementation is used as a reference for algorithm structure, options,
and comparison behavior. The CUDA implementation in this repository is our
own GPU-oriented reimplementation and adaptation; it is not a claim of
binary-identical OGDF behavior and does not redistribute the OGDF source tree.

The comparison work uses the pinned OGDF reference profile
`ogdf-foxglove-202510` at commit
`5b6795655399b9d8e2921afec9d97bab9107d5ee`. OGDF is an optional comparison
reference, not a runtime dependency of the Python CUDA extension.

## Data and execution flow

```text
Python graph input
        |
        v
Input normalization and boundary checks
        |
        v
Host -> GPU graph upload
        |
        v
GPU FMMM-inspired layout path
  hierarchy / coarsening
  placement / force evaluation
  Barnes-Hut or NMM-style device work
  uncoarsening / coordinate publication
        |
        v
GPU geometry admission and FULL_SQUARE_GPU recount
        |
        v
Python publication record
  status / coordinates / score / timing / provenance
```

The GPU admission and recount are part of the trust boundary. A layout is not
reported as accepted merely because the force loop produced coordinates. The
returned score contains the lexicographic objective `(K, nK, Phi, C)` together
with edge-crossing counts and validity fields.

## Python API

```python
from fmmm_gpu import seed_layout

result = seed_layout(
    [(1.0, 1.0), (10.0, 10.0)],
    [(0, 1, 0)],
    width=100,
    height=100,
    seed=7,
    iterations=3,
)

print(result["status"])
print(result["coordinates"])
print(result["score"])
```

The public facade intentionally exposes only `seed_layout` and hides the
native extension loading details. The native module is currently named
`planar_cuda_refactored` for ABI compatibility with the source project.

## Requirements

- Windows 10/11
- Visual Studio 2022 C++ build tools
- NVIDIA CUDA Toolkit 13.0, or a compatible CUDA 12/13 installation with
  `nvcc`
- NVIDIA GPU; the initial qualified target is Ada `sm_89`
- Python 3.11 or newer
- `pybind11` from [requirements.txt](requirements.txt)

The build is currently qualified on Windows with Python 3.11, CUDA 13.0,
MSVC 2022, and an RTX 4060 Laptop GPU. Other architectures may work after
changing the architecture setting, but they are not covered by the current
qualification statement.

## Build from a clean checkout

From a Developer PowerShell with `nvcc` available:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\build.ps1 -Python .\.venv\Scripts\python.exe
```

The direct build script performs one explicit NVCC invocation and writes the
native extension to `build-clean\planar_cuda_refactored.pyd`. It also places
NVCC temporary files under the ignored `build-clean\nvcc-tmp` directory so
that compiler-generated files do not depend on a system temporary path.

The CMake configuration is retained for users who prefer a generator-based
workflow:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DFMMM_PYTHON_EXECUTABLE="$((Resolve-Path .\.venv\Scripts\python.exe).Path)"
cmake --build build --config Release
```

## Validation and test evidence

### Standalone installation smoke test

The clean-copy installation procedure has been verified in a separate folder
without reusing the native build output:

```text
fmmm-gpu-install-check/build-clean/planar_cuda_refactored.pyd
status: OK
valid: True
exact_valid: True
K: 0
edge_crossings: [0]
```

The test graph contains two nodes and one edge. This is an installation and
API smoke test, not a quality or performance benchmark.

### Result contract

An accepted result is expected to contain:

- `status == "OK"`
- `coordinates`
- `score.valid` and `score.exact_valid`
- `score.K`, `score.nK`, `score.Phi`, and `score.C`
- `score.edge_crossings`
- admission/provenance fields supplied by the native publication path

Invalid inputs and rejected layouts are represented as structured records;
callers should not silently repair a rejected layout and then label it as an
FMMM result.

### Timing semantics

Timing fields must be interpreted with their seam. `layout_ms` is intended to
measure the layout schedule that produces coordinates. GPU admission,
geometry validation, exhaustive recount, and bounded result transfer are
reported as separate work when the surrounding experiment protocol records
them. Including scoring inside a layout timer answers a different question.

This repository currently makes no universal speedup, throughput, or global
layout-quality claim. Any future benchmark must freeze the graph corpus,
seed, FMMM profile, CUDA architecture, compiler, warm-up policy, and timing
boundaries, and must publish the corresponding raw manifests.

## Current qualification boundary

The current implementation is a GPU FMMM-inspired seed path with GPU-side
admission and recount for its declared test envelope. It is not yet a
formula-by-formula, coordinate-identical OGDF port. Known differences include
force arithmetic, precision, reduction/order choices, and selected hierarchy
behavior. Comparisons should therefore report validity and GPU-scored
objective values under a frozen protocol rather than claiming exact OGDF
coordinate parity.

The extracted native translation unit still contains a small number of legacy
compatibility bindings for ABI continuity. The Python facade does not expose
those unrelated operations; splitting the native binding into an FMMM-only
translation unit is a planned cleanup.

## Repository layout

```text
src/cuda_utils/oracle/       GPU graph storage and FULL_SQUARE_GPU authority
src/cuda_utils/release/      FMMM hierarchy, force, NMM, session, and binding
python/fmmm_gpu/             public Python facade
build.ps1                   direct Windows/NVCC build
CMakeLists.txt              generator-based build configuration
requirements.txt            build dependency
EXTRACTION_MANIFEST.md      source boundary and provenance notes
```

## License

Copyright 2026 Ming-Han Tsai.

This project is licensed under the [Apache License 2.0](LICENSE). OGDF and
the FMMM paper are acknowledged references; their own licenses and terms
remain separate and must be reviewed before redistributing any third-party
material.
