# Extraction manifest

This directory is the standalone repository for the CUDA Fast Multipole Multilevel
Method (FMMM) graph layout engine.

## Included

- `src/cuda_utils/release/fmmm_*.cu`, `.h`, and `.cuh`: CUDA FMMM hierarchy,
  force, NMM packing, workspace, CUDA Graph replay, and session implementation.
- `src/cuda_utils/release/resident_api.cu`: Cleaned FMMM native API owner, pruned of all legacy PTS/PDS/SharedSA dead code.
- `src/cuda_utils/oracle/binding.cu`: Dual-module Python entry point (`_fmmm_cuda` and `planar_cuda_refactored`).
- `src/cuda_utils/oracle/`: graph storage, device memory, AABB-accelerated continuous crossings,
  full-square GPU admission/recount, and incremental support required by the
  FMMM implementation.
- `python/fmmm_gpu/`: High-level Python API with typed dataclasses (`LayoutRecord`, `ObjectiveScore`, `TimingBreakdown`), dict backward compatibility, NetworkX adapter (`layout_networkx`), and NumPy/PyTorch converters.
- `tests/`: Standalone comprehensive pytest suite with synthetic mock CUDA engine fallback, topology corpus, invariant verification, and ecosystem interop tests.
- `CMakeLists.txt` & `build.sh` / `build.ps1`: Cross-platform build systems supporting Linux (`.so`) and Windows (`.pyd`) with multi-arch CUDA targets (`sm_75` through `sm_90`).

## Explicitly excluded

- `src/cuda_utils/search_v1/` and all PTS/PDS/SharedSA engines (completely excised).
- `src/lcn_layout/` pipeline orchestration and LCN application policy.
- seminar files, campaign scripts, experiment output, OGDF binaries/source,
  and recovery/forensics material.

## Release blocker status

1. **RESOLVED**: Purged `resident_api.cu` of legacy PTS/PDS/SharedSA search harnesses, workspace estimates, and dead simulated annealing registrations. The translation unit now strictly serves FMMM session and layout execution.
2. **RESOLVED**: Added a complete standalone test suite under `tests/` covering API validation, geometric invariants, NetworkX integration, and topology stress testing, complete with a CPU mock backend for GPU-less environments.
3. Keep the Apache-2.0 license and third-party provenance notice synchronized with future source or dependency changes.
4. **RESOLVED**: Module exports both `PyInit__fmmm_cuda` and `PyInit_planar_cuda_refactored`, enabling modern package naming (`fmmm_gpu`) while maintaining seamless drop-in ABI compatibility with legacy callers.
