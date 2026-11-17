# Extraction manifest

This directory is a staging tree for a future standalone GitHub repository.

## Included

- `src/cuda_utils/release/fmmm_*.cu`, `.h`, and `.cuh`: CUDA FMMM hierarchy,
  force, NMM packing, workspace, and session implementation.
- `src/cuda_utils/release/resident_api.cu`: current pybind11/native API owner.
- `src/cuda_utils/oracle/binding.cu`: Python module entry point and resident
  API registration.
- `src/cuda_utils/oracle/`: graph storage, device memory, continuous crossings,
  full-square GPU admission/recount, and incremental support required by the
  FMMM implementation.
- `python/fmmm_gpu/`: the intentionally small public Python facade.

## Explicitly excluded

- `src/cuda_utils/search_v1/` and all PTS/PDS/SharedSA engines.
- `src/lcn_layout/` pipeline orchestration and LCN application policy.
- seminar files, campaign scripts, experiment output, OGDF binaries/source,
  and recovery/forensics material.

## Provenance

The files were copied from the parent `GPU_LCN` working tree on 2026-11-17.
The parent repository had uncommitted changes at extraction time, so this
staging tree must not be published until a source commit is selected and the
copied files are compared against that commit.

## Release blockers

1. Split `resident_api.cu` into an FMMM-only binding translation unit; the
   current unit still contains legacy compatibility exports even though the
   facade does not expose them.
2. Add a real test corpus and run GPU build/replay on the supported matrix.
3. Keep the Apache-2.0 license and third-party provenance notice synchronized
   with future source or dependency changes.
4. Decide whether the public package name should be `fmmm-gpu` or a project
   namespace such as `gpu-fmmm`.
