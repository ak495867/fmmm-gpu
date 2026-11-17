# fmmm-gpu

Standalone CUDA FMMM layout seed engine extracted from the GPU_LCN research
repository. It accepts a graph and initial coordinates from Python, runs the
GPU FMMM pipeline, and returns a publication record containing coordinates,
status, timing, and GPU-recounted score data.

This repository is deliberately limited to the FMMM layout stage. PTS, PDS,
SharedSA, campaign scheduling, LCN orchestration, seminar material, and
research artifacts stay in the parent project and are not runtime dependencies.

## Current qualification boundary

The current implementation is the existing `fmmm-gpu` seed path qualified for
the declared CUDA envelope. It is not a claim of OGDF coordinate parity,
global optimality, or support for every GPU architecture. The initial build
target is Windows + CUDA + Python 3.11+ + pybind11 + `sm_89`, matching the
source project; architecture and packaging generalization should be a separate
release step.

## Build

From a Developer PowerShell with CUDA `nvcc` on `PATH`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip pybind11
cmake -S . -B build -G Ninja -DFMMM_PYTHON_EXECUTABLE="$((Resolve-Path .\.venv\Scripts\python.exe).Path)"
cmake --build build --config Release
```

For the cleanest Windows build path, the repository also provides a direct
single-invocation NVCC script. It avoids generator-specific custom-command
behavior:

```powershell
.\build.ps1 -Python ..\.venv\Scripts\python.exe
```

This writes the extension to `build-clean\planar_cuda_refactored.pyd`.

The resulting `planar_cuda_refactored.pyd` is the native extension. The Python
facade can then be used with `PYTHONPATH=python;build\Release`:

```python
from fmmm_gpu import seed_layout

result = seed_layout(
    [(1.0, 1.0), (10.0, 10.0)],
    [(0, 1, 0)],
    width=100,
    height=100,
    seed=7,
)
print(result["status"], result["coordinates"], result["score"])
```

## Requirements

- Windows 10/11 with Visual Studio C++ build tools
- NVIDIA CUDA Toolkit 13.0 or a compatible CUDA 12/13 toolchain with `nvcc`
- NVIDIA GPU with the configured `sm_89` target, or a deliberately changed
  `-Architecture`/`FMMM_CUDA_ARCHITECTURES` value for another device
- Python 3.11 or newer
- `pybind11` from `requirements.txt`

Install the Python build dependency with:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Extraction notes

The CUDA source is copied from the parent project at the extraction point; the
original research repository remains the provenance source. `resident_api.cu`
still contains a few legacy compatibility bindings in the native translation
unit, but the standalone Python facade exposes only the FMMM seed operation.
The next cleanup step is to split that translation unit into an FMMM-only
binding and move unrelated compatibility exports out of this repository.

## License and redistribution

This project is released under the MIT License; see [LICENSE](LICENSE).
Review NVIDIA CUDA and any optional third-party provenance before creating a
binary redistribution or a release package.
