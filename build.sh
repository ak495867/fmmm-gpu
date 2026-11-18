#!/usr/bin/env bash
#
# build.sh - Build the standalone FMMM GPU CUDA extension on Linux / WSL.
#
# Because not everyone enjoys navigating CMake cache quirks when all they
# want is a fast compiled .so sitting next to their Python code.
#
# Usage:
#   ./build.sh [--arch 89] [--python /path/to/python] [--cuda /usr/local/cuda]
#

set -euo pipefail

# Default target architecture: Ada Lovelace (RTX 40xx) is sm_89.
# Ampere users will want 80 or 86; Hopper is 90; Turing is 75.
ARCH="89"
PYTHON_BIN=""
CUDA_DIR=""

# --- Command line parsing ----------------------------------------------------
# We parse basic long and short flags without pulling in getopt baggage,
# keeping this script friendly to bare-bones containers.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch|-a)
            ARCH="$2"
            shift 2
            ;;
        --python|-p)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --cuda|-c)
            CUDA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -a, --arch ARCH       CUDA compute architecture (e.g., 75, 80, 86, 89, 90; default: 89)"
            echo "  -p, --python PATH     Path to python3 interpreter (default: detected from PATH)"
            echo "  -c, --cuda PATH       Path to CUDA toolkit directory (default: detected from nvcc)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BUILD_DIR="${REPO_ROOT}/build-clean"

# --- Discover Python ---------------------------------------------------------
# If not explicitly provided, search standard commands.
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "Error: Python was not found in PATH. Please specify --python /path/to/python3." >&2
        exit 1
    fi
fi

echo "Using Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

# --- Discover CUDA Toolkit & NVCC --------------------------------------------
# nvcc is non-negotiable here; if it's missing, the GPU compiler party ends early.
NVCC_BIN=""
if [[ -n "$CUDA_DIR" ]]; then
    NVCC_BIN="${CUDA_DIR}/bin/nvcc"
else
    if command -v nvcc >/dev/null 2>&1; then
        NVCC_BIN="$(command -v nvcc)"
        CUDA_DIR="$(dirname "$(dirname "$NVCC_BIN")")"
    elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        CUDA_DIR="/usr/local/cuda"
        NVCC_BIN="${CUDA_DIR}/bin/nvcc"
    else
        echo "Error: nvcc compiler was not found. Please specify --cuda /path/to/cuda." >&2
        exit 1
    fi
fi

if [[ ! -x "$NVCC_BIN" ]]; then
    echo "Error: nvcc is not executable at '${NVCC_BIN}'." >&2
    exit 1
fi

echo "Using NVCC:   ${NVCC_BIN}"
echo "Target Arch:  sm_${ARCH}"

# --- Inspect Python & pybind11 Include Headers ------------------------------
# Extract Python C header directory
PYTHON_INCLUDE="$("${PYTHON_BIN}" -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || true)"
if [[ -z "$PYTHON_INCLUDE" || ! -f "${PYTHON_INCLUDE}/Python.h" ]]; then
    echo "Error: Python.h not found under '${PYTHON_INCLUDE}'. Do you have python3-dev installed?" >&2
    exit 1
fi

# Verify pybind11 module presence and grab include flags
if ! "${PYTHON_BIN}" -m pybind11 --includes >/dev/null 2>&1; then
    echo "Error: pybind11 is not installed in ${PYTHON_BIN}. Install it via: ${PYTHON_BIN} -m pip install pybind11" >&2
    exit 1
fi
PYBIND_INCLUDES="$("${PYTHON_BIN}" -m pybind11 --includes)"

# Python extension suffix (e.g. .cpython-311-x86_64-linux-gnu.so)
EXT_SUFFIX="$("${PYTHON_BIN}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')" 2>/dev/null || echo '.so')"

mkdir -p "$BUILD_DIR"

# --- Source Files ------------------------------------------------------------
# The curated standalone FMMM sources. We explicitly list these rather than globbing
# so orphaned or experimental sandbox scripts don't accidentally get dragged into the compilation unit.
SOURCES=(
    "${REPO_ROOT}/src/cuda_utils/oracle/binding.cu"
    "${REPO_ROOT}/src/cuda_utils/oracle/device_memory.cu"
    "${REPO_ROOT}/src/cuda_utils/oracle/graph.cu"
    "${REPO_ROOT}/src/cuda_utils/oracle/full_square.cu"
    "${REPO_ROOT}/src/cuda_utils/oracle/continuous_crossings.cu"
    "${REPO_ROOT}/src/cuda_utils/oracle/incremental.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_bh_adapter.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_level_engine.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_advanced_uncoarsen.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_hierarchy.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_workspace.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_nmm_packing.cu"
    "${REPO_ROOT}/src/cuda_utils/release/fmmm_session.cu"
    "${REPO_ROOT}/src/cuda_utils/release/resident_api.cu"
)

OUTPUT_SO="${BUILD_DIR}/planar_cuda_refactored${EXT_SUFFIX}"
SYMLINK_SO="${BUILD_DIR}/_fmmm_cuda${EXT_SUFFIX}"

echo "Building standalone FMMM GPU extension..."

# Compiler flag rationale:
# -O3: Maximum compiler optimization.
# -std=c++17: Required by modern pybind11 and session data structures.
# --shared: Produce a dynamic shared library for Python import.
# --default-stream per-thread: Avoid global CUDA stream serialization bottlenecks.
# -fmad=false: Strict floating point consistency across compute architectures.
# -Xcompiler -fPIC: Position Independent Code for Linux shared object relocation.
# -DLCN_ENABLE_RESIDENT_ENGINES=1 -DLCN_ENABLE_FMMM_PHASE1=1: Enable core layout engines.
"${NVCC_BIN}" \
    -O3 -std=c++17 --shared --default-stream per-thread \
    "-arch=sm_${ARCH}" \
    -DLCN_ENABLE_RESIDENT_ENGINES=1 \
    -DLCN_ENABLE_FMMM_PHASE1=1 \
    -fmad=false \
    -Xcompiler -fPIC \
    -I"${REPO_ROOT}/src/cuda_utils/oracle" \
    -I"${REPO_ROOT}/src/cuda_utils/release" \
    -I"${PYTHON_INCLUDE}" \
    ${PYBIND_INCLUDES} \
    -o "${OUTPUT_SO}" \
    "${SOURCES[@]}" \
    -lcudart

# Create alias link so both module names ('_fmmm_cuda' and 'planar_cuda_refactored') resolve
ln -sf "$(basename "${OUTPUT_SO}")" "${SYMLINK_SO}"

echo "Successfully built:"
echo "  -> ${OUTPUT_SO}"
echo "  -> ${SYMLINK_SO}"
echo ""
echo "To test in Python:"
echo "  PYTHONPATH=\"${BUILD_DIR}:python\" ${PYTHON_BIN} -c \"import fmmm_gpu; print(fmmm_gpu.__file__)\""
