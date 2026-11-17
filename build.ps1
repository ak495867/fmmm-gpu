param(
    [string]$Python = "",
    [string]$CudaPath = "",
    [string]$Architecture = "89"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$BuildDir = Join-Path $Repo "build-clean"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $candidate = Join-Path $Repo "..\.venv\Scripts\python.exe"
        if (Test-Path $candidate) { $Python = (Resolve-Path $candidate).Path }
    }
}
if (-not (Test-Path $Python)) { throw "Python was not found. Pass -Python path\to\python.exe" }

if ([string]::IsNullOrWhiteSpace($CudaPath)) {
    $nvcc = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
    if ([string]::IsNullOrWhiteSpace($nvcc)) { throw "nvcc was not found. Pass -CudaPath C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0" }
    $CudaPath = Split-Path (Split-Path $nvcc -Parent) -Parent
}
$Nvcc = Join-Path $CudaPath "bin\nvcc.exe"
if (-not (Test-Path $Nvcc)) { throw "nvcc was not found at $Nvcc" }

$PythonInclude = (& $Python -c "import sysconfig; print(sysconfig.get_path('include'))").Trim()
$PythonLibrary = (& $Python -c "import sys; print(sys.base_prefix + '/libs/python%d%d.lib' % sys.version_info[:2])").Trim()
$PybindIncludes = (& $Python -m pybind11 --includes).Trim().Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
if (-not (Test-Path (Join-Path $PythonInclude "Python.h"))) { throw "Python.h was not found under $PythonInclude" }
if (-not (Test-Path $PythonLibrary)) { throw "Python library was not found at $PythonLibrary" }

New-Item -ItemType Directory -Force $BuildDir | Out-Null
$NvccTemp = Join-Path $BuildDir "nvcc-tmp"
New-Item -ItemType Directory -Force $NvccTemp | Out-Null
$env:TEMP = $NvccTemp
$env:TMP = $NvccTemp
$Sources = @(
    "src\cuda_utils\oracle\binding.cu",
    "src\cuda_utils\oracle\device_memory.cu",
    "src\cuda_utils\oracle\graph.cu",
    "src\cuda_utils\oracle\full_square.cu",
    "src\cuda_utils\oracle\continuous_crossings.cu",
    "src\cuda_utils\oracle\incremental.cu",
    "src\cuda_utils\release\fmmm_bh_adapter.cu",
    "src\cuda_utils\release\fmmm_level_engine.cu",
    "src\cuda_utils\release\fmmm_advanced_uncoarsen.cu",
    "src\cuda_utils\release\fmmm_hierarchy.cu",
    "src\cuda_utils\release\fmmm_workspace.cu",
    "src\cuda_utils\release\fmmm_nmm_packing.cu",
    "src\cuda_utils\release\fmmm_session.cu",
    "src\cuda_utils\release\resident_api.cu"
) | ForEach-Object { Join-Path $Repo $_ }

$args = @(
    "-O3", "-std=c++17", "--shared", "--default-stream", "per-thread",
    "-arch=sm_$Architecture", "-DLCN_ENABLE_RESIDENT_ENGINES=1",
    "-DLCN_ENABLE_FMMM_PHASE1=1", "-fmad=false",
    "-Xcompiler=/MD", "-Xcompiler=/wd4819", "-Xcompiler=/wd4005",
    "-I$(Join-Path $Repo 'src\cuda_utils\oracle')",
    "-I$(Join-Path $Repo 'src\cuda_utils\release')",
    "-I$PythonInclude"
) + $PybindIncludes + @(
    "-o", (Join-Path $BuildDir "planar_cuda_refactored.pyd")
) + $Sources + @(
    (Join-Path $CudaPath "lib\x64\cudart.lib"), $PythonLibrary
)

Write-Host "Building standalone FMMM GPU extension..."
Write-Host "$Nvcc $($args -join ' ')"
& $Nvcc @args
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Host "Built $(Join-Path $BuildDir 'planar_cuda_refactored.pyd')"
