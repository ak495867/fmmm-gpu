cmake_minimum_required(VERSION 3.20)
project(fmmm_gpu LANGUAGES NONE)

# Support modern NVIDIA architectures: Turing (75), Ampere (80, 86), Ada (89), Hopper (90).
# Can be overridden via -DFMMM_CUDA_ARCHITECTURES="89" for fast single-target compilation.
set(FMMM_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA SM architectures (semicolon-separated)")

set(FMMM_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python executable used for headers and pybind11")
if(FMMM_PYTHON_EXECUTABLE STREQUAL "")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(FMMM_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
else()
    set(Python3_EXECUTABLE "${FMMM_PYTHON_EXECUTABLE}")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

find_program(FMMM_NVCC_EXECUTABLE NAMES nvcc HINTS "$ENV{CUDA_PATH}/bin" REQUIRED)

# Query pybind11 include directories from the target Python environment
execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE FMMM_PYBIND_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE FMMM_PYBIND_RESULT)
if(NOT FMMM_PYBIND_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 must be installed in the selected Python environment")
endif()

execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE FMMM_PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)

get_filename_component(FMMM_CUDA_BIN "${FMMM_NVCC_EXECUTABLE}" DIRECTORY)
get_filename_component(FMMM_CUDA_ROOT "${FMMM_CUDA_BIN}" DIRECTORY)

# Platform-specific compiler flags and library extensions
if(WIN32)
    set(FMMM_EXT_SUFFIX ".pyd")
    set(FMMM_OBJ_EXT "obj")
    set(FMMM_HOST_FLAGS "-Xcompiler=/MD -Xcompiler=/wd4819 -Xcompiler=/wd4005")
    set(FMMM_CUDART_LIBRARY "${FMMM_CUDA_ROOT}/lib/x64/cudart.lib")
    execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -c "import sys; print(sys.base_prefix + '/libs/python%d%d.lib' % sys.version_info[:2])"
        OUTPUT_VARIABLE FMMM_PYTHON_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(FMMM_EXTRA_LINK_LIBS "${FMMM_CUDART_LIBRARY}" "${FMMM_PYTHON_LIBRARY}")
else()
    set(FMMM_EXT_SUFFIX ".so")
    set(FMMM_OBJ_EXT "o")
    set(FMMM_HOST_FLAGS "-Xcompiler=-fPIC -Xcompiler=-Wall")
    find_library(FMMM_CUDART_LIBRARY NAMES cudart cudart_static PATHS "${FMMM_CUDA_ROOT}/lib64" "${FMMM_CUDA_ROOT}/lib" REQUIRED)
    # On Linux/macOS Python modules resolve runtime symbols from the host interpreter executable
    set(FMMM_EXTRA_LINK_LIBS "${FMMM_CUDART_LIBRARY}")
endif()

# Build architecture flags: format as multiple -gencode flags or single arch
set(FMMM_NVCC_ARCH_FLAGS "")
foreach(ARCH IN LISTS FMMM_CUDA_ARCHITECTURES)
    string(STRIP "${ARCH}" ARCH)
    if(NOT ARCH STREQUAL "")
        list(APPEND FMMM_NVCC_ARCH_FLAGS "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    endif()
endforeach()

set(FMMM_SOURCES
    src/cuda_utils/oracle/binding.cu
    src/cuda_utils/oracle/device_memory.cu
    src/cuda_utils/oracle/graph.cu
    src/cuda_utils/oracle/full_square.cu
    src/cuda_utils/oracle/continuous_crossings.cu
    src/cuda_utils/oracle/incremental.cu
    src/cuda_utils/release/fmmm_bh_adapter.cu
    src/cuda_utils/release/fmmm_level_engine.cu
    src/cuda_utils/release/fmmm_advanced_uncoarsen.cu
    src/cuda_utils/release/fmmm_hierarchy.cu
    src/cuda_utils/release/fmmm_workspace.cu
    src/cuda_utils/release/fmmm_nmm_packing.cu
    src/cuda_utils/release/fmmm_session.cu
    src/cuda_utils/release/resident_api.cu)

set(FMMM_OBJECTS)
foreach(SOURCE IN LISTS FMMM_SOURCES)
    get_filename_component(UNIT "${SOURCE}" NAME_WE)
    set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/cuda/$<CONFIG>/${UNIT}.${FMMM_OBJ_EXT}")
    list(APPEND FMMM_OBJECTS "${OBJECT}")
    add_custom_command(OUTPUT "${OBJECT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/cuda/$<CONFIG>"
        COMMAND "${FMMM_NVCC_EXECUTABLE}" --std=c++17
            ${FMMM_NVCC_ARCH_FLAGS} -DLCN_ENABLE_RESIDENT_ENGINES=1
            -DLCN_ENABLE_FMMM_PHASE1=1 ${FMMM_HOST_FLAGS} -fmad=false
            -I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils/oracle
            -I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils/release
            ${FMMM_PYBIND_INCLUDES} -I${FMMM_PYTHON_INCLUDE}
            -c "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" -o "${OBJECT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" VERBATIM)
endforeach()

# Primary output: modern _fmmm_cuda extension
set(FMMM_EXTENSION "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/_fmmm_cuda${FMMM_EXT_SUFFIX}")
# Legacy compatibility alias: planar_cuda_refactored
set(FMMM_LEGACY_EXTENSION "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/planar_cuda_refactored${FMMM_EXT_SUFFIX}")

add_custom_command(OUTPUT "${FMMM_EXTENSION}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    COMMAND "${FMMM_NVCC_EXECUTABLE}" --shared ${FMMM_HOST_FLAGS}
        -o "${FMMM_EXTENSION}" ${FMMM_OBJECTS} ${FMMM_EXTRA_LINK_LIBS}
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${FMMM_EXTENSION}" "${FMMM_LEGACY_EXTENSION}"
    DEPENDS ${FMMM_OBJECTS} VERBATIM)

add_custom_target(fmmm_gpu ALL DEPENDS "${FMMM_EXTENSION}")

install(FILES "${FMMM_EXTENSION}" "${FMMM_LEGACY_EXTENSION}" DESTINATION .)
