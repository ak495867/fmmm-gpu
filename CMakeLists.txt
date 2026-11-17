cmake_minimum_required(VERSION 3.20)
project(fmmm_gpu LANGUAGES NONE)

set(FMMM_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA SM architecture")
set(FMMM_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python executable used for headers and pybind11")
if(FMMM_PYTHON_EXECUTABLE STREQUAL "")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(FMMM_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
else()
    set(Python3_EXECUTABLE "${FMMM_PYTHON_EXECUTABLE}")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

find_program(FMMM_NVCC_EXECUTABLE NAMES nvcc HINTS "$ENV{CUDA_PATH}/bin" REQUIRED)
execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE FMMM_PYBIND_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE FMMM_PYBIND_RESULT)
if(NOT FMMM_PYBIND_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 must be installed in the selected Python environment")
endif()
execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE FMMM_PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${FMMM_PYTHON_EXECUTABLE}" -c "import sys; print(sys.base_prefix + '/libs/python%d%d.lib' % sys.version_info[:2])"
    OUTPUT_VARIABLE FMMM_PYTHON_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(FMMM_CUDA_BIN "${FMMM_NVCC_EXECUTABLE}" DIRECTORY)
get_filename_component(FMMM_CUDA_ROOT "${FMMM_CUDA_BIN}" DIRECTORY)
set(FMMM_CUDART_LIBRARY "${FMMM_CUDA_ROOT}/lib/x64/cudart.lib")

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
    set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/cuda/$<CONFIG>/${UNIT}.obj")
    list(APPEND FMMM_OBJECTS "${OBJECT}")
    add_custom_command(OUTPUT "${OBJECT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/cuda/$<CONFIG>"
        COMMAND "${FMMM_NVCC_EXECUTABLE}" --use-local-env --std c++17
            -arch=sm_${FMMM_CUDA_ARCHITECTURES} -DLCN_ENABLE_RESIDENT_ENGINES=1
            -DLCN_ENABLE_FMMM_PHASE1=1 -Xcompiler=/MD -fmad=false
            -I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils/oracle
            -I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_utils/release
            ${FMMM_PYBIND_INCLUDES} -I${FMMM_PYTHON_INCLUDE}
            -c "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" -o "${OBJECT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" VERBATIM)
endforeach()

set(FMMM_EXTENSION "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/planar_cuda_refactored.pyd")
add_custom_command(OUTPUT "${FMMM_EXTENSION}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    COMMAND "${FMMM_NVCC_EXECUTABLE}" --shared -Xcompiler=/MD
        -o "${FMMM_EXTENSION}" ${FMMM_OBJECTS} "${FMMM_CUDART_LIBRARY}" "${FMMM_PYTHON_LIBRARY}"
    DEPENDS ${FMMM_OBJECTS} VERBATIM)
add_custom_target(fmmm_gpu ALL DEPENDS "${FMMM_EXTENSION}")

install(FILES "${FMMM_EXTENSION}" DESTINATION .)
