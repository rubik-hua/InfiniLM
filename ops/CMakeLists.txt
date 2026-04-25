cmake_minimum_required(VERSION 3.18)
project(infinicore_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INFINIOPS_ENABLE_CPU     "Enable CPU backend"     ON)
option(INFINIOPS_ENABLE_NVIDIA  "Enable NVIDIA backend"  ON)
option(INFINIOPS_BUILD_PYBIND   "Build _infinicore py module" ON)
option(INFINIOPS_ENABLE_CUDNN   "Link cuDNN"             ON)
option(INFINIOPS_ENABLE_GRAPH   "Enable device graph (e.g. CUDA Graph)" ON)

# Flash-attention integration. Point at a checkout of vllm-project/flash-attention
# (or Dao-AILab/flash-attention) to build libflash-attn-nvidia.so and enable the
# FlashAttention backend in libinfinicore.
set(INFINIOPS_FLASH_ATTN_DIR "" CACHE PATH "Path to flash-attention source repo")
set(INFINIOPS_FLASH_ATTN_ARCHS "" CACHE STRING "Override SM arches for flash-attn (default 80;86;89;90)")

set(INFINILM_TP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party" CACHE PATH "third_party root")
set(SPDLOG_INCLUDE "${INFINILM_TP_ROOT}/spdlog/include")
set(JSON_INCLUDE   "${INFINILM_TP_ROOT}/json/single_include")

# Runtime sibling subproject (libinfinirt)
set(INFINILM_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../runtime" CACHE PATH "Path to runtime subproject")
if(NOT TARGET infinirt)
    add_subdirectory("${INFINILM_RUNTIME_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_infinirt")
endif()

# ---- infiniop C API source collection ----
file(GLOB INFINIOP_ROOT_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/*.cc"
)
file(GLOB INFINIOP_OP_DISPATCH_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/operator.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/*/operator.cc"
)
# devices/handle.cc dispatcher always needed
list(APPEND INFINIOP_ROOT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/devices/handle.cc")

set(INFINIOP_CPU_SRC "")
set(INFINIOP_NVIDIA_CU_SRC "")
set(INFINIOP_DEFS "")
set(INFINIOP_LINK_LIBS "")

if(INFINIOPS_ENABLE_CPU)
    file(GLOB _cpu_dev_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/devices/cpu/*.cc")
    file(GLOB _cpu_ops_src CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/cpu/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/*/cpu/*.cc")
    file(GLOB _cpu_reduce CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/reduce/cpu/*.cc")
    list(APPEND INFINIOP_CPU_SRC ${_cpu_dev_src} ${_cpu_ops_src} ${_cpu_reduce})
    list(APPEND INFINIOP_DEFS ENABLE_CPU_API)
endif()

if(INFINIOPS_ENABLE_NVIDIA)
    enable_language(CUDA)
    file(GLOB _nv_dev_src  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/devices/nvidia/*.cu")
    file(GLOB _nv_ops_src1 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/nvidia/*.cu")
    file(GLOB _nv_ops_src2 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infiniop/ops/*/*/nvidia/*.cu")
    list(APPEND INFINIOP_NVIDIA_CU_SRC ${_nv_dev_src} ${_nv_ops_src1} ${_nv_ops_src2})
    list(APPEND INFINIOP_DEFS ENABLE_NVIDIA_API)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND INFINIOP_LINK_LIBS CUDA::cudart CUDA::cublas)
    if(INFINIOPS_ENABLE_CUDNN)
        # Debian/Ubuntu ship cuDNN headers under /usr/include/x86_64-linux-gnu;
        # NVIDIA tarball under $CUDNN_ROOT/include.
        find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS
                $ENV{CUDNN_ROOT} $ENV{CUDNN_HOME} $ENV{CUDNN_PATH}
                /usr/local/cuda
            PATH_SUFFIXES include x86_64-linux-gnu
        )
        if(NOT CUDNN_INCLUDE_DIR)
            message(FATAL_ERROR "cudnn.h not found. Install cuDNN or set CUDNN_ROOT.")
        endif()
        list(APPEND INFINIOP_LINK_LIBS cudnn)
        list(APPEND INFINIOP_DEFS ENABLE_CUDNN_API)
    endif()
endif()

# ---- infinicore C++ API source collection ----
file(GLOB INFINICORE_ROOT_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/*.cc"
)
file(GLOB_RECURSE INFINICORE_CTX_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/context/*.cc"
)
file(GLOB INFINICORE_TENSOR_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/tensor/*.cc"
)
file(GLOB INFINICORE_GRAPH_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/graph/*.cc"
)
file(GLOB INFINICORE_NN_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/nn/*.cc"
)
file(GLOB INFINICORE_OPS_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/ops/*/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/ops/*/*/*.cc"
)

set(INFINICORE_ADAPTOR_SRC "")
if(INFINIOPS_FLASH_ATTN_DIR)
    list(APPEND INFINICORE_ADAPTOR_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/adaptor/aten_adaptor.cc")
endif()

# ---- Utils ----
file(GLOB UTILS_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cc"
)

add_library(infinicore SHARED
    ${UTILS_SRC}
    ${INFINICORE_ADAPTOR_SRC}
    ${INFINIOP_ROOT_SRC}
    ${INFINIOP_OP_DISPATCH_SRC}
    ${INFINIOP_CPU_SRC}
    ${INFINIOP_NVIDIA_CU_SRC}
    ${INFINICORE_ROOT_SRC}
    ${INFINICORE_CTX_SRC}
    ${INFINICORE_TENSOR_SRC}
    ${INFINICORE_GRAPH_SRC}
    ${INFINICORE_NN_SRC}
    ${INFINICORE_OPS_SRC}
)

target_include_directories(infinicore
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${SPDLOG_INCLUDE}"
            "${JSON_INCLUDE}"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
if(INFINIOPS_ENABLE_CUDNN AND CUDNN_INCLUDE_DIR)
    target_include_directories(infinicore PRIVATE "${CUDNN_INCLUDE_DIR}")
endif()

target_compile_definitions(infinicore PRIVATE ${INFINIOP_DEFS})
if(INFINIOPS_ENABLE_GRAPH)
    target_compile_definitions(infinicore PRIVATE USE_INFINIRT_GRAPH)
endif()
target_compile_options(infinicore PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>
)

if(INFINIOPS_ENABLE_NVIDIA)
    set_target_properties(infinicore PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set_target_properties(infinicore PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_options(infinicore PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-function>
    )
endif()

# libinfinicore calls infiniccl symbols (distributed allreduce).
set(INFINILM_CCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ccl" CACHE PATH "Path to ccl subproject")
if(NOT TARGET infiniccl)
    add_subdirectory("${INFINILM_CCL_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_infiniccl")
endif()

target_link_libraries(infinicore PUBLIC infinirt infiniccl PRIVATE ${INFINIOP_LINK_LIBS})

# ---- Flash-attention (optional) ----
# When INFINIOPS_FLASH_ATTN_DIR points at a flash-attention checkout, build
# libflash-attn-nvidia.so from its FA2 sources and enable ENABLE_FLASH_ATTN +
# ENABLE_ATEN inside libinfinicore so the flash backends compile in.
if(INFINIOPS_FLASH_ATTN_DIR AND INFINIOPS_ENABLE_NVIDIA)
    if(NOT EXISTS "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn/flash_api.cpp")
        message(FATAL_ERROR "INFINIOPS_FLASH_ATTN_DIR='${INFINIOPS_FLASH_ATTN_DIR}' does not contain csrc/flash_attn/flash_api.cpp")
    endif()
    if(NOT EXISTS "${INFINIOPS_FLASH_ATTN_DIR}/csrc/cutlass/include/cutlass/cutlass.h")
        message(FATAL_ERROR "cutlass submodule missing under ${INFINIOPS_FLASH_ATTN_DIR}/csrc/cutlass; run 'git submodule update --init --recursive'")
    endif()

    # Locate PyTorch C++ (ATen) installation via Python.
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
        OUTPUT_VARIABLE TORCH_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_rc
    )
    if(NOT _torch_rc EQUAL 0 OR TORCH_DIR STREQUAL "")
        message(FATAL_ERROR "PyTorch not importable; flash-attn integration needs ATen headers (pip install torch).")
    endif()
    message(STATUS "PyTorch dir: ${TORCH_DIR}")

    # FA2 forward-only kernel set (skip backward + sparse).
    file(GLOB FA2_KERNEL_SRCS CONFIGURE_DEPENDS
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn/src/flash_fwd_*.cu"
    )
    list(FILTER FA2_KERNEL_SRCS EXCLUDE REGEX "flash_fwd_sparse_")

    add_library(flash-attn-nvidia SHARED
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn/flash_api.cpp"
        ${FA2_KERNEL_SRCS}
    )
    set_target_properties(flash-attn-nvidia PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
    # FA2 only has sm_80–sm_90 kernels. Avoid CMake's "native" probe (mis-detects
    # on hosts without a visible GPU at configure time).
    set(_fa_archs "${INFINIOPS_FLASH_ATTN_ARCHS}")
    if(NOT _fa_archs)
        if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
            set(_fa_archs "${CMAKE_CUDA_ARCHITECTURES}")
        else()
            # Single-arch default keeps the .so under ~200MB and compile under
            # 5 minutes. Multi-arch users override via INFINIOPS_FLASH_ATTN_ARCHS.
            set(_fa_archs "80")
        endif()
    endif()
    set_target_properties(flash-attn-nvidia PROPERTIES CUDA_ARCHITECTURES "${_fa_archs}")
    target_include_directories(flash-attn-nvidia PRIVATE
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn"
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn/src"
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/common"
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/cutlass/include"
        "${TORCH_DIR}/include"
        "${TORCH_DIR}/include/torch/csrc/api/include"
        "${Python3_INCLUDE_DIRS}"
    )
    target_compile_definitions(flash-attn-nvidia PRIVATE
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        FLASHATTENTION_DISABLE_PYBIND
        CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1
        TORCH_API_INCLUDE_EXTENSION_H
    )
    target_compile_options(flash-attn-nvidia PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-function>
    )
    target_link_directories(flash-attn-nvidia PUBLIC "${TORCH_DIR}/lib")
    target_link_libraries(flash-attn-nvidia PUBLIC torch torch_cuda c10 c10_cuda CUDA::cudart)

    # Tell libinfinicore to compile its flash-attn glue layers.
    target_compile_definitions(infinicore PRIVATE ENABLE_FLASH_ATTN ENABLE_ATEN)
    target_include_directories(infinicore PRIVATE
        "${TORCH_DIR}/include"
        "${TORCH_DIR}/include/torch/csrc/api/include"
        "${INFINIOPS_FLASH_ATTN_DIR}/csrc/flash_attn"
    )
    target_link_libraries(infinicore PRIVATE flash-attn-nvidia)
endif()

install(TARGETS infinicore LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/infiniop DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/infinicore DESTINATION include)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/infiniop.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/infinicore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/infinicore.hpp
    DESTINATION include
)

# ---- Python binding: _infinicore ----
if(INFINIOPS_BUILD_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Locate pybind11 via pip-installed package.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb_rc
    )
    if(NOT _pb_rc EQUAL 0 OR PYBIND11_CMAKE_DIR STREQUAL "")
        message(FATAL_ERROR "pybind11 not found (pip install pybind11)")
    endif()
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    file(GLOB_RECURSE INFINICORE_PYBIND_SRC CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infinicore/pybind11/*.cc"
    )

    pybind11_add_module(_infinicore MODULE ${INFINICORE_PYBIND_SRC})
    target_include_directories(_infinicore PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${SPDLOG_INCLUDE}"
        "${JSON_INCLUDE}"
    )
    target_link_libraries(_infinicore PRIVATE infinicore)
    target_compile_options(_infinicore PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    )

    set(INFINICORE_PY_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python/infinicore" CACHE PATH "infinicore python package install dir")
    install(TARGETS _infinicore LIBRARY DESTINATION "${INFINICORE_PY_PKG_DIR}/lib")
    # The three C++ shared libs are co-located with the InfiniLM binding (see top-level CMakeLists).
    # Resolve them via $ORIGIN/../../infinilm/lib relative to _infinicore.so.
    set_target_properties(_infinicore PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${INFINICORE_PY_PKG_DIR}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        # Primary location: sibling to InfiniLM (libinfinicore.so lives there).
        # $ORIGIN fallback supports installs without InfiniLM next door.
        INSTALL_RPATH "$ORIGIN/../../infinilm/lib;$ORIGIN"
    )
endif()
