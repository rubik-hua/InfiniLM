cmake_minimum_required(VERSION 3.18)
project(infinirt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INFINIRT_ENABLE_CPU    "Enable CPU backend"    ON)
option(INFINIRT_ENABLE_NVIDIA "Enable NVIDIA backend" ON)
option(INFINIRT_ENABLE_OMP    "Enable OpenMP"         ON)

set(INFINILM_TP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party" CACHE PATH "third_party root")
set(SPDLOG_INCLUDE "${INFINILM_TP_ROOT}/spdlog/include")
set(JSON_INCLUDE   "${INFINILM_TP_ROOT}/json/single_include")

file(GLOB UTILS_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cc")
file(GLOB INFINIRT_CORE_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infinirt/*.cc")

set(INFINIRT_BACKEND_SRC "")
set(INFINIRT_BACKEND_CU_SRC "")
set(INFINIRT_DEFS "")
set(INFINIRT_LINK_LIBS "")

if(INFINIRT_ENABLE_CPU)
    file(GLOB _cpu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infinirt/cpu/*.cc")
    list(APPEND INFINIRT_BACKEND_SRC ${_cpu_src})
    list(APPEND INFINIRT_DEFS ENABLE_CPU_API)
endif()

if(INFINIRT_ENABLE_NVIDIA)
    enable_language(CUDA)
    file(GLOB _cu_src CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/infinirt/cuda/*.cu")
    list(APPEND INFINIRT_BACKEND_CU_SRC ${_cu_src})
    list(APPEND INFINIRT_DEFS ENABLE_NVIDIA_API)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND INFINIRT_LINK_LIBS CUDA::cudart)
endif()

add_library(infinirt SHARED
    ${UTILS_SRC}
    ${INFINIRT_CORE_SRC}
    ${INFINIRT_BACKEND_SRC}
    ${INFINIRT_BACKEND_CU_SRC}
)

target_include_directories(infinirt
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${SPDLOG_INCLUDE}"
            "${JSON_INCLUDE}"
)

target_compile_definitions(infinirt PRIVATE ${INFINIRT_DEFS})
target_compile_options(infinirt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
)

if(INFINIRT_ENABLE_NVIDIA)
    set_target_properties(infinirt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set_target_properties(infinirt PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_options(infinirt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    )
endif()

if(INFINIRT_ENABLE_OMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(infinirt PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(infinirt PRIVATE ENABLE_OMP)
    endif()
endif()

target_link_libraries(infinirt PRIVATE ${INFINIRT_LINK_LIBS})

install(TARGETS infinirt LIBRARY DESTINATION lib)
# infinicore.h is shipped by the ops subproject (which owns the canonical copy);
# only the runtime-private API header is installed here.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/infinirt.h DESTINATION include)
