add_requires("pybind11")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

-- `InfiniOps` provides the operator kernels used by the `ops_shim` layer.
-- Its sources live outside this project so we locate them via `INFINIOPS_ROOT`
-- (the clone of `https://github.com/InfiniTensor/InfiniOps`). The library
-- object is expected at `$INFINIOPS_ROOT/build/src/libinfiniops.so`, built
-- with `WITH_TORCH=ON`.
local INFINIOPS_ROOT = os.getenv("INFINIOPS_ROOT")
if not INFINIOPS_ROOT or INFINIOPS_ROOT == "" then
    INFINIOPS_ROOT = os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/InfiniOps"
end

set_toolchains("gcc")

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

option("use-kv-caching")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile the path using the kv caching operator")
option_end()

if has_config("use-kv-caching") then
    add_defines("ENABLE_KV_CACHING")
end

option("use-classic-llama")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to using the classic LlamaForCausalLM")
option_end()

if has_config("use-classic-llama") then
    add_defines("USE_CLASSIC_LLAMA")
end

target("infinicore_infer")
    set_kind("shared")

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()

target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    add_includedirs("include", { public = false })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- `InfiniOps` headers (via the source tree) and built library. The
    -- `WITH_*` defines mirror the platforms `InfiniOps` was built with so
    -- that `ops_shim.cpp` registers the right `DeviceEnabled` specializations
    -- (see `csrc/ops_shim/ops_shim.cpp`).
    add_includedirs(INFINIOPS_ROOT.."/src", { public = false })
    add_linkdirs(INFINIOPS_ROOT.."/build/src")
    add_links("infiniops")
    add_rpathdirs(INFINIOPS_ROOT.."/build/src")
    add_defines("WITH_NVIDIA=1")

    -- Compile the `.cu` dispatch TU (`csrc/ops_shim/ops_shim_cuda.cu`)
    -- with `nvcc` so it can include `InfiniOps`' `cuda/nvidia/*/kernel.h`
    -- headers. Other files continue to build with the host C++ compiler.
    add_rules("cuda")
    add_cugencodes("native")

    -- Add src files
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")
    add_files("csrc/**.cu")

    set_installdir("python/infinilm")
target_end()
