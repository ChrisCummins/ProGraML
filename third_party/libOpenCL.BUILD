# The OpenCL ICD Loader project.
# See: https://github.com/KhronosGroup/OpenCL-ICD-Loader

licenses(["notice"])  # Apache 2.0.

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sources",
    srcs = [
        "icd.c",
        "icd.h",
        "icd_dispatch.c",
        "icd_dispatch.h",
        "icd_linux.c",
    ],
    visibility = ["//visibility:public"],
)

# The two libOpenCL targets build the same code, but for different purposes.
# :libOpenCL is a library that can be used in the deps attribute of cc_binary
# targets that need to link against OpenCL. The :libOpenCL.so target produces
# a shared object which can be used in the data attribute of other targets.

cc_library(
    name = "libOpenCL",
    srcs = [":sources"],
    copts = [
        "-DCL_TARGET_OPENCL_VERSION=220",
        "-isystem external/opencl_220_headers",
    ],
    visibility = ["//visibility:public"],
    deps = ["@opencl_220_headers//:headers"],
)

cc_binary(
    name = "libOpenCL.so",
    srcs = [":sources"],
    copts = [
        "-DCL_TARGET_OPENCL_VERSION=220",
        "-isystem external/opencl_220_headers",
    ],
    linkopts = select({
        "//:darwin": [],
        "//conditions:default": [
            "-ldl",
            "-lpthread",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["@opencl_220_headers//:headers"],
)
