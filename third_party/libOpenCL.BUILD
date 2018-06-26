# The OpenCL ICD Loader project.
# See: https://github.com/KhronosGroup/OpenCL-ICD-Loader

licenses(["notice"])  # Apache 2.0.

cc_library(
    name = "libOpenCL",
    srcs = [
        "icd.c",
        "icd.h",
        "icd_dispatch.c",
        "icd_dispatch.h",
        "icd_linux.c",
    ],
    copts = [
        "-DCL_TARGET_OPENCL_VERSION=220",
        "-isystem external/opencl_220_headers",
    ],
    visibility = ["//visibility:public"],
    deps = ["@opencl_220_headers//:headers"],
)
