cc_library(
    name = "main",
    srcs = glob(["lib/intel64/gcc4.7/libtbb.so*"]),
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
