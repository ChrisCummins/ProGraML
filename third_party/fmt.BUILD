# A modern formatting library. https://fmt.dev

cc_library(
    name = "fmt",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/fmt/*.h"]),
    copts = ["-Iexternal/fmt/include"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
