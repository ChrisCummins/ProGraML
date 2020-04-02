cc_library(
    name = "tbb",
    visibility = ["//visibility:public"],
    deps = select({
        "//:darwin": ["@tbb_mac//:main"],
        "//conditions:default": ["@tbb_lin//:main"],
    }),
)
