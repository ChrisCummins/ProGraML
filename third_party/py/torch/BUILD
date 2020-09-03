# A wrapper around pip package to pull in undeclared dependencies.

load("@programl_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-style

py_library(
    name = "torch",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("torch"),
        requirement("torch_cluster"),
        requirement("torch_geometric"),
        requirement("torch_scatter"),
        requirement("torch_sparse"),
        requirement("isodate"),
    ],
)
