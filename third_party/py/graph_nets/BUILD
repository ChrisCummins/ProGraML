# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache

py_library(
    name = "graph_nets",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("graph_nets"),
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
    ],
)
