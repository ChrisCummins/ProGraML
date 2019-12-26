# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache v2

py_library(
    name = "fasteners",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("fasteners"),
        requirement("monotonic"),  # Implicit dependency.
    ],
)
