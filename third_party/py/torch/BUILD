# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-style

py_library(
    name = "torch",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("torch"),
    ],
)
