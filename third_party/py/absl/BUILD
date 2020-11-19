# A wrapper around pip package to pull in undeclared dependencies.

load("@programl_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "absl",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("absl-py"),
        requirement("six"),
    ],
)
