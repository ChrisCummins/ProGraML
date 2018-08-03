# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache.

py_library(
    name = "absl",
    srcs = ["absl.py"],
    deps = [
        requirement("absl-py"),
    ],
)
