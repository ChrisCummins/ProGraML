# A wrapper around pip package to pull in undeclared dependencies.

load("@programl_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "labm8",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        # requirement("labm8"),
        requirement("absl-py"),
    ],
)
