# A wrapper around pip package to pull in undeclared dependencies.

load("@programl_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "keras_preprocessing",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("keras_preprocessing"),
        "//third_party/py/numpy",
    ],
)
