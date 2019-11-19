# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache

py_library(
    name = "sonnet",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("dm-sonnet"),
        "//third_party/py/tensorflow",
    ],
)
