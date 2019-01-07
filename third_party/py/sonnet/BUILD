# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache

py_library(
    name = "sonnet",
    srcs = ["sonnet.py"],
    deps = [
        requirement("dm-sonnet"),
        requirement("tensorflow"),
    ],
)
