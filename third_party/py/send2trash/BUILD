# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "send2trash",
    srcs = ["send2trash.py"],
    deps = [
        requirement("send2trash"),
    ],
)
