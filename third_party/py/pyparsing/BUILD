# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "pyparsing",
    srcs = ["pyparsing.py"],
    deps = [
        requirement("pyparsing"),
    ],
)
