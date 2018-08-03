# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "humanize",
    srcs = ["humanize.py"],
    deps = [
        requirement("humanize"),
    ],
)
