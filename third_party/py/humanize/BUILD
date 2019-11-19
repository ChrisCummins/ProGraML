# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "humanize",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("humanize"),
    ],
)
