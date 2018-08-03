# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        requirement("keras"),
    ],
)
