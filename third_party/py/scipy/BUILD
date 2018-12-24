# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "scipy",
    srcs = ["scipy.py"],
    deps = [
        requirement("scipy"),
    ],
)

py_library(
    name = "scikit_image",
    srcs = ["scipy.py"],
    deps = [
        requirement("scikit-image"),
    ],
)

py_library(
    name = "scikit_learn",
    srcs = ["scipy.py"],
    deps = [
        requirement("scikit-learn"),
    ],
)
