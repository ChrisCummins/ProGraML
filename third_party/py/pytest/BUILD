# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "pytest",
    srcs = ["pytest.py"],
    deps = [
        requirement("pytest"),
        requirement("pytest-benchmark"),
        requirement("pytest-cov"),
        requirement("pytest-mock"),
        requirement("flaky"),
    ],
)
