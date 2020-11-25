# A wrapper around pip package to pull in undeclared dependencies.

load("@tests_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "pytest",
    testonly = 1,
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("pytest"),
        requirement("attrs"),
        requirement("iniconfig"),
        requirement("packaging"),
        requirement("pluggy"),
        requirement("py"),
        requirement("pyparsing"),
        requirement("zipp"),
        requirement("importlib-metadata"),
        requirement("pytest-shard"),
        requirement("pytest-benchmark"),
    ],
)
