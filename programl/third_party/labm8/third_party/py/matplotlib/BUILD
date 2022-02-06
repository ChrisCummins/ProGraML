# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "matplotlib",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("matplotlib"),
        "//third_party/py/pyparsing",  # Implicit dependency.
        requirement("cycler"),  # Implicit dependency.
        requirement("kiwisolver"),  # Implicit dependency.
    ],
)
