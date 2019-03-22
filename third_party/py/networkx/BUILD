# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "networkx",
    srcs = ["networkx.py"],
    deps = [
        requirement("networkx"),
        requirement("decorator"),  # Implicit dependency.
    ],
)
