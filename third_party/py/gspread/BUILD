# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "gspread",
    srcs = ["gspread.py"],
    deps = [
        requirement("gspread"),
    ],
)
