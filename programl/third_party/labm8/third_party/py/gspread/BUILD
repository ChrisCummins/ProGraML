# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "gspread",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("gspread"),
        requirement("gspread-dataframe"),
        "//third_party/py/requests",
    ],
)
