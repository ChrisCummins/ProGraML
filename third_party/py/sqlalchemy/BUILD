# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "sqlalchemy",
    srcs = ["sqlalchemy.py"],
    deps = [
        requirement("sqlalchemy"),
        # Implicit dependency when connecting to MySQL databases.
        "//third_party/py/mysql",
    ],
)
