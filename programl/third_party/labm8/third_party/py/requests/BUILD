# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "requests",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("requests"),
        requirement("chardet"),  # Implicit dependency.
        requirement("certifi"),  # Implicit dependency.
        requirement("idna"),  # Implicit dependency.
        "//third_party/py/urllib3",  # Implicit dependency.
    ],
)
