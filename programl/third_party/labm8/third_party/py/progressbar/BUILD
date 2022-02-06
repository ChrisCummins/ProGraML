# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "progressbar",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("progressbar2"),
        # Progress bar has an undeclared dependency on python_utils.time module,
        # see issue #27.
        requirement("python-utils"),
    ],
)
