# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "jupyter",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("jupyter-client"),
        requirement("jupyter_http_over_ws"),
        requirement("notebook"),
        requirement("nbformat"),
        requirement("prometheus_client"),
        "//third_party/py/jinja2",
    ],
)
