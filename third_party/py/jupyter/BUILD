# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "jupyter",
    srcs = ["jupyter.py"],
    deps = [
        requirement("jupyter-client"),
        requirement("jupyter_http_over_ws"),
        requirement("notebook"),
        requirement("nbformat"),
        # requirement("nbconvert"),
        requirement("prometheus_client"),
        "//third_party/py/jinja2",
    ],
)
