# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "pytest",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("pytest"),
        requirement("pluggy"),  # Implicit dependency of pytest.
        requirement("importlib-metadata"),  # Implicit dependency of pytest.
        requirement("zipp"),  # Implicit dependency of pytest.
        requirement("more-itertools"),  # Implicit dependency of zipp.
        requirement("py"),  # Implicit dependency of pytest.
        requirement("packaging"),  # Implicit dependency of pytest.
        # Pytest plugins:
        requirement("pytest-benchmark"),
        requirement("pytest-cov"),
        requirement("pytest-mock"),
        requirement("flaky"),
        requirement("coverage"),  # Implicit dependency of pytest-cov.
    ],
)
