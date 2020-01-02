# A Python module for getting the GPU status from NVIDA GPUs using nvidia-smi
# programmically in Python.
#
# https://github.com/anderskm/gputil

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_library(
    name = "gputil",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("GPUtil"),
    ],
)
