load("@programl_requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0.

exports_files(["LICENSE"])

py_library(
    name = "tensorflow",
    srcs = ["//third_party/py:empty.py"],
    deps = [
        requirement("tensorflow"),
        "//third_party/py/numpy",
        "//third_party/py/protobuf",
    ],
)
