# A wrapper around tensorflow pip package to support optional gpu.
#
# If a python target requires TensorFlow, it should depend on this package
# (i.e. //third_party/tensorflow), instead of requirement("tensorflow").
# This is because the pip package for TensorFlow with CUDA support has a
# different name.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    deps = [
        requirement("tensorflow"),
        requirement("numpy"),
        requirement("protobuf"),
    ],
)
