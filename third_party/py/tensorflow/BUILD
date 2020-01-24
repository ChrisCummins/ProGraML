# A wrapper around tensorflow pip package to support optional gpu.
#
# If a python target requires TensorFlow, it should depend on this package
# (i.e. //third_party/py/tensorflow), instead of requirement("tensorflow").
# This is because the pip package for TensorFlow with CUDA support has a
# different name.
#
# Use:
#
#     from third_party.py.tensorflow import tf
#
# to import Tensorflow rather than "import tensorflow as tf" due to a bug in
# packing Tensorflow as a pip dependency for bazel.
# See github.com/bazelbuild/rules_python/issues/71

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0.

exports_files(["LICENSE"])

py_library(
    name = "tensorflow",
    srcs = [":tf.py"],
    deps = [
        requirement("tensorflow"),
        "//third_party/py/numpy",
        "//third_party/py/protobuf",
    ],
)

py_test(
    name = "smoke_test",
    srcs = ["smoke_test.py"],
    deps = [
        ":tensorflow",
        "//labm8/py:app",
        "//:getconfig",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
        "//labm8/py:test",
    ],
)

py_test(
    name = "overview",
    srcs = ["overview.py"],
    deps = [":tensorflow"],
)
