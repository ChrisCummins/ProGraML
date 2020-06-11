# ﻿Neural Code Comprehension: A Learnable Representation of Code Semantics.
#
# Adapting the ncc codebase to run in my repository. The ncc implementation was
# developed by the authors of this paper:
#
#       ﻿Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
#       Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
#       https://doi.org/arXiv:1806.07336v3
#
# The two main changes required to make the code fit my use case is:
#   1. Support for execution within bazel sandbox.
#   2. Tests!

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

licenses(["notice"])  # BSD

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "rgx_utils",
    srcs = ["rgx_utils.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    data = ["//third_party/py/ncc/published_results:vocabulary"],
    visibility = ["//visibility:public"],
    deps = [
        ":rgx_utils",
        "//third_party/py/labm8",
        "//third_party/py/ncc/inst2vec:inst2vec_preprocess",
    ],
)

py_test(
    name = "vocabulary_test",
    srcs = ["vocabulary_test.py"],
    data = ["//third_party/py/ncc/published_results:vocabulary"],
    deps = [
        ":vocabulary",
        "//third_party/py/labm8",
    ],
)
