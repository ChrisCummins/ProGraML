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

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

licenses(["notice"])  # BSD

exports_files([
    "LICENSE",
    "README.md",
])

py_proto_library(
    name = "inst2vec_py_pb2",
    protos = ["inst2vec.proto"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/protobuf"],
)

py_library(
    name = "rgx_utils",
    srcs = ["rgx_utils.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
)

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    data = ["//deeplearning/ncc/published_results:embeddings"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_py_pb2",
        ":rgx_utils",
        ":vocabulary",
        "//labm8:app",
        "//labm8:bazelutil",
        "//third_party/py/numpy",
        "//third_party/py/wget",
    ],
)

py_test(
    name = "task_utils_test",
    srcs = ["task_utils_test.py"],
    data = ["//deeplearning/ncc/published_results:vocabulary"],
    deps = [
        ":task_utils",
        ":vocabulary",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "train_inst2vec",
    srcs = ["train_inst2vec.py"],
    deps = [
        "//deeplearning/ncc/inst2vec:inst2vec_appflags",
        "//deeplearning/ncc/inst2vec:inst2vec_datagen",
        "//deeplearning/ncc/inst2vec:inst2vec_embedding",
        "//deeplearning/ncc/inst2vec:inst2vec_evaluate",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//deeplearning/ncc/inst2vec:inst2vec_vocabulary",
        "//labm8:app",
    ],
)

# TODO(cec): Create tiny dataset for < 60 second run.
#sh_test(
#    name = "train_inst2vec_smoke_test",
#    srcs = ["train_inst2vec_smoke_test.sh"],
#    data = [
#        ":train_inst2vec",
#    ],
#)

py_binary(
    name = "train_task_classifyapp",
    srcs = ["train_task_classifyapp.py"],
    deps = [
        ":task_utils",
        ":vocabulary",
        "//labm8:app",
        "//labm8:fs",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy:scikit_learn",
        "//third_party/py/tensorflow",
    ],
)

# TODO(cec): Create tiny dataset for < 60 second run.
#sh_test(
#    name = "train_task_classifyapp_smoke_test",
#    size = "enormous",
#    srcs = ["train_task_classifyapp_smoke_test.sh"],
#    data = [
#        ":train_task_classifyapp",
#        "//deeplearning/ncc/published_results:embeddings",
#        "//deeplearning/ncc/published_results:vocabulary",
#    ],
#)

py_binary(
    name = "train_task_devmap",
    srcs = ["train_task_devmap.py"],
    data = ["//deeplearning/ncc/published_results:task_devmap"],
    deps = [
        ":task_utils",
        ":vocabulary",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

# TODO(cec): Create tiny dataset for < 60 second run.
#sh_test(
#    name = "train_task_devmap_smoke_test",
#    srcs = ["train_task_devmap_smoke_test.sh"],
#    data = [
#        ":train_task_devmap",
#        "//deeplearning/ncc/published_results:embeddings",
#        "//deeplearning/ncc/published_results:vocabulary",
#    ],
#)

py_binary(
    name = "train_task_threadcoarsening",
    srcs = ["train_task_threadcoarsening.py"],
    data = ["//deeplearning/ncc/published_results:task_threadcoarsening"],
    deps = [
        ":task_utils",
        ":vocabulary",
        "//labm8:app",
        "//labm8:fs",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy:scikit_learn",
    ],
)

# TODO(cec): Create tiny dataset for < 60 second run.
#sh_test(
#    name = "train_task_threadcoarsening_smoke_test",
#    srcs = ["train_task_threadcoarsening_smoke_test.sh"],
#    data = [
#        ":train_task_threadcoarsening",
#        "//deeplearning/ncc/published_results:embeddings",
#        "//deeplearning/ncc/published_results:vocabulary",
#    ],
#)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_py_pb2",
        ":rgx_utils",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8:app",
        "//labm8:decorators",
    ],
)

py_test(
    name = "vocabulary_test",
    srcs = ["vocabulary_test.py"],
    data = ["//deeplearning/ncc/published_results:vocabulary"],
    deps = [
        ":vocabulary",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:test",
    ],
)
