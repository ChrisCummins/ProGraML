# ﻿Neural Code Comprehension: A Learnable Representation of Code Semantics.
#
# Adapting the ncc codebase to run in my repository. The ncc implementation was
# developed by the authors of this paper:
#
#       ﻿Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
#       Comprehension: A Learnable Representation of Code Semantics. In NeurIPS.
#       https://doi.org/arXiv:1806.07336v3

licenses(["notice"])  # BSD

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "rgx_utils",
    srcs = ["rgx_utils.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
)

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    data = ["//deeplearning/ncc/published_results:embeddings"],
    deps = [
        ":rgx_utils",
        ":vocabulary",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/wget",
    ],
)

py_binary(
    name = "train_task_classifyapp",
    srcs = ["train_task_classifyapp.py"],
    deps = [
        ":rgx_utils",
        ":task_utils",
        ":vocabulary",
        "//labm8:fs",
        "//third_party/py/absl",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy:scikit_learn",
        "//third_party/py/tensorflow",
    ],
)

sh_test(
    name = "train_task_classifyapp_smoke_test",
    srcs = ["train_task_classifyapp_smoke_test.sh"],
    data = [
        ":train_task_classifyapp",
        "//deeplearning/ncc/published_results:embeddings",
        "//deeplearning/ncc/published_results:vocabulary",
    ],
)

py_binary(
    name = "train_task_devmap",
    srcs = ["train_task_devmap.py"],
    deps = [
        ":rgx_utils",
        ":task_utils",
        ":vocabulary",
        "//labm8:fs",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

sh_test(
    name = "train_task_devmap_smoke_test",
    srcs = ["train_task_devmap_smoke_test.sh"],
    data = [
        ":train_task_devmap",
        "//deeplearning/ncc/published_results:embeddings",
        "//deeplearning/ncc/published_results:vocabulary",
    ],
)

py_binary(
    name = "train_task_threadcoarsening",
    srcs = ["train_task_threadcoarsening.py"],
    deps = [
        ":rgx_utils",
        ":task_utils",
        ":vocabulary",
        "//labm8:fs",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/scipy:scikit_learn",
    ],
)

sh_test(
    name = "train_task_threadcoarsening_smoke_test",
    srcs = ["train_task_threadcoarsening_smoke_test.sh"],
    data = [
        ":train_task_threadcoarsening",
        "//deeplearning/ncc/published_results:embeddings",
    ],
)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    deps = [
        "//third_party/py/absl",
    ],
)

py_test(
    name = "vocabulary_test",
    srcs = ["vocabulary_test.py"],
    data = ["//deeplearning/ncc/published_results:vocabulary"],
    deps = [
        ":vocabulary",
        "//labm8:bazelutil",
        "//third_party/py/absl",
        "//third_party/py/pytest",
    ],
)
