# Embeddings for code comprehension.

# Helper module for analogy generation.
py_library(
    name = "inst2vec_analogygen",
    srcs = ["inst2vec_analogygen.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        "//third_party/py/jinja2",
    ],
)

# Execution flags for inst2vec parameters.
py_library(
    name = "inst2vec_appflags",
    srcs = ["inst2vec_appflags.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        "//third_party/py/absl",
    ],
)

# Generate dataset for inst2vec training.
py_library(
    name = "inst2vec_datagen",
    srcs = ["inst2vec_datagen.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        "//third_party/py/wget",
    ],
)

# inst2vec embedding training.
py_library(
    name = "inst2vec_embedding",
    srcs = ["inst2vec_embedding.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        ":inst2vec_evaluate",
        ":inst2vec_utils",
        "//third_party/py/absl",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

# inst2vec embedding intrinsic evaluation.
py_library(
    name = "inst2vec_evaluate",
    srcs = ["inst2vec_evaluate.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        ":inst2vec_analogygen",
        ":inst2vec_utils",
        "//deeplearning/ncc:rgx_utils",
        "//third_party/py/absl",
        "//third_party/py/bokeh",
        "//third_party/py/numpy",
        "//third_party/py/scipy:scikit_learn",
        "//third_party/py/tensorflow",
        "//third_party/py/umap",
    ],
)

# Preprocess LLVM IR code to XFG for inst2vec training.
py_library(
    name = "inst2vec_preprocess",
    srcs = ["inst2vec_preprocess.py"],
    visibility = ["//deeplearning/ncc:__subpackages__"],
    deps = [
        ":inst2vec_utils",
        "//deeplearning/ncc:rgx_utils",
        "//third_party/py/networkx",
    ],
)

# inst2vec utility functions
py_library(
    name = "inst2vec_utils",
    srcs = ["inst2vec_utils.py"],
)

# Construct vocabulary from XFGs and indexify the data set.
py_library(
    name = "inst2vec_vocabulary",
    srcs = ["inst2vec_vocabulary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_utils",
        "//deeplearning/ncc:rgx_utils",
        "//third_party/py/absl",
        "//third_party/py/networkx",
        "//third_party/py/scipy",
    ],
)
