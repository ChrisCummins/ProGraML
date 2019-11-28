# Embeddings for code comprehension.

py_library(
    name = "api",
    srcs = ["api.py"],
    data = ["//deeplearning/ncc/published_results:dictionary"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_preprocess",
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    data = ["//deeplearning/ncc/published_results:vocabulary"],
    deps = [
        ":api",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

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
        "//labm8/py:app",
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

py_test(
    name = "inst2vec_datagen_test",
    srcs = ["inst2vec_datagen_test.py"],
    deps = [
        ":inst2vec_datagen",
        "//labm8/py:app",
        "//labm8/py:test",
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
        "//labm8/py:app",
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
        "//labm8/py:app",
        "//third_party/py/bokeh",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/tensorflow",
        "//third_party/py/umap",
    ],
)

# Preprocess LLVM IR code to XFG for inst2vec training.
py_library(
    name = "inst2vec_preprocess",
    srcs = ["inst2vec_preprocess.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_utils",
        "//deeplearning/ncc:rgx_utils",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "inst2vec_preprocess_test",
    srcs = ["inst2vec_preprocess_test.py"],
    deps = [
        ":inst2vec_preprocess",
        "//labm8/py:app",
        "//labm8/py:test",
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
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/scipy",
    ],
)
