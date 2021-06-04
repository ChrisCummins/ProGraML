# Embeddings for code comprehension.

py_library(
    name = "inst2vec_preprocess",
    srcs = ["inst2vec_preprocess.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":inst2vec_utils",
        "//programl/third_party/ncc:rgx_utils",
    ],
)

py_library(
    name = "inst2vec_utils",
    srcs = ["inst2vec_utils.py"],
)
