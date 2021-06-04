py_library(
    name = "programl",
    srcs = [
        "__init__.py",
        "version.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/analysis/py:analysis",
        "//programl/graph/format/py:format",
        "//programl/graph/py:program_graph_builder",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/ir/llvm/py:llvm",
        "//programl/ir/xla/py:xla",
        "//programl/proto:programl_py",
        "//programl/third_party/tensorflow:xla_py",
        "//programl/util/py:util",
    ],
)

genrule(
    name = "version_header-gen",
    srcs = ["//:version.txt"],
    outs = ["version.h"],
    cmd = "echo \"#define PROGRAML_VERSION \\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

genrule(
    name = "version_py-gen",
    srcs = ["//:version.txt"],
    outs = ["version.py"],
    cmd = "echo \"PROGRAML_VERSION=\\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

cc_library(
    name = "version_header",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)
