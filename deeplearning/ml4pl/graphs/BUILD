# Databases of graphs.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_library(
    name = "graph_iterators",
    srcs = ["graph_iterators.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_iterators_test",
    srcs = ["graph_iterators_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterators",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_query",
    srcs = ["graph_query.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterators",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_query_test",
    srcs = ["graph_query_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_query",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_viz",
    srcs = ["graph_viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterators",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:labtypes",
        "//third_party/py/networkx",
    ],
)

proto_library(
    name = "programl_pb",
    srcs = ["programl.proto"],
    visibility = [
        "//deeplearning/ml4pl/seq:__subpackages__",
    ],
)

python_proto_library(
    name = "programl_pb_py",
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [":programl_pb"],
)

cc_proto_library(
    name = "programl_pb_cc",
    visibility = [
        "//deeplearning/ml4pl/seq:__subpackages__",
    ],
    deps = [":programl_pb"],
)

py_binary(
    name = "programl",
    srcs = ["programl.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [
        ":programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "programl_test",
    size = "enormous",
    srcs = ["programl_test.py"],
    deps = [
        ":programl",
        ":programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)
