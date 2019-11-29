# Databases of graphs.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

py_library(
    name = "conftest",
    testonly = 1,
    srcs = ["conftest.py"],
    deps = [
        ":graph_database",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "copy_database",
    srcs = ["copy_database.py"],
    deps = [
        ":graph_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
    ],
)

py_library(
    name = "database_exporters",
    srcs = ["database_exporters.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":graph_database",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/bytecode:splitters",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
        "//labm8/py:system",
    ],
)

py_test(
    name = "database_exporters_test",
    srcs = ["database_exporters_test.py"],
    deps = [
        ":database_exporters",
        "//labm8/py:app",
        "//labm8/py:sqlutil",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_database",
    srcs = ["graph_database.py"],
    data = ["//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:embeddings"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_query",
        "//deeplearning/ml4pl/graphs/labelled/graph_tuple",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:sqlutil",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_test",
    srcs = ["graph_database_test.py"],
    deps = [
        ":graph_database",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_database_reader",
    srcs = ["graph_database_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_reader_test",
    timeout = "long",
    srcs = ["graph_database_reader_test.py"],
    deps = [
        ":conftest",
        ":graph_database",
        ":graph_database_reader",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_database_reader_benchmark_test",
    timeout = "long",
    srcs = ["graph_database_reader_benchmark_test.py"],
    deps = [
        ":conftest",
        ":graph_database",
        ":graph_database_reader",
        "//labm8/py:test",
    ],
)

py_library(
    name = "graph_database_stats",
    srcs = ["graph_database_stats.py"],
    data = [
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings:embeddings",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_stats_test",
    srcs = ["graph_database_stats_test.py"],
    deps = [
        ":graph_database",
        ":graph_database_stats",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
    ],
)

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
)

python_proto_library(
    name = "programl_pb_py",
    visibility = ["//visibility:public"],
    deps = [":programl_pb"],
)

py_library(
    name = "programl",
    srcs = ["programl.py"],
    deps = [
        ":programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "programl_test",
    srcs = ["programl_test.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
    ],
    deps = [
        ":programl",
        ":programl_pb_py",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)
