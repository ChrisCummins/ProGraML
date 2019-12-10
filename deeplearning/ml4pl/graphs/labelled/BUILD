# Package for defining labelled graph representations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "graph_batcher",
    srcs = ["graph_batcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database_reader",
        ":graph_tuple",
        "//labm8/py:app",
        "//labm8/py:progress",
    ],
)

py_test(
    name = "graph_batcher_test",
    size = "enormous",
    srcs = ["graph_batcher_test.py"],
    deps = [
        ":graph_batcher",
        ":graph_tuple",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "graph_database_reader",
    srcs = ["graph_database_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_reader_test",
    size = "enormous",
    srcs = ["graph_database_reader_test.py"],
    deps = [
        ":graph_database_reader",
        ":graph_tuple_database",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)

py_test(
    name = "graph_database_reader_benchmark_test",
    size = "enormous",
    srcs = ["graph_database_reader_benchmark_test.py"],
    deps = [
        ":graph_database_reader",
        ":graph_tuple_database",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_tuple_test",
    size = "enormous",
    srcs = ["graph_tuple_test.py"],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:fs",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "graph_tuple_database",
    srcs = ["graph_tuple_database.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/models:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_tuple_database_test",
    size = "enormous",
    srcs = ["graph_tuple_database_test.py"],
    deps = [
        ":graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
