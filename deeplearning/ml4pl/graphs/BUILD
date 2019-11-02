# Databases of graphs.

py_binary(
    name = "copy_database",
    srcs = ["copy_database.py"],
    deps = [
        ":graph_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
    ],
)

py_library(
    name = "database_exporters",
    srcs = ["database_exporters.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":graph_database",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:labtypes",
        "//labm8:prof",
        "//labm8:sqlutil",
    ],
)

py_test(
    name = "database_exporters_test",
    srcs = ["database_exporters_test.py"],
    deps = [
        ":database_exporters",
        "//labm8:app",
        "//labm8:sqlutil",
        "//labm8:test",
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
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:decorators",
        "//labm8:labdate",
        "//labm8:sqlutil",
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
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_database_reader",
    srcs = ["graph_database_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:prof",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_reader_test",
    srcs = ["graph_database_reader_test.py"],
    deps = [
        ":graph_database",
        ":graph_database_reader",
        "//labm8:app",
        "//labm8:test",
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
        "//labm8:app",
        "//labm8:decorators",
        "//labm8:humanize",
        "//labm8:prof",
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
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "graph_iterators",
    srcs = ["graph_iterators.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_iterators_test",
    srcs = ["graph_iterators_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterators",
        "//labm8:app",
        "//labm8:test",
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
        "//labm8:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "graph_query_test",
    srcs = ["graph_query_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_query",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_viz",
    srcs = ["graph_viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_iterators",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:labtypes",
        "//third_party/py/networkx",
    ],
)
