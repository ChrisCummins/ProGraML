# Databases of graphs.

py_library(
    name = "graph_database",
    srcs = ["graph_database.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:sqlutil",
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
        "//labm8:sqlutil",
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
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        "//labm8:app",
        "//labm8:decorators",
        "//labm8:prof",
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
