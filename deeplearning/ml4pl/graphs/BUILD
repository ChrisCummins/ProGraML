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
