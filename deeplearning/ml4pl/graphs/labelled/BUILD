# Package for defining labelled graph representations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_library(
    name = "data_flow_graphs",
    srcs = ["data_flow_graphs.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "data_flow_graphs_test",
    srcs = ["data_flow_graphs_test.py"],
    deps = [
        ":data_flow_graphs",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "graph_batcher",
    srcs = ["graph_batcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database",
        ":graph_database_reader",
        ":graph_database_stats",
        ":graph_tuple",
        "//deeplearning/ml4pl/models:log_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_batcher_test",
    srcs = ["graph_batcher_test.py"],
    deps = [
        ":graph_batcher",
        ":graph_database",
        "//labm8/py:app",
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
    timeout = "long",
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
    timeout = "long",
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
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_tuple_test",
    srcs = ["graph_tuple_test.py"],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "graph_tuple_database",
    srcs = ["graph_tuple_database.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/models:__subpackages__",
    ],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs/labelled:data_flow_graphs",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_tuple_database_test",
    srcs = ["graph_tuple_database_test.py"],
    deps = [
        ":graph_tuple",
        ":graph_tuple_database",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:random_cdfg_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/alias_set",
        "//deeplearning/ml4pl/graphs/labelled/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/liveness",
        "//deeplearning/ml4pl/graphs/labelled/polyhedra",
        "//deeplearning/ml4pl/graphs/labelled/reachability",
        "//deeplearning/ml4pl/graphs/labelled/subexpressions",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:prof",
        "//labm8/py:progress",
        "//third_party/py/networkx",
        "//third_party/py/sqlalchemy",
        "//third_party/py/tqdm",
    ],
)

py3_image(
    name = "make_data_flow_analysis_dataset_image",
    srcs = ["make_data_flow_analysis_dataset.py"],
    base = "@phd_base//image",
    main = "make_data_flow_analysis_dataset.py",
    deps = [":make_data_flow_analysis_dataset"],
)

py_test(
    name = "make_data_flow_analysis_dataset_test",
    srcs = ["make_data_flow_analysis_dataset_test.py"],
    deps = [
        ":make_data_flow_analysis_dataset",
        "//labm8/py:test",
    ],
)
