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

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:database_exporters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/alias_set",
        "//deeplearning/ml4pl/graphs/labelled/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/liveness",
        "//deeplearning/ml4pl/graphs/labelled/polyhedra",
        "//deeplearning/ml4pl/graphs/labelled/reachability",
        "//deeplearning/ml4pl/graphs/labelled/subexpressions",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
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
