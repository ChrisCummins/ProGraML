# Graphs labelled with dataflow analyses.

py_binary(
    name = "annotate",
    srcs = ["annotate.py"],
    deps = [
        ":data_flow_graphs",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/alias_set",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/liveness",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/reachability",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/subexpressions",
        "//deeplearning/ml4pl/testing:test_annotators",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)

py_test(
    name = "annotate_binary_test",
    size = "enormous",
    srcs = ["annotate_binary_test.py"],
    data = [":annotate"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:test",
    ],
)

py_library(
    name = "data_flow_graphs",
    srcs = ["data_flow_graphs.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs/labelled:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
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
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/alias_set",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/liveness",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/reachability",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/subexpressions",
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

# py3_image(
#     name = "make_data_flow_analysis_dataset_image",
#     srcs = ["make_data_flow_analysis_dataset.py"],
#     base = "@phd_base//image",
#     main = "make_data_flow_analysis_dataset.py",
#     deps = [":make_data_flow_analysis_dataset"],
# )

py_test(
    name = "make_data_flow_analysis_dataset_test",
    srcs = ["make_data_flow_analysis_dataset_test.py"],
    deps = [
        ":make_data_flow_analysis_dataset",
        "//labm8/py:test",
    ],
)
