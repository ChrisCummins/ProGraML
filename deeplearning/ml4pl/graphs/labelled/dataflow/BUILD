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
    name = "annotate_test",
    srcs = ["annotate_test.py"],
    data = [":annotate"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:test",
    ],
)

py_test(
    name = "annotators_benchmark_test",
    size = "enormous",
    timeout = "eternal",
    srcs = ["annotators_benchmark_test.py"],
    deps = [
        ":annotate",
        ":data_flow_graphs",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//labm8/py:prof",
        "//labm8/py:test",
        "//third_party/py/networkx",
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
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

sh_binary(
    name = "flaky_make_data_flow_analysis_dataset",
    srcs = ["flaky_make_data_flow_analysis_dataset.sh"],
    data = [":make_data_flow_analysis_dataset"],
)

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    data = [":annotate"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        ":annotate",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:ppar",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/psutil",
        "//third_party/py/sqlalchemy",
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
    size = "enormous",
    srcs = ["make_data_flow_analysis_dataset_test.py"],
    deps = [
        ":make_data_flow_analysis_dataset",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//deeplearning/ml4pl/testing:random_unlabelled_graph_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:app",
        "//labm8/py:progress",
        "//labm8/py:test",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "split",
    srcs = ["split.py"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_database_reader",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple_database",
        "//deeplearning/ml4pl/ir:ir_database",
        "//deeplearning/ml4pl/ir:split",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:labtypes",
        "//labm8/py:prof",
        "//third_party/py/numpy",
        "//third_party/py/scikit_learn",
        "//third_party/py/sqlalchemy",
    ],
)
