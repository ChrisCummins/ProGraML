# Package for defining labelled graph representations.

py_library(
    name = "database_exporters",
    srcs = ["database_exporters.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//labm8:app",
        "//labm8:humanize",
        "//labm8:labtypes",
        "//labm8:ppar",
        "//labm8:prof",
        "//labm8:sqlutil",
    ],
)

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled:__subpackages__"],
    deps = [
        ":database_exporters",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/bytecode:splitters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/graph_dict",
        "//deeplearning/ml4pl/graphs/labelled/liveness",
        "//deeplearning/ml4pl/graphs/labelled/reachability",
        "//deeplearning/ml4pl/graphs/unlabelled/cdfg:control_and_data_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:pbutil",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)
