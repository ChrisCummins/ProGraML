# Package for defining labelled graph representations.

py_binary(
    name = "make_data_flow_analysis_dataset",
    srcs = ["make_data_flow_analysis_dataset.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:database_exporters",
        "//deeplearning/ml4pl/graphs:graph_database",
        "//deeplearning/ml4pl/graphs/labelled/datadep:data_dependence",
        "//deeplearning/ml4pl/graphs/labelled/domtree:dominator_tree",
        "//deeplearning/ml4pl/graphs/labelled/liveness",
        "//deeplearning/ml4pl/graphs/labelled/reachability",
        "//deeplearning/ml4pl/graphs/labelled/subexpressions",
        "//labm8:app",
        "//third_party/py/numpy",
    ],
)
