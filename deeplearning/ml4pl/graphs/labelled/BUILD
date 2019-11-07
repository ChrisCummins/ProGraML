# Package for defining labelled graph representations.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

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
        "//deeplearning/ml4pl/graphs/labelled/reachability",
        "//deeplearning/ml4pl/graphs/labelled/subexpressions",
        "//labm8:app",
        "//third_party/py/numpy",
    ],
)

py3_image(
    name = "make_data_flow_analysis_dataset_image",
    srcs = ["make_data_flow_analysis_dataset.py"],
    base = "@base//image",
    main = "make_data_flow_analysis_dataset.py",
    deps = [":make_data_flow_analysis_dataset"],
)

py_test(
    name = "make_data_flow_analysis_dataset_test",
    srcs = ["make_data_flow_analysis_dataset_test.py"],
    deps = [
        ":make_data_flow_analysis_dataset",
        "//labm8:test",
    ],
)
