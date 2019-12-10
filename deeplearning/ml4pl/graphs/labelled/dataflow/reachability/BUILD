# Labelled graphs for reachability analysis.

py_library(
    name = "reachability",
    srcs = ["reachability.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "reachability_test",
    size = "enormous",
    srcs = ["reachability_test.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
    ],
    deps = [
        ":reachability",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/migrate:networkx_to_protos",
        "//labm8/py:bazelutil",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
