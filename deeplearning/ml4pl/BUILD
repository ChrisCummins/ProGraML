# Experiments in learning reachability analysis.

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")
load("//tools/bzl:exports.bzl", "exports_repo")

exports_repo(
    name = "export",
    extra_files = [
        "docs/wip_graph/notebooks/01_deeptune_adversary.ipynb",
        "docs/wip_graph/notebooks/02_learning_reachability.ipynb",
        "docs/wip_graph/notebooks/03_opencl_graph_representations.ipynb",
        "docs/wip_graph/notebooks/04_evaluating_graphnet.ipynb",
        "docs/wip_graph/notebooks/05_evaluating_lstm_reachability.ipynb",
        "docs/wip_graph/notebooks/06_neural_network_reachability.ipynb",
        "docs/wip_graph/notebooks/run_notebook.sh",
    ],
    github_repo = "ml4pl",
    targets = [
        "//deeplearning/ml4pl/...",
        "//docs/wip_graph/notebooks/...",
    ],
)

proto_library(
    name = "ml4pl_pb",
    srcs = ["ml4pl.proto"],
)

python_proto_library(
    name = "ml4pl_pb_py",
    visibility = ["//visibility:public"],
    deps = [":ml4pl_pb"],
)
