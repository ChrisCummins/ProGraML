# Protocol message schemas for ProGraML.
#
# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@build_stack_rules_proto//python:python_proto_library.bzl",
    "python_proto_library",
)

proto_library(
    name = "checkpoint",
    srcs = ["checkpoint.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "checkpoint_py",
    visibility = ["//visibility:public"],
    deps = [":checkpoint"],
)

cc_proto_library(
    name = "checkpoint_cc",
    visibility = ["//visibility:public"],
    deps = [":checkpoint"],
)

proto_library(
    name = "edge",
    srcs = ["edge.proto"],
    visibility = ["//visibility:public"],
    deps = [":features"],
)

python_proto_library(
    name = "edge_py",
    visibility = ["//visibility:public"],
    deps = [":edge"],
)

cc_proto_library(
    name = "edge_cc",
    visibility = ["//visibility:public"],
    deps = [":edge"],
)

proto_library(
    name = "epoch",
    srcs = ["epoch.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "epoch_py",
    visibility = ["//visibility:public"],
    deps = [":epoch"],
)

proto_library(
    name = "features",
    srcs = ["features.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "features_py",
    visibility = ["//visibility:public"],
    deps = [":features"],
)

cc_proto_library(
    name = "features_cc",
    visibility = ["//visibility:public"],
    deps = [":features"],
)

proto_library(
    name = "function",
    srcs = ["function.proto"],
    visibility = ["//visibility:public"],
    deps = [":features"],
)

python_proto_library(
    name = "function_py",
    visibility = ["//visibility:public"],
    deps = [":function"],
)

cc_proto_library(
    name = "function_cc",
    visibility = ["//visibility:public"],
    deps = [":function"],
)

proto_library(
    name = "ir",
    srcs = ["ir.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "ir_py",
    visibility = ["//visibility:public"],
    deps = [":ir"],
)

cc_proto_library(
    name = "ir_cc",
    visibility = ["//visibility:public"],
    deps = [":ir"],
)

proto_library(
    name = "program_graph_options",
    srcs = ["program_graph_options.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "program_graph_options_py",
    visibility = ["//visibility:public"],
    deps = [":program_graph_options"],
)

cc_proto_library(
    name = "program_graph_options_cc",
    visibility = ["//visibility:public"],
    deps = [":program_graph_options"],
)

cc_proto_library(
    name = "src_cc",
    visibility = ["//visibility:public"],
    deps = [":src"],
)

proto_library(
    name = "module",
    srcs = ["module.proto"],
    visibility = ["//visibility:public"],
    deps = [":features"],
)

python_proto_library(
    name = "module_py",
    visibility = ["//visibility:public"],
    deps = [":module"],
)

cc_proto_library(
    name = "module_cc",
    visibility = ["//visibility:public"],
    deps = [":module"],
)

proto_library(
    name = "node",
    srcs = ["node.proto"],
    visibility = ["//visibility:public"],
    deps = [":features"],
)

python_proto_library(
    name = "node_py",
    visibility = ["//visibility:public"],
    deps = [":node"],
)

cc_proto_library(
    name = "node_cc",
    visibility = ["//visibility:public"],
    deps = [":node"],
)

proto_library(
    name = "program_graph",
    srcs = ["program_graph.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":edge",
        ":features",
        ":function",
        ":module",
        ":node",
    ],
)

python_proto_library(
    name = "program_graph_py",
    visibility = ["//visibility:public"],
    deps = [":program_graph"],
)

cc_proto_library(
    name = "program_graph_cc",
    visibility = ["//visibility:public"],
    deps = [":program_graph"],
)

proto_library(
    name = "program_graph_features",
    srcs = ["program_graph_features.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":edge",
        ":features",
        ":function",
        ":node",
    ],
)

python_proto_library(
    name = "program_graph_features_py",
    visibility = ["//visibility:public"],
    deps = [":program_graph_features"],
)

cc_proto_library(
    name = "program_graph_features_cc",
    visibility = ["//visibility:public"],
    deps = [":program_graph_features"],
)

proto_library(
    name = "repo",
    srcs = ["repo.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "repo_py",
    visibility = ["//visibility:public"],
    deps = [":repo"],
)

cc_proto_library(
    name = "repo_cc",
    visibility = ["//visibility:public"],
    deps = [":repo"],
)

proto_library(
    name = "src",
    srcs = ["src.proto"],
    visibility = ["//visibility:public"],
)

python_proto_library(
    name = "src_py",
    visibility = ["//visibility:public"],
    deps = [":src"],
)
