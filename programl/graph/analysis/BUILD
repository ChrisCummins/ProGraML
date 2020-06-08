# This package defines analyses for program graphs.
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

cc_library(
    name = "analysis",
    srcs = ["analysis.cc"],
    hdrs = ["analysis.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":datadep",
        ":dominance",
        ":liveness",
        ":reachability",
        ":subexpressions",
        "//labm8/cpp:status",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_library(
    name = "data_flow_pass",
    srcs = ["data_flow_pass.cc"],
    hdrs = ["data_flow_pass.h"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_library(
    name = "datadep",
    srcs = ["datadep.cc"],
    hdrs = ["datadep.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_test(
    name = "datadep_test",
    srcs = ["datadep_test.cc"],
    deps = [
        ":datadep",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
        "//programl/test:analysis_test_macros",
        "//programl/test:llvm_program_graphs",
    ],
)

cc_library(
    name = "dominance",
    srcs = ["dominance.cc"],
    hdrs = ["dominance.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "dominance_test",
    srcs = ["dominance_test.cc"],
    deps = [
        ":dominance",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
        "//programl/test:analysis_test_macros",
        "//programl/test:llvm_program_graphs",
    ],
)

cc_library(
    name = "liveness",
    srcs = ["liveness.cc"],
    hdrs = ["liveness.h"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "liveness_test",
    srcs = ["liveness_test.cc"],
    deps = [
        ":liveness",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
        "//programl/test:analysis_test_macros",
    ],
)

cc_library(
    name = "reachability",
    srcs = ["reachability.cc"],
    hdrs = ["reachability.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
    ],
)

cc_test(
    name = "reachability_test",
    srcs = ["reachability_test.cc"],
    deps = [
        ":reachability",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
        "//programl/test:analysis_test_macros",
        "//programl/test:llvm_program_graphs",
    ],
)

cc_library(
    name = "subexpressions",
    srcs = ["subexpressions.cc"],
    hdrs = ["subexpressions.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//programl/graph:features",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "subexpressions_test",
    srcs = ["subexpressions_test.cc"],
    deps = [
        ":subexpressions",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
        "//programl/test:analysis_test_macros",
        "//programl/test:llvm_program_graphs",
    ],
)
