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
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "data_flow_pass",
    srcs = ["data_flow_pass.cc"],
    hdrs = ["data_flow_pass.h"],
    deps = [
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
    ],
)

cc_library(
    name = "datadep",
    srcs = ["datadep.cc"],
    hdrs = ["datadep.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "dominance",
    srcs = ["dominance.cc"],
    hdrs = ["dominance.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
    ],
)

cc_library(
    name = "liveness",
    srcs = ["liveness.cc"],
    hdrs = ["liveness.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "reachability",
    srcs = ["reachability.cc"],
    hdrs = ["reachability.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "subexpressions",
    srcs = ["subexpressions.cc"],
    hdrs = ["subexpressions.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":data_flow_pass",
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)
