# This package contains libraries for converting program graph formats.
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
    name = "cdfg",
    srcs = ["cdfg.cc"],
    hdrs = ["cdfg.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:statusor",
        "//programl/graph:features",
        "//programl/proto:edge_cc",
        "//programl/proto:node_cc",
        "//programl/proto:program_graph_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "cdfg_test",
    srcs = ["cdfg_test.cc"],
    deps = [
        ":cdfg",
        "//labm8/cpp:logging",
        "//labm8/cpp:test",
        "//programl/graph:program_graph_builder",
    ],
)

cc_library(
    name = "graph_serializer",
    srcs = ["graph_serializer.cc"],
    hdrs = ["graph_serializer.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//programl/proto:program_graph_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "graph_serializer_test",
    srcs = ["graph_serializer_test.cc"],
    deps = [
        ":graph_serializer",
        "//labm8/cpp:test",
        "//programl/proto:program_graph_cc",
    ],
)

cc_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.cc"],
    hdrs = ["graph_tuple.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:status",
        "//programl/proto:program_graph_cc",
    ],
)

cc_test(
    name = "graph_tuple_test",
    srcs = ["graph_tuple_test.cc"],
    deps = [
        ":graph_tuple",
        "//labm8/cpp:status",
        "//labm8/cpp:test",
        "//programl/proto:edge_cc",
        "//programl/proto:program_graph_cc",
    ],
)

cc_library(
    name = "graphviz_converter",
    srcs = ["graphviz_converter.cc"],
    hdrs = ["graphviz_converter.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//programl/proto:program_graph_cc",
        "@boost//:graph",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "node_link_graph",
    srcs = ["node_link_graph.cc"],
    hdrs = ["node_link_graph.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//programl/proto:program_graph_cc",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "node_link_graph_test",
    srcs = ["node_link_graph_test.cc"],
    deps = [
        ":node_link_graph",
        "//labm8/cpp:status",
        "//labm8/cpp:string",
        "//labm8/cpp:test",
        "//programl/proto:program_graph_cc",
        "@nlohmann_json//:json",
    ],
)
