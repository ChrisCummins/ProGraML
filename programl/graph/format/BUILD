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
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph:features",
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
    ],
)

cc_library(
    name = "graph_serializer",
    srcs = ["graph_serializer.cc"],
    hdrs = ["graph_serializer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@labm8//labm8/cpp:logging",
    ],
)

cc_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.cc"],
    hdrs = ["graph_tuple.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "graphviz_converter",
    srcs = ["graphviz_converter.cc"],
    hdrs = ["graphviz_converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@boost//:graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
    ],
)

cc_library(
    name = "node_link_graph",
    srcs = ["node_link_graph.cc"],
    hdrs = ["node_link_graph.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@nlohmann_json//:json",
    ],
)
