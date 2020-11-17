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

cc_test(
    name = "cdfg_test",
    srcs = ["cdfg_test.cc"],
    deps = [
        "//programl/graph:program_graph_builder",
        "//programl/graph/format:cdfg",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:test",
    ],
)

cc_test(
    name = "graph_serializer_test",
    srcs = ["graph_serializer_test.cc"],
    deps = [
        "//programl/graph/format:graph_serializer",
        "//programl/proto:program_graph_cc",
        "@labm8//labm8/cpp:test",
    ],
)

cc_test(
    name = "graph_tuple_test",
    srcs = ["graph_tuple_test.cc"],
    deps = [
        "//programl/graph/format:graph_tuple",
        "//programl/proto:edge_cc",
        "//programl/proto:program_graph_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:test",
    ],
)

cc_test(
    name = "graphviz_converter_test",
    srcs = ["graphviz_converter_test.cc"],
    deps = [
        "//programl/graph/format:graphviz_converter",
        "//programl/proto:program_graph_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:test",
    ],
)

cc_test(
    name = "node_link_graph_test",
    srcs = ["node_link_graph_test.cc"],
    deps = [
        "//programl/graph/format:node_link_graph",
        "//programl/proto:program_graph_cc",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:string",
        "@labm8//labm8/cpp:test",
        "@nlohmann_json//:json",
    ],
)
