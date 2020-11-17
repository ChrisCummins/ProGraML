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

py_test(
    name = "graph_serializer_test",
    srcs = ["graph_serializer_test.py"],
    deps = [
        "//programl/graph/format/py:graph_serializer",
        "//programl/proto:edge_py",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
        "//tests:test_main",
    ],
)

py_test(
    name = "graph_tuple_builder_test",
    srcs = ["graph_tuple_builder_test.py"],
    deps = [
        "//programl/graph/format/py:graph_tuple_builder",
        "//programl/proto:program_graph_py",
        "//tests:test_main",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "nx_format_test",
    srcs = ["nx_format_test.py"],
    deps = [
        "//programl/graph/format/py:nx_format",
        "//programl/proto:program_graph_py",
        "//tests:test_main",
        "//third_party/py/networkx",
    ],
)
