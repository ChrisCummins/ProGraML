# Databases of graphs.
#
# Copyright 2019 the ProGraML authors.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

cc_library(
    name = "graph_builder",
    srcs = ["graph_builder.cc"],
    hdrs = ["graph_builder.h"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        ":programl_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "graph_builder_test",
    srcs = ["graph_builder_test.cc"],
    deps = [
        ":graph_builder",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "graphviz_converter",
    srcs = ["graphviz_converter.cc"],
    hdrs = ["graphviz_converter.h"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        ":programl_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "@boost//:graph",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "graphviz_converter_test",
    srcs = ["graphviz_converter_test.cc"],
    deps = [
        ":graphviz_converter",
        ":programl_pb_cc",
        "//labm8/cpp:test",
    ],
)

py_library(
    name = "nx_utils",
    srcs = ["nx_utils.py"],
    visibility = ["//deeplearning/ml4pl/graphs:__subpackages__"],
    deps = [
        ":programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "nx_utils_test",
    srcs = ["nx_utils_test.py"],
    deps = [
        ":nx_utils",
        ":programl",
        ":programl_pb_py",
        "//labm8/py:test",
    ],
)

proto_library(
    name = "programl_pb",
    srcs = ["programl.proto"],
    visibility = [
        "//deeplearning/ml4pl/seq:__subpackages__",
    ],
)

python_proto_library(
    name = "programl_pb_py",
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [":programl_pb"],
)

cc_proto_library(
    name = "programl_pb_cc",
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
    ],
    deps = [":programl_pb"],
)

py_binary(
    name = "programl",
    srcs = ["programl.py"],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/seq:__subpackages__",
        "//deeplearning/ml4pl/testing:__subpackages__",
    ],
    deps = [
        ":programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "programl_test",
    size = "enormous",
    srcs = ["programl_test.py"],
    shard_count = 16,
    deps = [
        ":programl",
        ":programl_pb_py",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
    ],
)
