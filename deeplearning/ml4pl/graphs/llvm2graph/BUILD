# This package constructs program graphs from LLVM intermediate representations.
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
    name = "build_graph",
    srcs = ["build_graph.cc"],
    hdrs = ["build_graph.h"],
    visibility = ["//deeplearning/ml4pl/graphs/llvm2graph:__subpackages__"],
    deps = [
        ":graph_builder_pass",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//third_party/llvm",
    ],
)

cc_library(
    name = "graph_builder_pass",
    srcs = ["graph_builder_pass.cc"],
    hdrs = ["graph_builder_pass.h"],
    copts = [
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
    ],
    deps = [
        ":llvm_graph_builder",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//third_party/llvm",
    ],
)

cc_binary(
    name = "llvm2graph",
    srcs = ["llvm2graph.cc"],
    deps = [
        ":build_graph",
        "//deeplearning/ml4pl/graphs:graphviz_converter",
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//third_party/llvm",
        "@boost//:graph",
    ],
)

py_library(
    name = "llvm2graph_py",
    srcs = ["llvm2graph.py"],
    data = [":llvm2graph"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//third_party/py/networkx",
    ],
)

sh_binary(
    name = "llvm2graph_ld_preload",
    srcs = ["llvm2graph_ld_preload.sh"],
    data = [
        ":llvm2graph",
    ],
    deps = [
        "//labm8/sh:app",
    ],
)

py_test(
    name = "llvm2graph_test",
    srcs = ["llvm2graph_test.py"],
    shard_count = 4,
    deps = [
        ":llvm2graph_py",
        "//deeplearning/ml4pl/graphs:nx_utils",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing/fixtures:llvm_ir",
        "//labm8/py:test",
    ],
)

py_test(
    name = "llvm2graph_regression_test",
    timeout = "long",
    srcs = ["llvm2graph_regression_test.py"],
    data = ["//deeplearning/ml4pl/testing/data/bytecode_regression_tests:bytecodes"],
    deps = [
        ":llvm2graph_py",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

cc_library(
    name = "llvm_graph_builder",
    srcs = ["llvm_graph_builder.cc"],
    hdrs = ["llvm_graph_builder.h"],
    deps = [
        "//deeplearning/ml4pl/graphs:graph_builder",
        "//deeplearning/ml4pl/graphs:programl_pb_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//third_party/llvm",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

py_binary(
    name = "node_encoder",
    srcs = ["node_encoder.py"],
    data = [
        "//deeplearning/ml4pl/graphs/llvm2graph/node_embeddings:dictionary",
        "//deeplearning/ml4pl/graphs/llvm2graph/node_embeddings:embeddings",
    ],
    visibility = [
        "//deeplearning/ml4pl/graphs:__subpackages__",
        "//deeplearning/ml4pl/models:__subpackages__",
    ],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ncc/inst2vec:inst2vec_preprocess",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:decorators",
        "//labm8/py:fs",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "node_encoder_test",
    srcs = ["node_encoder_test.py"],
    deps = [
        ":node_encoder",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing/fixtures:llvm_program_graph",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
