# This package contains the commandline binaries for ProGraML.
#
# #####################################################
# House keeping notes:
#
#   1) When you add a new command line tool, consider adding
#      it to the //programl:install so that it is installed
#      into the user path.
#
#   2) Run ./programl/cmd/update_docs.sh and commit any
#      changes.
# #####################################################
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

cc_binary(
    name = "analyze",
    srcs = ["analyze.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:status",
        "//programl/graph/analysis",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
    ],
)

cc_binary(
    name = "clang2graph",
    srcs = ["clang2graph.cc"],
    copts = [
        "-std=c++14",
        "-fno-rtti",
        "-DGOOGLE_PROTOBUF_NO_RTTI",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:status",
        "//programl/ir/llvm",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_options_cc",
        "@llvm//10.0.0",
    ],
)

cc_binary(
    name = "graph2cdfg",
    srcs = ["graph2cdfg.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//programl/graph/format:cdfg",
        "//programl/proto:program_graph_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
    ],
)

cc_binary(
    name = "graph2dot",
    srcs = ["graph2dot.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//programl/graph/format:graphviz_converter",
        "//programl/proto:program_graph_cc",
        "//programl/util:stdin_fmt",
    ],
)

cc_binary(
    name = "graph2json",
    srcs = ["graph2json.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//programl/graph/format:node_link_graph",
        "//programl/proto:program_graph_cc",
        "//programl/util:stdin_fmt",
    ],
)

py_binary(
    name = "graph2nx",
    srcs = ["graph2nx.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//programl/graph/format/py:nx_format",
        "//programl/proto:program_graph_py",
        "//programl/util/py:stdin_fmt",
    ],
)

cc_binary(
    name = "graph2seq",
    srcs = ["graph2seq.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/format:graph_serializer",
        "//programl/proto:node_cc",
        "//programl/proto:program_graph_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
    ],
)

py_binary(
    name = "inst2vec",
    srcs = ["inst2vec.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:pbutil",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:program_graph_py",
        "//programl/util/py:stdin_fmt",
        "//programl/util/py:stdout_fmt",
    ],
)

cc_binary(
    name = "llvm2graph",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++14"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:status",
        "//labm8/cpp:statusor",
        "//labm8/cpp:strutil",
        "//programl/ir/llvm",
        "//programl/proto:ir_cc",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_options_cc",
        "//programl/util:stdout_fmt",
        "@llvm//10.0.0",
    ],
)

cc_binary(
    name = "pbq",
    srcs = ["pbq.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//programl/proto:checkpoint_cc",
        "//programl/proto:ir_cc",
        "//programl/proto:program_graph_cc",
        "//programl/proto:program_graph_features_cc",
        "//programl/proto:src_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_binary(
    name = "xla2graph",
    srcs = ["xla2graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//programl/ir/xla:hlo_module_graph_builder",
        "//programl/util:stdin_fmt",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto",
    ],
)
