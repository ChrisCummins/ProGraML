# This package contains the commandline binaries for ProGraML.
#
# #####################################################
# House keeping notes:
#
#   1) When you add a new command line tool, consider adding
#      it to the //:install target so that it is installed
#      into the user path.
#
#   2) Run ./bin/update_docs.sh and commit any
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
        "//programl:version_header",
        "//programl/graph/analysis",
        "//programl/proto:programl_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:status",
    ],
)

genrule(
    name = "clang2graph-latest",
    srcs = [":clang2graph-10"],
    outs = ["clang2graph"],
    cmd = "ln -s $$(basename $<) $@",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "clang2graph-10",
    srcs = ["clang2graph.cc"],
    copts = [
        "-std=c++17",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/ir/llvm:llvm-10",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@llvm//10.0.0",
    ],
)

cc_binary(
    name = "graph2cdfg",
    srcs = ["graph2cdfg.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/graph/format:cdfg",
        "//programl/proto:programl_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
    ],
)

cc_binary(
    name = "graph2dot",
    srcs = ["graph2dot.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/graph/format:graphviz_converter",
        "//programl/proto:programl_cc",
        "//programl/util:stdin_fmt",
        "@labm8//labm8/cpp:app",
    ],
)

cc_binary(
    name = "graph2json",
    srcs = ["graph2json.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/graph/format:node_link_graph",
        "//programl/proto:programl_cc",
        "//programl/util:stdin_fmt",
        "@labm8//labm8/cpp:app",
    ],
)

py_binary(
    name = "graph2nx",
    srcs = ["graph2nx.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/graph/format/py:nx_format",
        "//programl/proto:programl_py",
        "//programl/util/py:init_app",
        "//programl/util/py:stdin_fmt",
    ],
)

cc_binary(
    name = "graph2seq",
    srcs = ["graph2seq.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/graph/format:graph_serializer",
        "//programl/proto:programl_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
    ],
)

py_binary(
    name = "inst2vec",
    srcs = ["inst2vec.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:programl_py",
        "//programl/util/py:init_app",
        "//programl/util/py:stdin_fmt",
        "//programl/util/py:stdout_fmt",
    ],
)

genrule(
    name = "llvm2graph-latest",
    srcs = [":llvm2graph-10"],
    outs = ["llvm2graph"],
    cmd = "ln -s $$(basename $<) $@",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "llvm2graph-3.8",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/ir/llvm:llvm-3.8",
        "//programl/proto:programl_cc",
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:strutil",
        "@llvm//3.8.0",
    ],
)

cc_binary(
    name = "llvm2graph-10",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/ir/llvm:llvm-10",
        "//programl/proto:programl_cc",
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:strutil",
        "@llvm//10.0.0",
    ],
)

cc_binary(
    name = "pbq",
    srcs = ["pbq.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/proto:programl_cc",
        "//programl/third_party/tensorflow:xla_cc",
        "//programl/util:stdin_fmt",
        "//programl/util:stdout_fmt",
        "@com_google_protobuf//:protobuf",
        "@labm8//labm8/cpp:app",
    ],
)

cc_binary(
    name = "xla2graph",
    srcs = ["xla2graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:version_header",
        "//programl/ir/xla:hlo_module_graph_builder",
        "//programl/third_party/tensorflow:xla_cc",
        "//programl/util:stdin_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
    ],
)
