# This package contains the commandline binaries for ProGraML.
#
# #####################################################
# House keeping notes:
#
#   1) When you add a new command line tool, run
#      ./bin/update_docs.sh and commit any changes.
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

filegroup(
    name = "bin",
    srcs = [
        ":clang2graph-10",
        ":graph2dot",
        ":graph2json",
        ":llvm2graph-10",
        ":llvm2graph-3.8",
        ":llvm2graph-6",
        ":xla2graph",
        # Not used by current Python API: ":analyze",
        # Not used by current Python API: ":graph2cdfg",
        # Not used by current Python API: ":graph2seq",
        # Not used by current Python API: ":pbq",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            ":libLLVMPolly-10.0.0",
            ":libLLVMPolly-6.0.0",
        ],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "libLLVMPolly-6.0.0",
    srcs = [
        "@clang-llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04//:all_files",
        "@clang-llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04//:clang",
    ],
    outs = ["libLLVMPolly.so.6.0.0"],
    cmd = "cp $$(dirname $(location @clang-llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04//:clang))/../lib/LLVMPolly.so $@",
    visibility = ["//visibility:public"],
)

genrule(
    name = "libLLVMPolly-10.0.0",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["libLLVMPolly.so.10.0.0"],
    cmd = "cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/LLVMPolly.so $@",
    visibility = ["//visibility:public"],
)

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
    name = "clang2graph-bin",
    srcs = [":clang2graph-10-prelinked"],
    outs = ["clang2graph-10"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :clang2graph-10-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :clang2graph-10-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "patchelf --replace-needed libLLVMPolly.so libLLVMPolly.so.10.0.0 $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "clang2graph-10-prelinked",
    srcs = ["clang2graph.cc"],
    copts = [
        "-std=c++14",
        "-fno-rtti",
    ],
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

# TODO(cummins): Move this to a different package. Compiled binaries only in
# //programl/bin.
py_binary(
    name = "inst2vec",
    srcs = ["inst2vec.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto",
        "//programl/util/py:init_app",
        "//programl/util/py:stdin_fmt",
        "//programl/util/py:stdout_fmt",
    ],
)

cc_binary(
    name = "llvm2graph-3.8",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++14"],
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

genrule(
    name = "llvm2graph-6-bin",
    srcs = [":llvm2graph-6-prelinked"],
    outs = ["llvm2graph-6"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :llvm2graph-6-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :llvm2graph-6-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "patchelf --replace-needed libLLVMPolly.so libLLVMPolly.so.6.0.0 $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "llvm2graph-6-prelinked",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++14"],
    deps = [
        "//programl:version_header",
        "//programl/ir/llvm:llvm-6",
        "//programl/proto:programl_cc",
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:strutil",
        "@llvm//6.0.0",
    ],
)

genrule(
    name = "llvm2graph-10-bin",
    srcs = [":llvm2graph-10-prelinked"],
    outs = ["llvm2graph-10"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :llvm2graph-10-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :llvm2graph-10-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "patchelf --replace-needed libLLVMPolly.so libLLVMPolly.so.10.0.0 $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "llvm2graph-10-prelinked",
    srcs = ["llvm2graph.cc"],
    copts = ["-std=c++14"],
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
        "//programl/util:stdout_fmt",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:statusor",
        "@labm8//labm8/cpp:string",
    ],
)
