# Python bindings for LLVM-IR libraries.
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

load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

py_library(
    name = "llvm",
    srcs = ["llvm.py"],
    data = [":graph_builder_bin-10-files"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:programl_py",
        "//programl/util/py:runfiles_path",
    ],
)

# This target includes the service binary and its runtime dependencies and
# libraries.
filegroup(
    name = "graph_builder_bin-10-files",
    srcs = [
        ":graph_builder_bin-10",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [":libLLVMPolly"],
    }),
    visibility = ["//visibility:public"],
)

# Post-build shenanigans to strip the absolute path to LLVMPolly.so for linux
# builds.
genrule(
    name = "graph_builder_bin-10-bin",
    srcs = [":graph_builder_bin-10-prelinked"],
    outs = ["graph_builder_bin-10"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :graph_builder_bin-10-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :graph_builder_bin-10-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
)

# This target copies the LLVMPolly.so file from LLVM to libLLVMPolly.so.
genrule(
    name = "libLLVMPolly",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["libLLVMPolly.so"],
    cmd = "cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/LLVMPolly.so $@",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "graph_builder_bin-10-prelinked",
    srcs = ["graph_builder_bin.cc"],
    copts = ["-std=c++17"],
    deps = [
        "//programl/ir/llvm:llvm-10",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:status",
        "@llvm//10.0.0",
    ],
)
