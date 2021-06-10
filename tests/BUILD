# This package contains shared code and data for tests.
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
    name = "analysis_test_macros",
    testonly = 1,
    hdrs = ["analysis_test_macros.h"],
    visibility = ["//tests:__subpackages__"],
)

cc_library(
    name = "llvm_program_graphs",
    testonly = 1,
    srcs = ["llvm_program_graphs.cc"],
    hdrs = ["llvm_program_graphs.h"],
    data = ["//tests/data:llvm_ir_graphs"],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//programl/proto:programl_cc",
        "@boost//:filesystem",
        "@labm8//labm8/cpp:bazelutil",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:strutil",
    ],
)

py_test(
    name = "from_clang_test",
    srcs = ["from_clang_test.py"],
    shard_count = 4,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "from_cpp_test",
    srcs = ["from_cpp_test.py"],
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "from_llvm_ir_test",
    srcs = ["from_llvm_ir_test.py"],
    shard_count = 4,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "from_xla_hlo_proto_test",
    srcs = ["from_xla_hlo_proto_test.py"],
    data = ["//tests/data:hlo_protos"],
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "serialize_graphs_test",
    srcs = ["serialize_graphs_test.py"],
    shard_count = 3,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_test(
    name = "to_dot_test",
    srcs = ["to_dot_test.py"],
    shard_count = 8,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "to_json_test",
    srcs = ["to_json_test.py"],
    shard_count = 8,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)

py_test(
    name = "to_networkx_test",
    srcs = ["to_networkx_test.py"],
    shard_count = 8,
    deps = [
        "//programl",
        "//tests:test_main",
        "//tests/plugins",
    ],
)
