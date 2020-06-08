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
    visibility = ["//programl:__subpackages__"],
)

cc_library(
    name = "llvm_program_graphs",
    testonly = 1,
    srcs = ["llvm_program_graphs.cc"],
    hdrs = ["llvm_program_graphs.h"],
    data = ["//programl/test/data:llvm_ir_graphs"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "//labm8/cpp:bazelutil",
        "//labm8/cpp:logging",
        "//labm8/cpp:strutil",
        "//programl/proto:program_graph_cc",
        "@boost//:filesystem",
    ],
)
