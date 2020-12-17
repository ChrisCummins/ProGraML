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
    name = "clang_test",
    srcs = ["clang_test.cc"],
    deps = [
        "//programl/ir/llvm:clang",
        "//programl/proto:programl_cc",
        "@labm8//labm8/cpp:test",
    ],
)

py_test(
    name = "inst2vec_encoder_test",
    timeout = "long",
    srcs = ["inst2vec_encoder_test.py"],
    deps = [
        "//programl/graph/py:program_graph_builder",
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto:programl_py",
        "//tests:test_main",
        "//tests/plugins:llvm_program_graph",
    ],
)
