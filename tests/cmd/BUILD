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
    name = "llvm2graph_strict_mode",
    srcs = ["llvm2graph_strict_mode.py"],
    data = [
        "//programl/bin:llvm2graph-10",
        "//tests/data:module_with_unreachable_instructions",
    ],
    deps = [
        "//programl/util/py:runfiles_path",
        "//tests:test_main",
    ],
)
