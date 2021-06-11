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

py_binary(
    name = "create",
    srcs = ["create.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl",
        "//tasks/dataflow/dataset:pathflag",
    ],
)

py_test(
    name = "create_test",
    timeout = "long",
    srcs = ["create_test.py"],
    deps = [
        "//tasks/devmap/dataset:create",
        "//tests:test_main",
        "//tests/plugins:tempdir",
    ],
)
