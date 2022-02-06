# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
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
load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "print_workspace_status",
    testonly = 1,
    srcs = ["print_workspace_status.py"],
    data = ["//labm8/py/internal:workspace_status"],
)

py_test(
    name = "workspace_status_par_test",
    srcs = ["workspace_status_par_test.py"],
    data = [":print_workspace_status.par"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

py_test(
    name = "workspace_status_test",
    srcs = ["workspace_status_test.py"],
    data = ["//labm8/py/internal:workspace_status"],
    deps = [
        "//labm8/py:test",
    ],
)
