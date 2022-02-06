# Classes for working within Bazel's hemetic and sandboxed world.
#
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

java_library(
    name = "BazelRunfiles",
    srcs = ["BazelRunfiles.java"],
    visibility = ["//visibility:public"],
    deps = [
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_test(
    name = "BazelRunfilesTest",
    srcs = ["BazelRunfilesTest.java"],
    data = ["//labm8/java/test_data:hello_world"],
    test_class = "labm8.java.bazelutil.BazelRunfilesTest",
    deps = [
        ":BazelRunfiles",
        "@org_junit//jar",
    ],
)
