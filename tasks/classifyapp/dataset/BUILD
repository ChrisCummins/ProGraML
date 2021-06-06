# Creating the POJ-104 dataset.
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
    name = "create",
    srcs = ["create.cc"],
    deps = [
        ":poj104",
        "@boost//:filesystem",
        "@labm8//labm8/cpp:app",
    ],
)

cc_library(
    name = "poj104",
    srcs = ["poj104.cc"],
    hdrs = ["poj104.h"],
    deps = [
        "//programl/ir/llvm:clang",
        "//programl/ir/llvm:llvm-10",
        "//programl/proto:programl_cc",
        "//programl/util:filesystem_cache",
        "//third_party/tbb",
        "@boost//:filesystem",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
        "@ctpl",
        "@labm8//labm8/cpp:crypto",
        "@labm8//labm8/cpp:fsutil",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
        "@labm8//labm8/cpp:status_macros",
        "@labm8//labm8/cpp:statusor",
        "@subprocess",
    ],
)
