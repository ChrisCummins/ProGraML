# Utility libraries.
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
    name = "filesystem_cache",
    srcs = ["filesystem_cache.cc"],
    hdrs = ["filesystem_cache.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:filesystem",
        "@labm8//labm8/cpp:fsutil",
        "@labm8//labm8/cpp:logging",
    ],
)

cc_library(
    name = "nproc",
    srcs = ["nproc.cc"],
    hdrs = ["nproc.h"],
    visibility = ["//programl:__subpackages__"],
    deps = [
        "@subprocess",
    ],
)

cc_library(
    name = "stdin_fmt",
    srcs = ["stdin_fmt.cc"],
    hdrs = ["stdin_fmt.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)

cc_library(
    name = "stdout_fmt",
    srcs = ["stdout_fmt.cc"],
    hdrs = ["stdout_fmt.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@labm8//labm8/cpp:app",
        "@labm8//labm8/cpp:logging",
        "@labm8//labm8/cpp:status",
    ],
)
