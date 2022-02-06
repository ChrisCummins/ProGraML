# Data files for //labm8/py tests.
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

load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

package(
    default_testonly = 1,
    default_visibility = ["//labm8/py:__subpackages__"],
)

py3_image(
    name = "basic_app",
    srcs = ["basic_app.py"],
    base = "@phd_base//image",
    deps = ["//labm8/py:app"],
)

proto_library(
    name = "test_protos_pb",
    srcs = ["test_protos.proto"],
)

python_proto_library(
    name = "test_protos_pb_py",
    deps = [":test_protos_pb"],
)

filegroup(
    name = "data1_file",
    srcs = ["data1"],
)

filegroup(
    name = "testdir_files",
    srcs = glob(["testdir/**"]),
)

filegroup(
    name = "makeproj_files",
    srcs = glob(["makeproj/**"]),
)

filegroup(
    name = "empty_file_file",
    srcs = ["empty_file"],
)

filegroup(
    name = "hello_world_file",
    srcs = ["hello_world"],
)

filegroup(
    name = "jsoncache_file",
    srcs = ["jsoncache.json"],
)
