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

py_library(
    name = "programl",
    srcs = [
        "__init__.py",
        "create_ops.py",
        "exceptions.py",
        "serialize_ops.py",
        "transform_ops.py",
        "version.py",
    ],
    data = [
        "//programl/bin",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/ir/llvm:inst2vec_encoder",
        "//programl/proto",
        "//programl/third_party/tensorflow:xla_py",
        "//programl/util/py:util",
    ],
)

genrule(
    name = "version_header-gen",
    srcs = ["//:version.txt"],
    outs = ["version.h"],
    cmd = "echo \"#define PROGRAML_VERSION \\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

genrule(
    name = "version_py-gen",
    srcs = ["//:version.txt"],
    outs = ["version.py"],
    cmd = "echo \"PROGRAML_VERSION=\\\"$$(tr -d '\n' < $<)\\\"\" > $@",
)

cc_library(
    name = "version_header",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)
