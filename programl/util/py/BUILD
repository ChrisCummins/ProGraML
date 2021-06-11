# Python utility libraries.
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

py_library(
    name = "util",
    srcs = [
        "cc_system_includes.py",
        "decorators.py",
        "executor.py",
        "humanize.py",
        "init_app.py",
        "pbutil.py",
        "progress.py",
        "runfiles_path.py",
        "stdin_fmt.py",
        "stdout_fmt.py",
        "threaded_iterator.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "humanize",
    srcs = ["humanize.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "init_app",
    srcs = ["init_app.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl",
    ],
)

py_library(
    name = "pbutil",
    srcs = ["pbutil.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_library(
    name = "progress",
    srcs = ["progress.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":humanize",
    ],
)

py_library(
    name = "runfiles_path",
    srcs = ["runfiles_path.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "stdin_fmt",
    srcs = ["stdin_fmt.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/util/py:pbutil",
    ],
)

py_library(
    name = "stdout_fmt",
    srcs = ["stdout_fmt.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_library(
    name = "threaded_iterator",
    srcs = ["threaded_iterator.py"],
    visibility = ["//visibility:public"],
)
