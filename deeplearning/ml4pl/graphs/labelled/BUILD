# Package for defining labelled graph representations.
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

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "graph_batcher",
    srcs = ["graph_batcher.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_database_reader",
        ":graph_tuple",
        "//labm8/py:app",
        "//labm8/py:progress",
    ],
)

py_test(
    name = "graph_batcher_test",
    size = "enormous",
    srcs = ["graph_batcher_test.py"],
    shard_count = 8,
    deps = [
        ":graph_batcher",
        ":graph_tuple",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "graph_database_reader",
    srcs = ["graph_database_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_tuple_database",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_database_reader_test",
    size = "enormous",
    srcs = ["graph_database_reader_test.py"],
    shard_count = 8,
    deps = [
        ":graph_database_reader",
        ":graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)

py_test(
    name = "graph_database_reader_benchmark_test",
    size = "enormous",
    srcs = ["graph_database_reader_benchmark_test.py"],
    shard_count = 8,
    deps = [
        ":graph_database_reader",
        ":graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:test",
    ],
)

py_library(
    name = "graph_tuple",
    srcs = ["graph_tuple.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "graph_tuple_test",
    size = "enormous",
    srcs = ["graph_tuple_test.py"],
    shard_count = 8,
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//labm8/py:fs",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "graph_tuple_database",
    srcs = ["graph_tuple_database.py"],
    visibility = [
        "//deeplearning/ml4pl:__subpackages__",
    ],
    deps = [
        ":graph_tuple",
        "//deeplearning/ml4pl:run_id",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:decorators",
        "//labm8/py:humanize",
        "//labm8/py:jsonutil",
        "//labm8/py:progress",
        "//labm8/py:sqlutil",
        "//third_party/py/networkx",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "graph_tuple_database_test",
    size = "enormous",
    srcs = ["graph_tuple_database_test.py"],
    shard_count = 8,
    deps = [
        ":graph_tuple_database",
        "//deeplearning/ml4pl/testing:random_graph_tuple_database_generator",
        "//deeplearning/ml4pl/testing:random_graph_tuple_generator",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//deeplearning/ml4pl/testing:testing_databases",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)
