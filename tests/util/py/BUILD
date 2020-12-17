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
    name = "decorators_test",
    srcs = ["decorators_test.py"],
    deps = [
        "//programl/util/py:decorators",
        "//tests:test_main",
    ],
)

py_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.py"],
    deps = [
        "//programl/util/py:pbutil",
        "//tests:test_main",
        "//tests/data:test_protos_py",
        "//tests/plugins:tempdir",
    ],
)

py_test(
    name = "progress_test",
    srcs = ["progress_test.py"],
    shard_count = 4,
    deps = [
        "//programl/util/py:progress",
        "//tests:test_main",
    ],
)
