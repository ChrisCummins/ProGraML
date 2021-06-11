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
from concurrent.futures.thread import ThreadPoolExecutor

import pytest

import programl as pg
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_program_graph"]


@pytest.fixture(scope="session")
def graph() -> pg.ProgramGraph:
    return pg.from_cpp("int A() { return 0; }")


def test_to_json_simple_graph(graph: pg.ProgramGraph):
    jsons = list(pg.to_json([graph]))
    assert len(jsons) == 1
    assert isinstance(jsons[0], dict)


def test_to_json_two_inputs(graph: pg.ProgramGraph):
    jsons = list(pg.to_json([graph, graph]))
    assert len(jsons) == 2
    assert jsons[0] == jsons[1]


def test_to_json_generator(graph: pg.ProgramGraph):
    jsons = list(pg.to_json((graph for _ in range(10)), chunksize=3))
    assert len(jsons) == 10
    for x in jsons[1:]:
        assert x
        assert x == jsons[0]


def test_to_json_generator_parallel_executor(graph: pg.ProgramGraph):
    with ThreadPoolExecutor() as executor:
        jsons = list(
            pg.to_json((graph for _ in range(10)), chunksize=3, executor=executor)
        )
    assert len(jsons) == 10
    for x in jsons[1:]:
        assert x
        assert x == jsons[0]


def test_to_json_smoke_test(llvm_program_graph: pg.ProgramGraph):
    jsons = list(pg.to_json([llvm_program_graph]))
    assert len(jsons) == 1
    assert isinstance(jsons[0], dict)


if __name__ == "__main__":
    main()
