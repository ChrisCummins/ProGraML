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
import pytest

import programl as pg
from programl.exceptions import GraphCreationError
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_ir"]


def test_from_invalid_c():
    with pytest.raises(
        pg.GraphCreationError,
        match="error: expected function body after function declarator",
    ):
        pg.from_cpp("int main() syntax error!")


def test_c_input():
    graph = pg.from_cpp("int A() { return 0; }")
    assert isinstance(graph, pg.ProgramGraph)


def test_c_input_with_std_header():
    graph = pg.from_cpp(
        """
#include <stdio.h>
void A() {
    printf("Hello, world\\n");
}
"""
    )
    assert isinstance(graph, pg.ProgramGraph)


def test_cxx_input_with_std_header():
    graph = pg.from_cpp(
        """
#include <iostream>
void A() {
    std::cout << "Hello, world" << std::endl;
}
"""
    )
    assert isinstance(graph, pg.ProgramGraph)


def test_c_input_with_undefined_identifier():
    """Implicit use is allowed in C."""
    graph = pg.from_cpp(
        """
void A() {
    B();
}
""",
        language="c",
    )
    assert isinstance(graph, pg.ProgramGraph)


def test_cxx_input_with_undefined_identifier():
    """Implicit use is an error in C++."""
    with pytest.raises(GraphCreationError, match="error: use of undeclared identifier"):
        pg.from_cpp(
            """
void A() {
    B();
}
"""
        )


if __name__ == "__main__":
    main()
