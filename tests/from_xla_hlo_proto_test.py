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
from absl import flags

import programl as pg
from programl.third_party.tensorflow import xla_pb2
from programl.util.py import pbutil
from programl.util.py.runfiles_path import runfiles_path
from tests.test_main import main

FLAGS = flags.FLAGS


def test_empty_proto():
    """Build from an empty proto."""
    proto = xla_pb2.HloProto()
    with pytest.raises(ValueError) as e_ctx:
        pg.from_xla_hlo_proto(proto)

    assert "Failed to locate entry computation" in str(e_ctx.value)


def test_non_empty_proto():
    """Build a graph proto from an example proto."""
    proto = pbutil.FromFile(runfiles_path("tests/data/a.hlo.pb"), xla_pb2.HloProto())
    graph = pg.from_xla_hlo_proto(proto)
    assert len(graph.node) == 155
    assert len(graph.function) == 5


if __name__ == "__main__":
    main()
