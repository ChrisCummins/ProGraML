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
import subprocess
import sys

from labm8.py import bazelutil, test

TRAIN_GGNN = bazelutil.DataPath("programl/programl/task/dataflow/train_ggnn")


REACHABILITY_DATAFLOW_DATASET = bazelutil.DataArchive(
    "programl/programl/test/data/reachability_dataflow_dataset.tar.bz2"
)


def test_reachability_end_to_end():
    with REACHABILITY_DATAFLOW_DATASET as d:
        p = subprocess.Popen(
            [
                TRAIN_GGNN,
                f"--path={d}",
                "--analysis",
                "reachability",
                "--limit_max_data_flow_steps",
                "--layer_timesteps=10",
                "--val_graph_count=10",
                "--val_seed=204",
                "--train_graph_counts=10,20",
                "--batch_size=8",
            ]
        )
        p.communicate()
        if p.returncode:
            sys.exit(1)


if __name__ == "__main__":
    test.Main()
