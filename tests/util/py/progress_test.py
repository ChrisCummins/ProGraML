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
import time

import pytest

from programl.util.py import progress
from tests.test_main import main


class MockJob(progress.Progress):
    """A mock job."""

    def __init__(self, *args, sleep_time: float = 0.1, **kwargs):
        self._sleep_time = sleep_time
        super(MockJob, self).__init__(*args, **kwargs)

    def Run(self):
        n = self.ctx.n or 0
        for self.ctx.i in range(self.ctx.i, n + 1):
            self.ctx.Log(1, "I did a thing")
            time.sleep(self._sleep_time)


@pytest.mark.parametrize("name", ("example",))
# Test with invalid combinations of i and n, where i > n. The progress bar
# should be robust to these cases.
@pytest.mark.parametrize(
    "i",
    (
        0,
        1,
        25,
    ),
)
@pytest.mark.parametrize(
    "n",
    (
        None,
        1,
        20,
        50,
    ),
)
@pytest.mark.parametrize("refresh_time", (0.2, 0.5))
@pytest.mark.parametrize("unit", ("", "spam"))
@pytest.mark.parametrize("vertical_position", (0,))
def test_Run_MockJob_smoke_test(
    name: str,
    i: int,
    n: int,
    refresh_time: float,
    unit: str,
    vertical_position: int,
):
    job = MockJob(name, i, n, unit=unit, vertical_position=vertical_position)
    progress.Run(job, refresh_time=refresh_time)
    assert job.ctx.i == max(i, n or 0)


def test_Run_patience():
    """Test that patience is fine when not exceeded."""
    job = MockJob("name", 0, 10, sleep_time=1)
    progress.Run(job, patience=2)


def test_Run_patience_exceeded():
    """Test that error is raised when patentience is set and exceeded."""
    job = MockJob("name", 0, 10, sleep_time=5)
    with pytest.raises(OSError):
        progress.Run(job, patience=2)


if __name__ == "__main__":
    main()
