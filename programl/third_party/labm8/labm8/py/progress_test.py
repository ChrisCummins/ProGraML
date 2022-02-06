"""Unit tests for //labm8/py:progress."""
import time

from labm8.py import progress, test

FLAGS = test.FLAGS


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


@test.Parametrize("name", ("example",))
# Test with invalid combinations of i and n, where i > n. The progress bar
# should be robust to these cases.
@test.Parametrize(
    "i",
    (
        0,
        1,
        25,
    ),
)
@test.Parametrize(
    "n",
    (
        None,
        1,
        20,
        50,
    ),
)
@test.Parametrize("refresh_time", (0.2, 0.5))
@test.Parametrize("unit", ("", "spam"))
@test.Parametrize("vertical_position", (0,))
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
    with test.Raises(OSError):
        progress.Run(job, patience=2)


if __name__ == "__main__":
    test.Main()
