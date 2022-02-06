# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //labm8/py:gpu_scheduler."""
from labm8.py import gpu_scheduler, test

FLAGS = test.FLAGS


class MockGpu(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __repr__(self):
        return f"{self.id} {self.name}"


class MockLock(object):
    def __init__(self, locked=False):
        self.locked = locked

    def acquire(self, blocking: bool = True):
        if self.locked:
            return False
        self.locked = True
        return True

    def release(self):
        pass


def test_LockExclusiveProcessGpuAccess_returns_the_same_gpu():
    """Test that LockExclusiveProcessGpuAccess returns the same GPU when called
    multiple times.

    This allows scripts to call LockExclusiveProcessGpuAccess() multiple times,
    safe in the knowledge that only a single GPU is locked.
    """
    a = MockGpu(0, "a")
    b = MockGpu(1, "b")
    gpus = {a: MockLock(), b: MockLock()}

    scheduler = gpu_scheduler.GpuScheduler(gpus)

    gpus = set()
    gpus.add(gpu_scheduler.LockExclusiveProcessGpuAccess(scheduler=scheduler))
    gpus.add(gpu_scheduler.LockExclusiveProcessGpuAccess(scheduler=scheduler))

    assert len(gpus) == 1
    assert list(gpus)[0] in {a, b}


def test_LockExclusiveProcessGpuAccess_locked_devices():
    """Test that LockExclusiveProcessGpuAccess returns the free GPU when multiple
    GPUs are available but all-but-one is locked.
    """
    a = MockGpu(0, "a")
    b = MockGpu(1, "b")
    c = MockGpu(2, "c")
    gpus = {
        a: MockLock(locked=True),
        b: MockLock(),
        c: MockLock(locked=True),
    }

    scheduler = gpu_scheduler.GpuScheduler(gpus)

    gpu = gpu_scheduler.LockExclusiveProcessGpuAccess(scheduler=scheduler)

    assert gpu == b


def test_LockExclusiveProcessGpuAccess_locked_timeout():
    """Test that LockExclusiveProcessGpuAccess raises an error when the GPU is
    locked."""
    a = MockGpu(0, "a")
    gpus = {
        a: MockLock(locked=True),
    }

    scheduler = gpu_scheduler.GpuScheduler(gpus)

    with test.Raises(gpu_scheduler.NoGpuAvailable):
        gpu_scheduler.LockExclusiveProcessGpuAccess(scheduler=scheduler, timeout=1)


if __name__ == "__main__":
    test.Main()
