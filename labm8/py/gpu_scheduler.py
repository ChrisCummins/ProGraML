# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""This modules defines an inter-process locking scheduler for exclusive access
to CUDA GPUs.

A frequent pattern for workloads is to require exclusive access to a GPU for the
duration of a process. This module implements a simple inter-process locking
scheduler to enable that. Scripts can call LockExclusiveProcessGpuAccess() to
get access to a single GPU which is set through CUDA_VISIBLE_DEVICES.

    from labm8.py import gpu_scheduler

    gpu_schduler.LockExclusiveProcessGpuAccess()

    # go nuts ...

This does of course assume that all GPU users go through this interface. There
is nothing stopping another user or process from coming along and violating the
lock granted to a script which has "exclusive" GPU access.

The default arguments for LockExclusiveProcessGpuAccess() work transparently
on systems with no GPUs. If no GPUs are available, the return value is None.
Alternatively, use requires_gpus=True argument to raise an OSError.

By default, GPUs are disabled during testing (as determined by the presence
of $TEST_TMPDIR which bazel sets). To enable the GPUs for tests, use
--test_with_gpu, but note that when executing large numbers of tests, they may
have to queue and execute sequentially, causing timeouts.
"""
import contextlib
import functools
import os
import pathlib
import time
from typing import Any
from typing import Dict
from typing import Optional

import fasteners
import GPUtil

from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS

app.DEFINE_boolean(
  "test_with_gpu", False, "If set, enable access to GPUs during testing."
)

_LOCK_DIR = pathlib.Path("/tmp/phd/labm8/gpu_scheduler_locks")

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from
# nvidia-smi.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class NoGpuAvailable(OSError):
  """Error raised if no GPU is available."""


class GpuScheduler(object):
  """A simple interprocess locking scheduler for GPUs."""

  def __init__(self, gpu_locks: Dict[GPUtil.GPU, Any]):
    self.gpu_locks = gpu_locks

  def TryToAcquireGpu(self, gpu: GPUtil.GPU) -> bool:
    if gpu not in self.gpu_locks:
      raise ValueError(f"GPU not found: {gpu}")

    return self.gpu_locks[gpu].acquire(blocking=False)

  def ReleaseGpu(self, gpu: GPUtil.GPU) -> None:
    if gpu not in self.gpu_locks:
      raise ValueError(f"GPU not found: {gpu}")
    self.gpu_locks[gpu].release()

  def BlockOnAvailableGpu(
    self, timeout: Optional[int] = None, print_status: bool = True
  ):
    start_time = time.time()
    end_time = start_time + (timeout or 0)

    while True:
      for gpu in self.gpu_locks:
        if self.TryToAcquireGpu(gpu):
          if print_status:
            print("\r")
          wait = ""
          if time.time() - start_time > 1:
            wait = f" after {humanize.Duration(time.time() - start_time)} wait"
          app.Log(1, "Acquired GPU %s (%s)%s", gpu.id, gpu.name, wait)

          os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.id)

          return gpu

      if timeout and time.time() > end_time:
        raise NoGpuAvailable(
          f"No GPU available after waiting for {humanize.Duration}"
        )

      if print_status:
        print(
          f"\rwaiting on a free gpu ... {humanize.Duration(time.time() - start_time)}",
          end="",
        )
      time.sleep(0.5)


@functools.lru_cache(1)
def GetDefaultScheduler() -> GpuScheduler:
  gpus = GPUtil.getGPUs()
  if not gpus:
    raise NoGpuAvailable("No GPUs available")

  if os.environ.get("TEST_TMPDIR") and not FLAGS.test_with_gpu:
    raise NoGpuAvailable("GPUs disabled for tests")

  app.Log(
    2, "Creating default scheduler for %s", humanize.Plural(len(gpus), "GPU")
  )
  return GpuScheduler(
    {gpu: fasteners.InterProcessLock(_LOCK_DIR / str(gpu.id)) for gpu in gpus}
  )


# This function is memoized since we can always acquire the same lock twice.
@functools.lru_cache(1)
def LockExclusiveProcessGpuAccess(
  scheduler: Optional[GpuScheduler] = None,
  timeout: Optional[int] = None,
  print_status: bool = True,
  require_gpus: bool = False,
) -> Optional[GPUtil.GPU]:
  """Lock exclusive access to the given GPU."""
  try:
    scheduler = scheduler or GetDefaultScheduler()
  except NoGpuAvailable as e:
    if require_gpus:
      raise e
    else:
      return None

  gpu = scheduler.BlockOnAvailableGpu(
    timeout=timeout, print_status=print_status
  )

  return gpu


@contextlib.contextmanager
def ExclusiveGpuAccess(
  scheduler: Optional[GpuScheduler] = None,
  timeout: Optional[int] = None,
  print_status: bool = True,
  require_gpus: bool = False,
) -> Optional[GPUtil.GPU]:
  """Get exclusive access to a GPU with a scoped session.

  Args:
    scheduler:
    timeout:
    print_status:
    require_gpus:

  Returns:

  """
  try:
    scheduler = scheduler or GetDefaultScheduler()
  except NoGpuAvailable as e:
    if require_gpus:
      raise e
    else:
      return None

  gpu = scheduler.BlockOnAvailableGpu(
    timeout=timeout, print_status=print_status
  )

  try:
    yield gpu
  finally:
    scheduler.ReleaseGpu(gpu)
    app.Log(1, "Released GPU %s (%s)", gpu.id, gpu.name)
