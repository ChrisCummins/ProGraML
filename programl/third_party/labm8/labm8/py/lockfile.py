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
"""Lock file mechanism."""
import datetime
import inspect
import os
import pathlib
import sys
import time
import typing

from labm8.py import app, humanize, labdate, pbutil, system
from labm8.py.internal import lockfile_pb2

# Use absolute paths for imports so as to prevent a conflict with the
# system "time" module.

FLAGS = app.FLAGS

app.DEFINE_float(
    "lockfile_block_seconds",
    10.0,
    "The number of seconds to block for when waiting for a lock file.",
)


class Error(Exception):
    pass


class UnableToAcquireLockError(Error):
    """ thrown if cannot acquire lock """

    def __init__(self, lock: "LockFile"):
        self.lock = lock

    def __str__(self):
        return f"""\
Unable to acquire file lock owned by a different process.
Lock acquired by process {self.lock.pid} on \
{self.lock.user}@{self.lock.hostname} at {self.lock.date}.

If you believe that this is an error and that no other
process holds the lock, you may remove the lock file:

   {self.lock.path}"""


class MalformedLockfileError(UnableToAcquireLockError):
    """ thrown if lockfile is malformed """

    def __init__(self, path: pathlib.Path):
        self.path = path

    def __str__(self):
        return f"Lock file is malformed: {self.path}"


class UnableToReleaseLockError(Error):
    """ thrown if cannot release lock """

    def __init__(self, lock: "LockFile"):
        self.lock = lock

    def __str__(self):
        return f"""\
Unable to release file lock owned by a different process.
Lock acquired by process {self.lock.pid} on \
{self.lock.user}@{self.lock.hostname} at {self.lock.date}.

If you believe that this is an error and that no other
process holds the lock, you may remove the lock file:

   {self.lock.path}"""


class LockFile:
    """A lock file.

    Attributes:
      path: Path of lock file.
    """

    def __init__(self, path: typing.Union[str, pathlib.Path]):
        """Create a new directory lock.

        Args:
          path: Path to lock file.
        """
        self.path = pathlib.Path(path).expanduser().absolute()

    @property
    def pid(self) -> typing.Optional[datetime.datetime]:
        """The process ID of the lock. Value is None if lock is not claimed."""
        lockfile = self.read(self.path)
        if lockfile.HasField("owner_process_id"):
            return lockfile.owner_process_id
        else:
            return None

    @property
    def date(self) -> typing.Optional[datetime.datetime]:
        """The date that the lock was acquired. Value is None if lock is unclaimed."""
        lockfile = self.read(self.path)
        if lockfile.date_acquired_unix_epoch_ms:
            return labdate.DatetimeFromMillisecondsTimestamp(
                lockfile.date_acquired_unix_epoch_ms,
            )
        else:
            return None

    @property
    def hostname(self) -> typing.Optional[str]:
        """The hostname of the lock owner. Value is None if lock is unclaimed."""
        lockfile = self.read(self.path)
        if lockfile.HasField("owner_hostname"):
            return lockfile.owner_hostname
        else:
            return None

    @property
    def user(self) -> typing.Optional[str]:
        """The username of the lock owner. Value is None if lock is unclaimed."""
        lockfile = self.read(self.path)
        if lockfile.HasField("owner_user"):
            return lockfile.owner_user
        else:
            return None

    @property
    def islocked(self) -> bool:
        """Whether the directory is locked."""
        return self.path.is_file()

    @property
    def owned_by_self(self):
        """
        Whether the directory is locked by the current process.
        """
        return self.hostname == system.HOSTNAME and self.pid == os.getpid()

    def acquire(
        self,
        replace_stale: bool = False,
        force: bool = False,
        pid: int = None,
        block: bool = False,
    ) -> "LockFile":
        """Acquire the lock.

        A lock can be claimed if any of these conditions are true:
          1. The lock is not held by anyone.
          2. The lock is held but the 'force' argument is set.
          3. The lock is held by the current process.

        Args:
          replace_stale: If true, lock can be aquired from stale processes. A stale
            process is one which currently owns the parent lock, but no process with
            that PID is alive. A lock which is owned by a different hostname is
            never stale, since we cannot determine if the PID of a remote system is
            alive.
          force: If true, ignore any existing lock. If false, fail if lock already
            claimed.
          pid: If provided, force the process ID of the lock to this value.
            Otherwise the ID of the current process is used.
          block: If True, block indefinitely until the lock is available. Use with
            care!

        Returns:
          Self.

        Raises:
          UnableToAcquireLockError: If the lock is already claimed
            (not raised if force option is used).
        """

        def _create_lock():
            lockfile = lockfile_pb2.LockFile(
                owner_process_id=os.getpid() if pid is None else pid,
                owner_process_argv=" ".join(sys.argv),
                date_acquired_unix_epoch_ms=labdate.MillisecondsTimestamp(
                    labdate.GetUtcMillisecondsNow(),
                ),
                owner_hostname=system.HOSTNAME,
                owner_user=system.USERNAME,
            )
            pbutil.ToFile(lockfile, self.path, assume_filename="LOCK.pbtxt")

        while True:
            if self.islocked:
                lock_owner_pid = self.pid
                if self.owned_by_self:
                    pass  # don't replace existing lock
                    break
                elif force:
                    _create_lock()
                    break
                elif (
                    replace_stale
                    and self.hostname == system.HOSTNAME
                    and not system.isprocess(lock_owner_pid)
                ):
                    _create_lock()
                    break
                elif not block:
                    raise UnableToAcquireLockError(self)
                # Block and try again later.
                app.Log(
                    1,
                    "Blocking on lockfile %s for %s seconds",
                    self.path,
                    humanize.Duration(FLAGS.lockfile_block_seconds),
                )
                time.sleep(FLAGS.lockfile_block_seconds)
            else:  # new lock
                _create_lock()
                break
        return self

    def release(self, force=False):
        """Release lock.

        To release a lock, we must already own the lock.

        Args:
          force: If true, ignore any existing lock owner.

        Raises:
          UnableToReleaseLockError: If the lock is claimed by another process (not
            raised if force option is used).
        """
        # There's no lock, so do nothing.
        if not self.islocked:
            return

        if self.owned_by_self or force:
            os.remove(self.path)
        else:
            raise UnableToReleaseLockError(self)

    def __repr__(self):
        return str(self.path)

    def __enter__(self):
        return self.acquire(replace_stale=True, block=True)

    def __exit__(self, type, value, tb):
        self.release()

    @staticmethod
    def read(path: typing.Union[str, pathlib.Path]) -> lockfile_pb2.LockFile:
        """Read the contents of a LockFile.

        Args:
          path: Path to lockfile.

        Returns:
          A LockFile proto.
        """
        path = pathlib.Path(path)
        if path.is_file():
            try:
                return pbutil.FromFile(
                    path,
                    lockfile_pb2.LockFile(),
                    assume_filename="LOCK.pbtxt",
                )
            except pbutil.DecodeError:
                raise MalformedLockfileError(path)
        else:
            return lockfile_pb2.LockFile()


class AutoLockFile(LockFile):
    """A lockfile which derives it's path automatically.

    This LockFile suclass automatically derives its path from its position in
    the calling code. The path is:

      <root>/<calling_module>_<calling_method>_<calling_lineno>.pbtxt

    Where `root` is a root directory (by default, /tmp/phd/labm8/autolockfiles),
    `calling_file` is the name of the module containing the calling code,
    `calling_method` is the name of the method containing the calling code, and
    `calling_lineno` is the line number of the calling code.

    Use this class to conveniently mark sections of code that can only be executed
    by a single process at a time, e.g.

        def MyFunc():
          mutex = AutoLockFile()
          with mutex:
            DoSomeWork()

    Now, if MyFunc is invoked by two separate processes, one of them will fail.
    """

    def __init__(
        self,
        root: typing.Union[str, pathlib.Path] = "/tmp/phd/labm8/autolockfiles",
        granularity: str = "line",
    ):
        """Constructor

        Args:
          root: The directory to place the lock file in.
          granularity: The granularity of the lock, one of {line,function,module}.
            'line' means that the lock is unique to the calling line. 'function'
            means that the lock is unique to the calling function. 'module' means
            that the lock is unique to the calling module.
        """
        root = pathlib.Path(root)

        # Unlike the regular LockFile, an AutoLockFile will create the necessary
        # parent directory.
        root.mkdir(parents=True, exist_ok=True)

        # Inspect the calling code to get the lockfile path components.
        frame = inspect.stack()[1]
        module_name = pathlib.Path(frame.filename).stem
        function_name = frame.function
        lineno = frame.lineno

        if granularity == "line":
            path = root / f"{module_name}_{function_name}_{lineno}.pbtxt"
        elif granularity == "function":
            path = root / f"{module_name}_{function_name}.pbtxt"
        elif granularity == "module":
            path = root / f"{module_name}.pbtxt"
        else:
            raise TypeError(
                f"Invalid granularity '{granularity}'. Must be one of: "
                f"{{line,function,module}}",
            )

        super(AutoLockFile, self).__init__(path)
