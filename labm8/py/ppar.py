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
"""This module contains utility functions for parallelism in python.

The goal of the module is to provide easy to use implementations of typical
parallel workloads, such as data parallel map operations.
"""
import multiprocessing
import queue
import subprocess
import threading
import typing

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import pbutil
from labm8.py import sqlutil

FLAGS = app.FLAGS


class MapWorkerError(EnvironmentError):
  """Resulting error from a _MapWorker that fails."""

  def __init__(self, returncode: int):
    """Create a _MapWorker error.

    Args:
      returncode: The process return code.
    """
    self._returncode = returncode

  def __repr__(self) -> str:
    return f"Command exited with code {self.returncode}"

  @property
  def returncode(self) -> int:
    """Get the return code of the process."""
    return self._returncode


class _MapWorker(object):
  """A work unit for a data parallel workload.

  A _MapWorker executes a command as a subprocess, passes it a protocol buffer,
  and decodes a protocol buffer output.

  This is a helper class created by MapNativeProtoProcessingBinary() and
  returned to the user. It is not to be used by user code.
  """

  def __init__(
    self, id: int, cmd: typing.List[str], input_proto: pbutil.ProtocolBuffer,
  ):
    """Create a map worker.

    Args:
      id: The numeric ID of the map worker.
      cmd: The command to execute, as a list of arguments to subprocess.Popen().
      input_proto: The protocol buffer to pass to the command.
    """
    self._id = id
    self._cmd = cmd
    # We store the input proto in wire format (as a serialized string) rather
    # than as a class object as pickle can get confused by the types.
    # See: https://stackoverflow.com/a/1413299
    self._input_proto: typing.Optional[pbutil.ProtocolBuffer] = None
    self._input_proto_string = input_proto.SerializeToString()
    self._output_proto_string: typing.Optional[str] = None
    self._output_proto: typing.Optional[pbutil.ProtocolBuffer] = None
    self._output_proto_decoded = False
    self._returncode: typing.Optional[int] = None
    self._done = False

  def Run(self) -> None:
    """Execute the process and store the output.

    If the process fails, no exception is raised. The error can be accessed
    using the error() method. After calling this method, SetProtos() *must* be
    called.
    """
    assert not self._done

    # Run the C++ worker process, capturing it's output.
    process = subprocess.Popen(
      self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    # Send the input proto to the C++ worker process.
    # TODO: Add timeout.
    stdout, _ = process.communicate(self._input_proto_string)
    self._returncode = process.returncode
    del self._input_proto_string

    if not process.returncode:
      # Store the C++ binary output in wire format.
      self._output_proto_string = stdout

  def SetProtos(
    self, input_proto: pbutil.ProtocolBuffer, output_proto_class: typing.Type,
  ) -> None:
    """Set the input protocol buffer, and decode the output protocol buffer.

    This is performed by the SetProtos() method (rather than during Run()) so
    that when pickled, this class contains only basic types, not protocol buffer
    instances.

    Args:
      input_proto: The input protocol buffer message.
      output_proto_class: The protocol buffer class of the output message.
    """
    assert not self._done
    self._done = True
    self._input_proto = input_proto
    # Only parse the output if the worker completed successfully.
    if not self._returncode:
      # Be careful that errors during protocol buffer decoding (e.g.
      # unrecognized fields, conflicting field type/tags) are silently ignored
      # here.
      self._output_proto = output_proto_class.FromString(
        self._output_proto_string,
      )
    # No need to hand onto the string message any more.
    del self._output_proto_string

  @property
  def id(self):
    """Return the numeric ID of the map worker."""
    return self._id

  def input(self) -> pbutil.ProtocolBuffer:
    """Get the input protocol buffer."""
    assert self._done
    return self._input_proto

  def output(self) -> typing.Optional[pbutil.ProtocolBuffer]:
    """Get the protocol buffer decoded from stdout of the executed binary.

    If the process failed (e.g. not _MapWorker.ok()), None is returned.
    """
    assert self._done
    return self._output_proto

  def error(self) -> typing.Optional[MapWorkerError]:
    """Get the error generated by a failed binary execution.

    If the process succeeded (e.g. _MapWorker.ok()), None is returned.
    """
    if self._returncode:
      return MapWorkerError(self._returncode)

  def ok(self) -> bool:
    """Return whether binary execution succeeded."""
    return not self._returncode


def _RunNativeProtoProcessingWorker(map_worker: _MapWorker) -> _MapWorker:
  """Private helper message to execute Run() method of _MapWorker.

  This is passed to Pool.imap_unordered() as the function to execute for every
  work unit. This is needed because only module-level functions can be pickled.
  """
  map_worker.Run()
  return map_worker


def MapNativeProtoProcessingBinary(
  binary_data_path: str,
  input_protos: typing.List[pbutil.ProtocolBuffer],
  output_proto_class: typing.Type,
  binary_args: typing.Optional[typing.List[str]] = None,
  pool: typing.Optional[multiprocessing.Pool] = None,
  num_processes: typing.Optional[int] = None,
) -> typing.Iterator[_MapWorker]:
  """Run a protocol buffer processing binary over a set of inputs.

  Args:
    binary_data_path: The path of the binary to execute, as provied to
      bazelutil.DataPath().
    input_protos: An iterable list of input protos.
    output_proto_class: The proto class of the output.
    binary_args: An optional list of additional arguments to pass to binaries.
    pool: The multiprocessing pool to use.
    num_processes: The number of processes for the multiprocessing pool.

  Returns:
    A generator of _MapWorker instances. The order is random.
  """
  binary_path = bazelutil.DataPath(binary_data_path)
  binary_args = binary_args or []
  cmd = [str(binary_path)] + binary_args

  # Read all inputs to a list. We need the inputs in a list so that we can
  # map an inputs position in the list to a _MapWorker.id.
  input_protos = list(input_protos)

  # Create the multiprocessing pool to use, if not provided.
  pool = pool or multiprocessing.Pool(processes=num_processes)

  map_worker_iterator = (
    _MapWorker(i, cmd, input_proto)
    for i, input_proto in enumerate(input_protos)
  )

  for map_worker in pool.imap_unordered(
    _RunNativeProtoProcessingWorker, map_worker_iterator,
  ):
    map_worker.SetProtos(input_protos[map_worker.id], output_proto_class)
    yield map_worker


def MapNativeProcessingBinaries(
  binaries: typing.List[str],
  input_protos: typing.List[pbutil.ProtocolBuffer],
  output_proto_classes: typing.List[typing.Type],
  pool: typing.Optional[multiprocessing.Pool] = None,
  num_processes: typing.Optional[int] = None,
) -> typing.Iterator[_MapWorker]:
  """Run a protocol buffer processing binary over a set of inputs.

  Args:
    binary_data_path: The path of the binary to execute, as provied to
      bazelutil.DataPath().
    input_protos: An iterable list of input protos.
    output_proto_class: The proto class of the output.
    binary_args: An optional list of additional arguments to pass to binaries.
    pool: The multiprocessing pool to use.
    num_processes: The number of processes for the multiprocessing pool.

  Returns:
    A generator of _MapWorker instances. The order is random.
  """
  if not len(binaries) == len(input_protos):
    raise ValueError("Number of binaries does not equal protos")

  cmds = [[bazelutil.DataPath(b)] for b in binaries]

  # Read all inputs to a list. We need the inputs in a list so that we can
  # map an inputs position in the list to a _MapWorker.id.
  input_protos = list(input_protos)
  output_proto_classes = list(output_proto_classes)

  # Create the multiprocessing pool to use, if not provided.
  pool = pool or multiprocessing.Pool(processes=num_processes)

  map_worker_iterator = (
    _MapWorker(id, cmd, input_proto,)
    for id, (cmd, input_proto,) in enumerate(zip(cmds, input_protos))
  )

  for map_worker in pool.imap_unordered(
    _RunNativeProtoProcessingWorker, map_worker_iterator,
  ):
    map_worker.SetProtos(
      input_protos[map_worker.id], output_proto_classes[map_worker.id],
    )
    yield map_worker


# Type annotations for MapDatabaseRowBatchProcessor().
WorkUnitType = typing.Callable[[typing.List[typing.Any]], typing.Any]
WorkUnitArgGenerator = typing.Callable[[typing.Any], typing.Any]
ResultCallback = typing.Callable[[typing.Any], None]
BatchCallback = typing.Callable[[int], None]


def MapDatabaseRowBatchProcessor(
  work_unit: WorkUnitType,
  query: sqlutil.Query,
  generate_work_unit_args: WorkUnitArgGenerator = lambda rows: rows,
  work_unit_result_callback: ResultCallback = lambda result: None,
  start_of_batch_callback: BatchCallback = lambda i: None,
  end_of_batch_callback: BatchCallback = lambda i: None,
  batch_size: int = 256,
  rows_per_work_unit: int = 5,
  start_at: int = 0,
  pool: typing.Optional[multiprocessing.Pool] = None,
) -> None:
  """Execute a database row-processesing function in parallel.

  Use this function to orchestrate the parallel execution of a function that
  takes batches of rows from the result set of a query.

  This is equivalent to a serial implementation:

    for row in query:
      work_unit_result_callback(work_unit(generate_work_unit_args)))
    end_of_batch_callback()

  Args:
    work_unit: A function which takes an input a list of the values returned
      by generate_work_unit_args callback, and produces a list of zero or more
      instances of output_table_class.
    query: The query which produces inputs to the work units.
    generate_work_unit_args: A callback which transforms a single result of the
      query into an input to a work unit.
    batch_size:
    rows_per_work_unit:
    start_at:
    pool:

  Returns:
    Foo.
  """

  pool = pool or multiprocessing.Pool()

  i = start_at
  row_batches = sqlutil.OffsetLimitBatchedQuery(query, batch_size=batch_size)

  for batch in row_batches:
    rows_batch = batch.rows
    start_of_batch_callback(i)

    work_unit_args = [
      generate_work_unit_args(rows_batch[i : i + rows_per_work_unit])
      for i in range(0, len(rows_batch), rows_per_work_unit)
    ]

    for result in pool.starmap(work_unit, work_unit_args):
      work_unit_result_callback(result)

    i += len(rows_batch)

    end_of_batch_callback(i)


class _ForcedNonDaemonProcess(multiprocessing.Process):
  """A process which is never a daemon."""

  # make 'daemon' attribute always return False
  def _get_daemon(self):
    return False

  def _set_daemon(self, value):
    pass

  daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class UnsafeNonDaemonPool(multiprocessing.pool.Pool):
  """A multiprocessing.Pool where the processes are not daemons.

  Python's multiprocessing Pool creates daemonic processes. Deamonic processes
  are killed automatically when the parent process terminates. This is nice
  behaviour as it prevents orphaned processes lying around. However, a downside
  of daemonic processes is that they cannot create child processes. E.g. a
  worker in an parallel map cannot create child processes. This is occasionally
  desirable. For cases where you need non-daemonic processes, use this class.

  Disclaimer: USING THIS CLASS CAN RESULT IN ORPHAN PROCESSES IF YOU DO NOT
  EXPLICITLY CLOSE THE POOL!

  Example usage:

    pool = ppar.UnsafeNonDaemonPool(5)

    try:
      # go nuts ...
    finally:
      pool.close()
      pool.join()
  """

  Process = _ForcedNonDaemonProcess


class ThreadedIterator:
  """An iterator that computes its elements in a parallel thread to be ready to
  be consumed.

  Exceptions raised by the threaded iterator are propagated to consumer.
  """

  def __init__(
    self,
    iterator: typing.Iterable[typing.Any],
    max_queue_size: int = 2,
    start: bool = True,
  ):
    self._queue = queue.Queue(maxsize=max_queue_size)
    self._thread = threading.Thread(target=lambda: self.worker(iterator))
    if start:
      self.Start()

  def Start(self):
    self._thread.start()

  def worker(self, iterator):
    try:
      for element in iterator:
        self._queue.put(self._ValueOrError(value=element), block=True)
    except Exception as e:
      # Propagate an error in the iterator.
      self._queue.put(self._ValueOrError(error=e))
    # Mark that the iterator is done.
    self._queue.put(self._EndOfIterator(), block=True)

  def __iter__(self):
    next_element = self._queue.get(block=True)
    while not isinstance(next_element, self._EndOfIterator):
      value = next_element.GetOrRaise()
      yield value
      next_element = self._queue.get(block=True)
    self._thread.join()

  class _EndOfIterator(object):
    """Tombstone marker object for iterators."""

    pass

  class _ValueOrError(typing.NamedTuple):
    """A tuple which represents the union of either a value or an error."""

    value: typing.Any = None
    error: Exception = None

    def GetOrRaise(self) -> typing.Any:
      """Return the value or raise the exception."""
      if self.error is None:
        return self.value
      else:
        raise self.error
