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
"""Unit tests for //labm8/py:ppar."""
import multiprocessing

import progressbar
import pytest
from labm8.py import ppar, test
from labm8.py.test_data.ppar import protos_pb2


def test_MapWorker_okay():
    inputs = [protos_pb2.AddXandY(x=2, y=2), protos_pb2.AddXandY(x=0, y=1)]
    ret = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            input_protos=inputs,
            output_proto_class=protos_pb2.AddXandY,
        ),
    )

    # Two inputs produce two outputs.
    assert len(ret) == 2

    # Check that both work units completed successfully.
    assert ret[0].ok()
    assert ret[1].ok()

    # Get the output protos.
    output_protos = [r.output() for r in ret]

    # The proto worker binary does not set the x and y fields.
    assert not output_protos[0].x
    assert not output_protos[0].y
    assert not output_protos[1].x
    assert not output_protos[1].y

    # Sort the results by input values.
    ret = sorted(ret, key=lambda r: r.input().x)

    # Check that the input protos are unchanged.
    assert not ret[0].input().x
    assert ret[0].input().y == 1
    assert not ret[0].input().result

    assert ret[1].input().x == 2
    assert ret[1].input().y == 2
    assert not ret[1].input().result

    # Check that the output protos have the expected values.
    assert not ret[0].output().x
    assert not ret[0].output().y
    assert ret[0].output().result == 0 + 1

    assert not ret[1].output().x
    assert not ret[1].output().y
    assert ret[1].output().result == 2 + 2


def test_MapWorker_one_failure():
    inputs = [protos_pb2.AddXandY(x=2, y=2), protos_pb2.AddXandY(x=10, y=1)]
    workers = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            input_protos=inputs,
            output_proto_class=protos_pb2.AddXandY,
        ),
    )

    # Two inputs produce two outputs.
    assert len(workers) == 2

    # Sort the results by input values. The first worker (2 + 2) will be good,
    # the second worker (10 + 0) will have failed (see proto_worker.cc for the
    # CHECK() macro that causes this failure).
    workers = sorted(workers, key=lambda r: r.input().x)

    # Check that the first result has the correct values.
    assert workers[0].ok()
    assert workers[0].output().result == 4
    assert workers[0].error() is None

    # Check that the failed worker has the correct values.
    assert not workers[1].ok()
    assert workers[1].output() is None
    error = workers[1].error()
    assert type(error) is ppar.MapWorkerError
    assert error.returncode


def test_MapWorker_wrap_progressbar():
    """Test wrapping the worker generator with a progress bar."""
    inputs = [protos_pb2.AddXandY(x=2, y=2), protos_pb2.AddXandY(x=10)]
    bar = progressbar.ProgressBar(max_value=len(inputs))

    worker_generator = ppar.MapNativeProtoProcessingBinary(
        "phd/labm8/py/test_data/ppar/proto_worker",
        input_protos=inputs,
        output_proto_class=protos_pb2.AddXandY,
    )

    map_workers = []
    for map_worker in bar(worker_generator):
        map_workers.append(map_worker)

    # Two inputs produce two outputs.
    assert len(map_workers) == 2


def test_MapWorker_output_decode_error_silently_ignore():
    """Test that message that cannot be decoded is silently ignored."""
    inputs = [protos_pb2.AddXandY(x=2, y=2)]

    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/unexpected_output_proto_worker",
            inputs,
            protos_pb2.AddXandY,
        ),
    )

    assert len(results) == 1
    assert results[0].ok()
    assert not results[0].output().x
    assert not results[0].output().y
    assert not results[0].output().result


def test_MapWorker_binary_not_found():
    """Test that FileNotFoundError raised when binary does not exist."""
    # Note that calling the function does not raise the error (since it is a
    # generator and is evaluated lazily. The error is raised when we attempt to
    # read the results.
    generator = ppar.MapNativeProtoProcessingBinary(
        "phd/labm8/py/test_data/ppar/not/a/real/binary",
        [protos_pb2.AddXandY()],
        protos_pb2.AddXandY,
    )
    with test.Raises(FileNotFoundError):
        next(generator)


def test_MapWorker_no_inputs():
    """Test that no output is produced when run with no inputs."""
    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            [],
            protos_pb2.AddXandY,
        ),
    )
    assert not results


def test_MapWorker_generator_inputs():
    """Test that generator input is accepted."""

    def InputGenerator():
        """A generator for MapNativeProtoProcessingBinary() inputs."""
        yield protos_pb2.AddXandY(x=1, y=2)
        yield protos_pb2.AddXandY(x=2, y=2)

    assert len(list(InputGenerator())) == 2

    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            InputGenerator(),
            protos_pb2.AddXandY,
        ),
    )

    assert len(results) == 2


def test_MapWorker_pool():
    """Test that multiprocessing.Pool can be passed to function."""
    pool = multiprocessing.Pool(processes=1)
    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            [protos_pb2.AddXandY(x=2, y=2)],
            protos_pb2.AddXandY,
            pool=pool,
        ),
    )

    assert len(results) == 1
    assert results[0].ok()
    assert results[0].output().result == 4


def test_MapWorker_num_processes():
    """Test running with explicit number of processes."""
    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker",
            [protos_pb2.AddXandY(x=2, y=2)],
            protos_pb2.AddXandY,
            num_processes=1,
        ),
    )

    assert len(results) == 1
    assert results[0].ok()
    assert results[0].output().result == 4


def test_MapWorker_binary_args():
    """Test that args are passed to the binary."""
    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker_requires_args",
            [protos_pb2.AddXandY(x=2, y=2)],
            protos_pb2.AddXandY,
            binary_args=["-required_arg"],
        ),
    )
    assert len(results) == 1
    assert results[0].ok()

    # Run again without the required arg, causing the binary to crash.
    results = list(
        ppar.MapNativeProtoProcessingBinary(
            "phd/labm8/py/test_data/ppar/proto_worker_requires_args",
            [protos_pb2.AddXandY(x=2, y=2)],
            protos_pb2.AddXandY,
        ),
    )
    assert len(results) == 1
    assert not results[0].ok()


if __name__ == "__main__":
    test.Main()
