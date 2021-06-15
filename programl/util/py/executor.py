# Executor implementation from Nevergrad, available at:
#
#     https://github.com/facebookresearch/nevergrad
#
# Nevergrad license header:
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from itertools import islice
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

if sys.version_info > (3, 8, 0):
    from typing import Protocol

    class ExecutorLike(Protocol):
        # pylint: disable=pointless-statement, unused-argument

        def map(self, func: Callable[..., T], *iterables: Any) -> Iterable[T]:
            ...


else:

    ExecutorLike = TypeVar("ExecutorLike")


class SequentialExecutor:
    """Executor which run sequentially and locally."""

    def map(self, func: Callable[..., T], *iterables: Any) -> Iterable[T]:
        return map(func, *iterables)


def execute(
    run_one: Callable[[T], U],
    inputs: Iterable[T],
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Iterable[U]:
    """Split inputs chunksize lists and process in parallel using executor.

    Similar to Executor.map, but inputs is evaluated lazily rather than ahead of
    time, to support very large input interators.
    """
    executor = executor or SequentialExecutor()
    chunksize = chunksize or 512
    inputs = iter(inputs)
    chunk = list(islice(inputs, chunksize))
    while chunk:
        yield from executor.map(run_one, chunk)
        chunk = list(islice(inputs, chunksize))
