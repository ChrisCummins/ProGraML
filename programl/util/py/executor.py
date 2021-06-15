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
from collections import deque
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

if sys.version_info > (3, 8, 0):
    from typing import Protocol

    class JobLike(Protocol[T]):
        # pylint: disable=pointless-statement

        def done(self) -> bool:
            ...

        def result(self) -> T:
            ...

    class ExecutorLike(Protocol):
        # pylint: disable=pointless-statement, unused-argument

        def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> JobLike[T]:
            ...


else:

    JobLike = TypeVar("JobLike")
    ExecutorLike = TypeVar("ExecutorLike")


class DelayedJob:
    """Future-like object which delays computation"""

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result: Optional[Any] = None
        self._computed = False

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        if not self._computed:
            self._result = self.func(*self.args, **self.kwargs)
            self._computed = True
        return self._result


class SequentialExecutor:
    """Executor which run sequentially and locally."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> DelayedJob:
        return DelayedJob(fn, *args, **kwargs)


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
    chunksize = chunksize or 1024
    inputs = iter(inputs)
    futures = deque()
    while True:
        try:
            item = next(inputs)
        except StopIteration:
            break
        futures.append(executor.submit(run_one, item))

        while len(futures) > chunksize:
            future = futures.popleft()
            yield future.result()

    for future in futures:
        yield future.result()
