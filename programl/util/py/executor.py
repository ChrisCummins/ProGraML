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
from typing import Any, Callable, Optional, TypeVar

X = TypeVar("X", covariant=True)

if sys.version_info > (3, 8, 0):
    from typing import Protocol

    class JobLike(Protocol[X]):
        # pylint: disable=pointless-statement

        def done(self) -> bool:
            ...

        def result(self) -> X:
            ...

    class ExecutorLike(Protocol):
        # pylint: disable=pointless-statement, unused-argument

        def submit(self, fn: Callable[..., X], *args: Any, **kwargs: Any) -> JobLike[X]:
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
    """Executor which run sequentially and locally
    (just calls the function and returns a FinishedJob)
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> DelayedJob:
        return DelayedJob(fn, *args, **kwargs)
