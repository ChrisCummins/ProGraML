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
import contextlib
import functools
import signal
from typing import Any, Callable

Function = Callable[..., Any]


def memoized_property(func: Function) -> property:
    """A property decorator that memoizes the result.
    This is used to memoize the results of class properties, to be used when
    computing the property value is expensive.
    Args:
      func: The function which should be made to a property.
    Returns:
      The decorated property function.
    """
    # Based on Danijar Hafner's implementation of a lazy property, available at:
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute_name = "_memoized_property_" + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute_name):
            setattr(self, attribute_name, func(self))
        return getattr(self, attribute_name)

    return decorator


@contextlib.contextmanager
def timeout(seconds: int):
    """A function decorator that raises TimeoutError after specified time limit.
    Args:
      seconds: The number of seconds before timing out.
    Raises:
      TimeoutError: If the number of seconds have been reached.
    """

    def _RaiseTimoutError(signum, frame):
        raise TimeoutError(f"Function failed to complete within {seconds} seconds")

    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, _RaiseTimoutError)
    signal.alarm(seconds)

    try:
        yield
    except TimeoutError as e:
        raise e
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
