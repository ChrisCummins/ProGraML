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
"""Logging interface.
"""
import json

from labm8.py import system


def colourise(colour, *args):
    return "".join([colour] + list(args) + [Colours.RESET])


def printf(colour, *args, **kwargs):
    string = colourise(colour, *args)
    print(string, **kwargs)


def pprint(data, **kwargs):
    print(json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")), **kwargs)


def info(*args, **kwargs):
    print("[INFO  ]", *args, **kwargs)


def debug(*args, **kwargs):
    print("[DEBUG ]", *args, **kwargs)


def warn(*args, **kwargs):
    print("[WARN  ]", *args, **kwargs)


def error(*args, **kwargs):
    print("[ERROR ]", *args, **kwargs)


def fatal(*args, **kwargs):
    returncode = kwargs.pop("status", 1)
    error("fatal:", *args, **kwargs)
    system.exit(returncode)


def prof(*args, **kwargs):
    """
    Print a profiling message.

    Profiling messages are intended for printing runtime performance
    data. They are prefixed by the "PROF" title.

    Arguments:

        *args, **kwargs: Message payload.
    """
    print("[PROF  ]", *args, **kwargs)


class Colours:
    """
    Shell escape colour codes.
    """

    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
