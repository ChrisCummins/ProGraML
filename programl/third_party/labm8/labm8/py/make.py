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
"""Wrapper for invoking Makefiles.
"""
import re

from labm8.py import fs, system


class Error(Exception):
    """
    Module-level error class.
    """

    pass


class MakeError(Error):
    """
    Thrown if make command fails.
    """

    pass


class NoMakefileError(Error):
    """
    Thrown if a directory does not contain a Makefile.
    """

    pass


class NoTargetError(Error):
    """
    Thrown if there is no rule for the requested target.
    """

    pass


_BAD_TARGET_RE = re.compile(r"No rule to make target " "'.+'.  Stop.")


def make(target="all", dir=".", **kwargs):
    """
    Run make.

    Arguments:

        target (str, optional): Name of the target to build. Defaults
          to "all".
        dir (str, optional): Path to directory containing Makefile.
        **kwargs (optional): Any additional arguments to be passed to
          system.run().

    Returns:

        (int, str, str): The first element is the return code of the
          make command. The second and third elements are the stdout
          and stderr of the process.

    Raises:

        NoMakefileError: In case a Makefile is not found in the target
          directory.
        NoTargetError: In case the Makefile does not support the
          requested target.
        MakeError: In case the target rule fails.
    """
    if not fs.isfile(fs.path(dir, "Makefile")):
        raise NoMakefileError("No makefile in '{}'".format(fs.abspath(dir)))

    fs.cd(dir)

    # Default parameters to system.run()
    if "timeout" not in kwargs:
        kwargs["timeout"] = 300

    ret, out, err = system.run(["make", target], **kwargs)
    fs.cdpop()

    if ret > 0:
        if re.search(_BAD_TARGET_RE, err):
            raise NoTargetError("No rule for target '{}'".format(target))
        else:
            raise MakeError("Target '{}' failed".format(target))

        raise MakeError("Failed")

    return ret, out, err


def clean(**kwargs):
    """
    Run make clean.

    Equivalent to invoking make() with target="clean".

    Arguments:

        **kwargs (optional): Any additional arguments to be passed to
          make().

    Returns:

        (int, str, str): The first element is the return code of the
          make command. The second and third elements are the stdout
          and stderr of the process.

    Raises:

        NoMakefileError: In case a Makefile is not found in the target
          directory.
        NoTargetError: In case the Makefile does not support the
          requested target.
        MakeError: In case the target rule fails.
    """
    make(target="clean", **kwargs)
