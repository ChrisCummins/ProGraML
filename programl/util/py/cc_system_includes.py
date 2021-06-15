# Get C/C++ compiler includes. Code from CompilerGym, available at:
#
#     https://github.com/facebookresearch/CompilerGym
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List


def _communicate(process, input=None, timeout=None):
    """subprocess.communicate() which kills subprocess on timeout."""
    try:
        return process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        # kill() was added in Python 3.7.
        if sys.version_info >= (3, 7, 0):
            process.kill()
        else:
            process.terminate()
        raise


def _get_system_includes() -> Iterable[Path]:
    """Run the system compiler in verbose mode on a dummy input to get the
    system header search path.
    """
    system_compiler = os.environ.get("CXX", "c++")
    # Create a temporary directory to write the compiled 'binary' to, since
    # GNU assembler does not support piping to stdout.
    with tempfile.TemporaryDirectory() as d:
        process = subprocess.Popen(
            [system_compiler, "-xc++", "-v", "-c", "-", "-o", str(Path(d) / "a.out")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        _, stderr = _communicate(process, input="", timeout=30)
    if process.returncode:
        raise OSError(
            f"Failed to invoke {system_compiler}. "
            f"Is there a working system compiler?\n"
            f"Error: {stderr.strip()}"
        )

    # Parse the compiler output that matches the conventional output format
    # used by clang and GCC:
    #
    #     #include <...> search starts here:
    #     /path/1
    #     /path/2
    #     End of search list
    in_search_list = False
    for line in stderr.split("\n"):
        if in_search_list and line.startswith("End of search list"):
            break
        elif in_search_list:
            # We have an include path to return.
            path = Path(line.strip())
            yield path
            # Compatibility fix for compiling benchmark sources which use the
            # '#include <endian.h>' header, which on macOS is located in a
            # 'machine/endian.h' directory.
            if (path / "machine").is_dir():
                yield path / "machine"
        elif line.startswith("#include <...> search starts here:"):
            in_search_list = True
    else:
        raise OSError(
            f"Failed to parse '#include <...>' search paths from {system_compiler}:\n"
            f"{stderr.strip()}"
        )


# Memoized search paths. Call get_system_includes() to access them.
_SYSTEM_INCLUDES = None


def get_system_includes() -> List[Path]:
    """Determine the system include paths for C/C++ compilation jobs.

    This uses the system compiler to determine the search paths for C/C++ system
    headers. By default, :code:`c++` is invoked. This can be overridden by
    setting :code:`os.environ["CXX"]`.

    :return: A list of paths to system header directories.

    :raises OSError: If the compiler fails, or if the search paths cannot be
        determined.
    """
    # Memoize the system includes paths.
    global _SYSTEM_INCLUDES
    if _SYSTEM_INCLUDES is None:
        _SYSTEM_INCLUDES = list(_get_system_includes())
    return _SYSTEM_INCLUDES
