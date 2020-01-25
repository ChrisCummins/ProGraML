"""A module for working within Bazel's hemetic and sandboxed world.

The design goals of Bazel has led it to have a lot of quirks. This module
contains utility code designed to help live with these quirks.
"""
import os
import pathlib
import pkgutil
import re
import subprocess
import sys
import typing

from labm8.py import app
from labm8.py import archive
from labm8.py import fs

# Regular expression to everything in a path up until the '*.runfiles'
# directory, e.g. for the path '/private/var/bazel/foo/bar.runfiles/a/b/c',
# this regex will match '/private/var/bazel/foo/bar.runfiles'
RUNFILES_PATTERN = re.compile(r"^(.*\.runfiles)/")


def FindRunfilesDirectory() -> typing.Optional[pathlib.Path]:
  """Find the '.runfiles' directory, if there is one.

  Returns:
    The absolute path of the runfiles directory, else None if not found.
  """
  # Follow symlinks, looking for my module space
  stub_filename = os.path.abspath(__file__)
  module_space = stub_filename + ".runfiles"
  if os.path.isdir(module_space):
    return pathlib.Path(module_space)
  match = RUNFILES_PATTERN.match(os.path.abspath(__file__))
  if match:
    return pathlib.Path(match.group(1))
  return None


def DataPath(
  path: typing.Union[str, pathlib.Path], must_exist: bool = True,
) -> pathlib.Path:
  """Return the absolute path to a data file.

  This allows you to access files from the 'data' attribute of a Python
  target in Bazel. This is needed because the path to the file changes depending
  on whether the current process is executing with a 'bazel run' environment, or
  as a 'bazel-bin' script.

  !!!WARNING!!! For par_binary() rules, this function does not work. See
  DataString() instead.

  Args:
    path: The path to the data, including the name of the workspace.
    must_exist: Require that the file exists, else raise FileNotFoundError.

  Returns:
    An absolute file path.

  Raises:
    FileNotFoundError: If the requested path is not found and must_exist is
      True.
  """
  if not str(path):
    if must_exist:
      raise FileNotFoundError(f"No such file or directory: '{path}'")
    else:
      # An empty path yields the runfiles directory.
      return FindRunfilesDirectory()
  runfiles = FindRunfilesDirectory()
  real_path = runfiles / path if runfiles else pathlib.Path(path).absolute()
  if must_exist and not (real_path.is_file() or real_path.is_dir()):
    raise FileNotFoundError(f"No such file or directory: '{path}'")
  return real_path


def DataString(path: typing.Union[str, pathlib.Path]) -> bytes:
  """Return the contents of a data file.

  This allows you to access files from the 'data' attribute of a Python
  target in Bazel. This is needed because the path to the file changes depending
  on whether the current process is executing with a 'bazel run' environment, or
  as a 'bazel-bin' script. "String" is a misnomer, this returns a byte array.

  The reason to use this function over DataPath() is that this supports
  accessing the 'data' attributes of par_binary rules. When a par binary is
  created, the contents of data files are embedded directly into the
  executable, so DataPath will raise FileNotFoundError.

  Args:
    path: The path to the data, including the name of the workspace.

  Returns:
    The contents of the file as an array of bytes.

  Raises:
    FileNotFoundError: If the requested path is not found.
  """
  try:
    # Support for reading data files from par binaries.
    # See: github.com/google/subpar/issues/43
    return pkgutil.get_data("__main__", path)
  except ValueError:
    # pkgutil.get_data() raises ValueError if __spec__ is not found.
    with open(DataPath(path), "rb") as f:
      return f.read()


class DataArchive(archive.Archive):
  """A compressed archive file.

  TODO: data=[...] attribute of target.

  Provides access to an unzipped data file when used as a context manager by
  extracting the zip contents to a temporary directory.

  Example:
    >>> with DataArchive("phd/data.zip") as uncompressed_root:
    ...   print(uncompressed_root.iterdir())
    ['a', 'README.txt']
  """

  def __init__(self, path: typing.Union[str, pathlib.Path]):
    """Constructor.

    Args:
      path: The path to the data, including the name of the workspace.

    Raises:
      FileNotFoundError: If path is not a file.
    """
    super(DataArchive, self).__init__(DataPath(path))


class Workspace(object):
  """Class representing a bazel workspace."""

  def __init__(self, root: pathlib.Path):
    """Create a bazel workspace object.

    The workspace must already exist, this object does not modify files.

    Args:
       root: The root of the workspace.

    Raises:
      OSError: If the root is not a workspace.
    """
    self._root = root
    if not (self._root / "WORKSPACE").is_file():
      raise OSError(f"`{self._root}/WORKSPACE` not found")

  @property
  def workspace_root(self) -> pathlib.Path:
    return self._root

  def BazelQuery(
    self,
    args: typing.List[str],
    timeout_seconds: int = 360,
    **subprocess_kwargs,
  ):
    """Run bazel query with the specified args in the workspace.

    Args:
      args: The list of arguments to pass to bazel query.
      timeout_seconds: The number of seconds before failing.
      subprocess_kwargs: Additional arguments to pass to Popen().
    """
    return self.Bazel(
      "query", args, timeout_seconds=timeout_seconds, **subprocess_kwargs
    )

  def Bazel(
    self,
    command: str,
    args: typing.List[str],
    timeout_seconds: int = 360,
    **subprocess_kwargs,
  ):
    cmd = [
      "timeout",
      "-s9",
      str(timeout_seconds),
      "bazel",
      command,
      "--noshow_progress",
    ] + args
    app.Log(2, "$ %s", " ".join(cmd))
    with fs.chdir(self.workspace_root):
      return subprocess.Popen(cmd, **subprocess_kwargs)

  def MaybeTargetToPath(
    self, fully_qualified_target: str,
  ) -> typing.Optional[pathlib.Path]:
    """Determine if a bazel target refers to a file, and if so return the path.

    Args:
      fully_qualified_target: The bazel target, beginning with '//'.

    Returns:
      A path, relative to the root of the workspace, else NOne.

    Raises:
      ValueError: If the given target is not fully qualified.
    """

    def RelpathIfExists(path: str) -> typing.Optional[pathlib.Path]:
      """Return if given relative path is a file."""
      abspath = self.workspace_root / path
      return path if abspath.is_file() else None

    if fully_qualified_target.startswith("//:"):
      return RelpathIfExists(fully_qualified_target[3:])
    elif fully_qualified_target.startswith("//"):
      return RelpathIfExists(fully_qualified_target[2:].replace(":", "/"))
    else:
      raise ValueError(
        "Target is not fully qualified (does not begin with `//`): "
        f"{fully_qualified_target}",
      )

  def GetDependentFiles(
    self, target: str, excluded_targets: typing.Iterable[str],
  ) -> typing.List[pathlib.Path]:
    """Get the file dependencies of the target.

    Args:
      target: The target to get the dependencies of.

    Returns:
      A list of paths, relative to the root of the workspace.

    Raises:
      OSError: If bazel query fails.
    """
    # First run through bazel query to expand globs.
    bazel = self.BazelQuery([target], stdout=subprocess.PIPE)
    grep = subprocess.Popen(
      ["grep", "^/"],
      stdout=subprocess.PIPE,
      stdin=bazel.stdout,
      universal_newlines=True,
    )

    stdout, _ = grep.communicate()
    if bazel.returncode:
      raise OSError("bazel query failed")
    if grep.returncode:
      raise OSError("grep of bazel query output failed")
    targets = stdout.rstrip().split("\n")

    # Now get the transitive dependencies of each target.
    targets = sorted(
      [target for target in targets if target not in excluded_targets]
    )
    all_targets = set(targets)
    for i, target in enumerate(targets, start=1):
      print(
        f"\r\033[KCollecting transitive dependencies ({len(all_targets)}): {target}",
        end="",
      )
      sys.stdout.flush()
      bazel = self.BazelQuery(
        [f"deps({target})"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
      )
      grep = subprocess.Popen(
        ["grep", "^/"],
        stdout=subprocess.PIPE,
        stdin=bazel.stdout,
        universal_newlines=True,
      )

      stdout, _ = grep.communicate()
      if bazel.returncode:
        raise OSError("bazel query failed")
      if grep.returncode:
        raise OSError("grep of bazel query output failed")

      deps = stdout.rstrip().split("\n")
      all_targets = all_targets.union(set(deps))

    print(f"\r\033[KCollected {len(all_targets)} transitive deps")
    paths = [
      self.MaybeTargetToPath(target)
      for target in all_targets
      if not target in excluded_targets
    ]
    return [path for path in paths if path]

  def GetBuildFiles(self, target: str) -> typing.List[pathlib.Path]:
    """Get the BUILD files required for the given target.

    Raises:
      OSError: If bazel query fails.
    """
    bazel = self.BazelQuery(
      [f"buildfiles(deps({target}))"], stdout=subprocess.PIPE,
    )
    cut = subprocess.Popen(
      ["cut", "-f1", "-d:"], stdout=subprocess.PIPE, stdin=bazel.stdout,
    )
    grep = subprocess.Popen(
      ["grep", "^/"],
      stdout=subprocess.PIPE,
      stdin=cut.stdout,
      universal_newlines=True,
    )

    stdout, _ = grep.communicate()
    if bazel.returncode:
      raise OSError("bazel query failed")
    if cut.returncode:
      raise OSError("bazel query output cut failed")
    if grep.returncode:
      raise OSError("bazel query output search failed")

    for line in stdout.rstrip().split("\n"):
      if line == "//external":
        # Files in //external are virtual.
        continue
      path = os.path.join(line[2:], "BUILD")
      abspath = self.workspace_root / path
      if not abspath.is_file():
        raise OSError(f"BUILD file not found: {path}")
      yield path
