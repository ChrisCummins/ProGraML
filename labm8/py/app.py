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
"""A wrapper around absl's app and logging modules.

See: <https://github.com/abseil/abseil-py>
"""
import functools
import json
import pathlib
import re
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from absl import app as absl_app
from absl import flags as absl_flags
from absl import logging as absl_logging

from labm8.py import shell
from labm8.py.internal import flags_parsers
from labm8.py.internal import logging

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_boolean(
  "version", False, "Print version information and exit.",
)
absl_flags.DEFINE_boolean(
  "dump_flags", False, "Print the defined flags and their values and exit."
)
absl_flags.DEFINE_boolean(
  "dump_flags_to_json",
  False,
  "Print the defined flags and their values to JSON and exit.",
)
absl_flags.DEFINE_boolean(
  "log_colors", True, "Whether to colorize logging output."
)

# A decorator to mark a function as ignored when computing the log prefix.
#
# Example usage:
#
#   @skip_log_prefix
#   def LogFoo():
#     app.Log(1, "Foo")
skip_log_prefix = absl_logging.skip_log_prefix


class UsageError(absl_app.UsageError):
  """Exception raised when the arguments supplied by the user are invalid.
  Raise this when the arguments supplied are invalid from the point of
  view of the application. For example when two mutually exclusive
  flags have been supplied or when there are not enough non-flag
  arguments.
  """

  def __init__(self, message, exitcode=1):
    super(UsageError, self).__init__(message)
    self.exitcode = exitcode


def AssertOrRaise(
  stmt: bool, exception: Exception, *exception_args, **exception_kwargs
) -> None:
  """If the statement is false, raise the given exception class."""
  if not stmt:
    raise exception(*exception_args, **exception_kwargs)


def GetVersionInformationString() -> str:
  """Return a string of version information, as printed by --version flag."""
  # If this is a bazel environment, then the //:build_info package will be
  # available. However, if this is a labm8 pip package install, then
  # //:build_info will not be available, so use pkg_resources to get the
  # version information.
  try:
    import build_info

    version = "\n".join(
      [build_info.FormatVersion(), build_info.FormatShortBuildDescription(),]
    )
    url = build_info.GetGithubCommitUrl()
  except ModuleNotFoundError:
    import pkg_resources

    version = f'version: {pkg_resources.get_distribution("labm8").version}'
    url = "https://github.com/ChrisCummins/labm8"
  return "\n".join(
    [
      version,
      "Copyright (C) 2014-2019 Chris Cummins <chrisc.101@gmail.com>",
      f"<{url}>",
    ]
  )


def RunWithArgs(
  main: Callable[[List[str]], None], argv: Optional[List[str]] = None,
):
  """Begin executing the program.

  Args:
    main: The main function to execute. It takes an single argument "argv",
      which is a list of command line arguments with parsed flags removed.
      If it returns an integer, it is used as the process's exit code.
    argv: A non-empty list of the command line arguments including program name,
      sys.argv is used if None.
  """

  def DoMain(argv):
    """Run the user-provided main method, with app-level arg handling."""
    if FLAGS.version:
      print(GetVersionInformationString())
      sys.exit(0)
    elif FLAGS.dump_flags:
      print(FlagsToString())
      sys.exit(0)
    elif FLAGS.dump_flags_to_json:
      print(
        json.dumps(
          FlagsToDict(), sort_keys=True, indent=2, separators=(",", ": ")
        )
      )
      sys.exit(0)
    main(argv)

  try:
    absl_app.run(DoMain, argv=argv)
  except KeyboardInterrupt:
    FlushLogs()
    sys.stdout.flush()
    sys.stderr.flush()
    print("keyboard interrupt")
    sys.exit(1)


def Run(main: Callable[[], None]):
  """Begin executing the program.

  Args:
    main: The main function to execute. It takes no arguments. If any command
    line arguments remain after flags parsing, an error is raised. If it
    returns an integer, it is used as the process's exit code.
  """

  def RunWithoutArgs(argv: List[str]):
    """Run the given function without arguments."""
    if len(argv) > 1:
      raise UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))
    main()

  RunWithArgs(RunWithoutArgs)


# Logging functions.


def GetVerbosity() -> int:
  """Get the verbosity level.

  This can be set per-module using --vmodule flag.
  """
  return logging.GetModuleVerbosity(logging.GetCallingModuleName())


def _MaybeColorizeLog(color: str, msg: str, *args) -> str:
  """Conditionally apply shell colorization to the given format string."""
  string = str(msg) % args
  if FLAGS.log_colors:
    return f"{shell.ShellEscapeCodes.BOLD}{color}{string}{shell.ShellEscapeCodes.END}"
  else:
    return string


# Skip this function when determining the calling module and line number for
# logging.
@skip_log_prefix
def Log(level: int, msg, *args, **kwargs):
  """Logs a message at the given level.

  Per-module verbose level. The argument has to contain a comma-separated
  list of <module name>=<log level>. <module name> is a glob pattern (e.g., "
    "gfs* for all modules whose name starts with \"gfs\"), matched against the "
    "filename base (that is, name ignoring .py). <log level> overrides any "
    "value given by --v."
  """
  logging.Log(
    logging.GetCallingModuleName(),
    level,
    _MaybeColorizeLog(
      shell.ShellEscapeCodes.YELLOW
      if level > 1
      else shell.ShellEscapeCodes.CYAN,
      msg,
      *args,
    ),
    **kwargs,
  )


@skip_log_prefix
def LogIf(level: int, condition, msg, *args, **kwargs):
  if condition:
    Log(level, msg, *args, **kwargs)


@skip_log_prefix
def Fatal(msg, *args, **kwargs):
  """Logs a fatal message."""
  logging.Fatal(
    _MaybeColorizeLog(shell.ShellEscapeCodes.RED, msg, *args), **kwargs
  )


@skip_log_prefix
def FatalWithoutStackTrace(msg, *args, **kwargs):
  """Logs a fatal message without stacktrace."""
  Error(msg, *args, **kwargs)
  sys.exit(1)


@skip_log_prefix
def Error(msg, *args, **kwargs):
  """Logs an error message."""
  logging.Error(
    _MaybeColorizeLog(shell.ShellEscapeCodes.RED, msg, *args), **kwargs
  )


@skip_log_prefix
def Warning(msg, *args, **kwargs):
  """Logs a warning message."""
  logging.Warning(
    _MaybeColorizeLog(shell.ShellEscapeCodes.RED, msg, *args), **kwargs
  )


def FlushLogs():
  """Flushes all log files."""
  logging.FlushLogs()


# TODO(cec): Consider emoving DebugLogging() in favour of GetVerbosity().
def DebugLogging() -> bool:
  """Return whether debug logging is enabled."""
  return logging.DebugLogging()


def SetLogLevel(level: int) -> None:
  """Sets the logging verbosity.

  Causes all messages of level <= v to be logged, and all messages of level > v
  to be silently discarded.

  Args:
    level: the verbosity level as an integer.
  """
  logging.SetLogLevel(level)


# Flags functions.

# This is a set of module ids for the modules that disclaim key flags.
# This module is explicitly added to this set so that we never consider it to
# define key flag.
disclaim_module_ids = set([id(sys.modules[__name__])])


@functools.lru_cache(maxsize=1)
def get_main_module_name(abspath) -> str:
  # Strip everything until the runfiles directory.
  name = re.sub(r".*\.runfiles/[^/]+/", "", abspath)
  # Strip file extension.
  name = ".".join(name.split(".")[:-1])
  # Change path separator to python module separator.
  name = ".".join(name.split("/"))
  # Prefix name with a space so that it come first in the list of modules.
  return f" {name}"


def get_module_object_and_name(globals_dict):
  """Returns the module that defines a global environment, and its name.
  Args:
    globals_dict: A dictionary that should correspond to an environment
      providing the values of the globals.
  Returns:
    _ModuleObjectAndName - pair of module object & module name.
    Returns (None, None) if the module could not be identified.
  """
  name = globals_dict.get("__name__", None)
  module = sys.modules.get(name, None)
  # Pick a more informative name for the main module.
  return module, (sys.argv[0] if name == "__main__" else name)


def get_calling_module_name():
  """Returns the module that's calling into this module.
  We generally use this function to get the name of the module calling a
  DEFINE_foo... function.
  Returns:
    The module name that called into this one.
  Raises:
    AssertionError: Raised when no calling module could be identified.
  """
  for depth in range(1, sys.getrecursionlimit()):
    # sys._getframe is the right thing to use here, as it's the best
    # way to walk up the call stack.
    globals_for_frame = sys._getframe(
      depth
    ).f_globals  # pylint: disable=protected-access
    module, module_name = get_module_object_and_name(globals_for_frame)
    if id(module) not in disclaim_module_ids and module_name is not None:
      return module_name
  raise AssertionError("No module was found")


# TODO(cec): Add flag_values argument to enable better testing.
# TODO(cec): Add validator callbacks.


def FlagsToDict(json_safe: bool = False) -> Dict[str, Any]:
  """Return a dictionary of flags and their values.

  Keys are the names of flags, prefixed by their defining module, e.g.
  "absl.flags.alsologtosterr" refers to flag "alsologtosterr" to module
  "absl.flags". Values are the string values in their defined types.

  Returns:
    A <flag, value> dictionary.
  """
  flags_dict = FLAGS.flags_by_module_dict()
  flattened_flags_dict = {}
  for module in flags_dict:
    for flag in flags_dict[module]:
      flattened_flags_dict[f"{module}.{flag.name}"] = flag.value

  if json_safe:
    # Flags values can have non-serializable types, so try each one and
    # stringify those that cannot be serialized to JSON. An alternative would
    # be to stringify all values, but this would lose type information on
    # ints/floats/etc.
    for flag in flattened_flags_dict:
      try:
        json.dumps(flattened_flags_dict[flag])
      except TypeError:
        flattened_flags_dict[flag] = str(flattened_flags_dict[flag])

  return flattened_flags_dict


def FlagsToString() -> str:
  """Return the defined flags as a string.

  The string returned by this method is suitable to be used as a flagfile.

  Returns:
    A string of newline-separated flag values, in the form "--someflag=val".
  """
  return FLAGS.flags_into_string()


def DEFINE_string(
  name: str,
  default: Optional[str],
  help: str,
  required: bool = False,
  validator: Callable[[str], bool] = None,
):
  """Registers a flag whose value can be any string."""
  absl_flags.DEFINE_string(
    name, default, help, module_name=get_calling_module_name(),
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_integer(
  name: str,
  default: Optional[int],
  help: str,
  required: bool = False,
  lower_bound: Optional[int] = None,
  upper_bound: Optional[int] = None,
  validator: Callable[[int], bool] = None,
):
  """Registers a flag whose value must be an integer."""
  absl_flags.DEFINE_integer(
    name,
    default,
    help,
    module_name=get_calling_module_name(),
    lower_bound=lower_bound,
    upper_bound=upper_bound,
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_float(
  name: str,
  default: Optional[float],
  help: str,
  required: bool = False,
  lower_bound: Optional[float] = None,
  upper_bound: Optional[float] = None,
  validator: Callable[[float], bool] = None,
):
  """Registers a flag whose value must be a float."""
  absl_flags.DEFINE_float(
    name,
    default,
    help,
    module_name=get_calling_module_name(),
    lower_bound=lower_bound,
    upper_bound=upper_bound,
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_boolean(
  name: str,
  default: Optional[bool],
  help: str,
  required: bool = False,
  validator: Callable[[bool], bool] = None,
):
  """Registers a flag whose value must be a boolean."""
  absl_flags.DEFINE_boolean(
    name, default, help, module_name=get_calling_module_name(),
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_list(
  name: str,
  default: Optional[List[Any]],
  help: str,
  required: bool = False,
  validator: Callable[[List[Any]], bool] = None,
):
  """Registers a flag whose value must be a list."""
  absl_flags.DEFINE_list(
    name, default, help, module_name=get_calling_module_name(),
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


# My custom flag types.


def DEFINE_input_path(
  name: str,
  default: Union[None, str, pathlib.Path],
  help: str,
  required: bool = False,
  is_dir: bool = False,
  validator: Callable[[pathlib.Path], bool] = None,
):
  """Registers a flag whose value is an input path.

  An "input path" is a path to a file or directory that exists. The parsed value
  is a pathlib.Path instance. Flag parsing will fail if the value of this flag
  is not a path to an existing file or directory.

  Args:
    name: The name of the flag.
    default: The default value for the flag. While None is a legal value, it
      will fail during parsing - input paths are required flags.
    help: The help string.
    is_dir: If true, require the that the value be a directory. Else, require
      that the value be a file. Parsing will fail if this is not the case.
  """
  parser = flags_parsers.PathParser(must_exist=True, is_dir=is_dir)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
    parser,
    name,
    default,
    help,
    absl_flags.FLAGS,
    serializer,
    module_name=get_calling_module_name(),
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_output_path(
  name: str,
  default: Union[None, str, pathlib.Path],
  help: str,
  required: bool = False,
  is_dir: bool = False,
  exist_ok: bool = True,
  must_exist: bool = False,
  validator: Callable[[pathlib.Path], bool] = None,
):
  """Registers a flag whose value is an output path.

  An "output path" is a path to a file or directory that may or may not already
  exist. The parsed value is a pathlib.Path instance. The idea is that this flag
  can be used to specify paths to files or directories that will be created
  during program execution. However, note that specifying an output path does
  not guarantee that the file will be produced.

  Args:
    name: The name of the flag.
    default: The default value for the flag. While None is a legal value, it
      will fail during parsing - output paths are required flags.
    help: The help string.
    is_dir: If true, require the that the value be a directory. Else, require
      that the value be a file. Parsing will fail if the path already exists and
      is of the incorrect type.
    exist_ok: If False, require that the path not exist, else parsing will fail.
    must_exist: If True, require that the path exists, else parsing will fail.
  """
  parser = flags_parsers.PathParser(
    must_exist=must_exist, exist_ok=exist_ok, is_dir=is_dir,
  )
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
    parser,
    name,
    default,
    help,
    absl_flags.FLAGS,
    serializer,
    module_name=get_calling_module_name(),
  )
  if required:
    absl_flags.mark_flag_as_required(name)
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_database(
  name: str,
  database_class,
  default: Optional[str],
  help: str,
  must_exist: bool = False,
  validator: Callable[[Any], bool] = None,
):
  """Registers a flag whose value is a sqlutil.Database class.

  Unlike other DEFINE_* functions, the value produced by this flag is not an
  instance of the value, but a lambda that will instantiate a database of the
  requested type. This flag value must be called (with no arguments) in order to
  instantiate a database.

  Args:
    name: The name of the flag.
    database_class: The subclass of sqlutil.Database which is to be instantiated
      when this value is called, using the URL declared in 'default'.
    default: The default URL of the database. This is a required value.
    help: The help string.
    must_exist: If True, require that the database exists. Else, the database is
      created if it does not exist.
  """
  parser = flags_parsers.DatabaseParser(database_class, must_exist=must_exist)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
    parser,
    name,
    default,
    help,
    absl_flags.FLAGS,
    serializer,
    module_name=get_calling_module_name(),
  )
  if validator:
    RegisterFlagValidator(name, validator)


def DEFINE_enum(
  name: str,
  enum_class,
  default,
  help: str,
  validator: Callable[[Any], bool] = None,
):
  """Registers a flag whose value is an enum.Enum class.

  Unlike other DEFINE_* functions, the value produced by this flag is not an
  instance of the value, but a lambda that will instantiate a database of the
  requested type. This flag value must be called (with no arguments) in order
  to instantiate an enum.

  Args:
    name: The name of the flag.
    enum_class: The subclass of enum.Enum which is to be instantiated when this
      value is called.
    default: The default value of the enum. Either the string name or an enum
      value.
    help: The help string.
    must_exist: If True, require that the database exists. Else, the database is
      created if it does not exist.
  """
  parser = flags_parsers.EnumParser(enum_class)
  serializer = absl_flags.ArgumentSerializer()
  absl_flags.DEFINE(
    parser,
    name,
    default,
    help,
    absl_flags.FLAGS,
    serializer,
    module_name=get_calling_module_name(),
  )
  if validator:
    RegisterFlagValidator(name, validator)


def RegisterFlagValidator(
  flag_name: str,
  checker: Callable[[Any], bool],
  message: str = "Flag validation failed",
):
  """Adds a constraint, which will be enforced during program execution.

  The constraint is validated when flags are initially parsed, and after each
  change of the corresponding flag's value.

  Args:
    flag_name: str, name of the flag to be checked.
    checker: callable, a function to validate the flag.
        input - A single positional argument: The value of the corresponding
            flag (string, boolean, etc.  This value will be passed to checker
            by the library).
        output - bool, True if validator constraint is satisfied.
            If constraint is not satisfied, it should either return False or
            raise flags.ValidationError(desired_error_message).
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.

  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  """
  absl_flags.register_validator(flag_name, checker, message)


def LogToDirectory(
  logdir: Union[str, pathlib.Path], name="info"
) -> pathlib.Path:
  """Write logs to a directory.

  This disables printing of logs to stderr, unless the --alsologtostderr flag
  is provided.

  Args:
    logdir: The directory to write logs to. This is created if it does not
      exist.
    name: The name of the log file.
  """
  logdir = pathlib.Path(logdir)
  logdir.mkdir(exist_ok=True, parents=True)
  absl_logging.get_absl_handler().use_absl_log_file(str(name), str(logdir))
  return logdir


# Get the thread ID as an unsigned integer.
UnsignedThreadId = logging.UnsignedThreadId
