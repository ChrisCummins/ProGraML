"""Implementations of custom flag types for absl.flags."""
import enum
import pathlib
from typing import Callable

from absl import app as absl_app
from absl import flags as absl_flags


class PathParser(absl_flags.ArgumentParser):
  """Parser of path values."""

  def __init__(
    self, must_exist: bool = True, exist_ok: bool = True, is_dir: bool = False
  ):
    """Create a path values parser.

    Args:
      must_exist: If true, the path mst exist.
      exist_ok: If not true, the path must not exist. Not effect if true.
        Implied true if must_exist.
      is_dir: If true, the path must be a directory.
    """
    self.must_exist = must_exist
    self.exist_ok = exist_ok
    self.is_dir = is_dir

  def parse(self, argument) -> pathlib.Path:
    """See base class."""
    val = self.convert(argument)
    if self.must_exist:
      if not val.exists():
        raise ValueError("not found")
      if self.is_dir and not val.is_dir():
        raise ValueError("not a directory")
      elif not self.is_dir and not val.is_file():
        raise ValueError("not a file")
    elif not self.exist_ok and val.exists():
      raise ValueError("already exists")
    return val

  def convert(self, argument: str) -> pathlib.Path:
    """Returns the value of this argument."""
    if not argument:
      raise absl_app.UsageError("Path flag must be set")
    return pathlib.Path(argument)


class DatabaseFlag:
  """A parsed database. This is instantiated by DatabaseParser.convert() and
  used to provide a repr()-friendly method for instantiating databases.

  E.g. repr(FLAGS.db) will yield "sqlite:////path/to/db" rather than an
  anonymous lambda.
  """

  def __init__(self, database_class, url: str, must_exist: bool):
    self.url = url
    self.database_class = database_class
    self.must_exist = must_exist

  def __call__(self):
    try:
      return self.database_class(url=self.url, must_exist=self.must_exist)
    except Exception as e:
      raise absl_app.UsageError(
        f"Failed to construct database {self.database_class.__name__}({self.url}): {e}"
      )

  def __repr__(self):
    return str(self.url)

  def __str__(self):
    return self.__repr__()


class DatabaseParser(absl_flags.ArgumentParser):
  """Parser of path values."""

  def __init__(self, database_class, must_exist: bool = True):
    """Create a path values parser.

    Args:
      must_exist: If true, the database must exist. Else, it is created.
    """
    # TODO(cec): Raise TypeError if database_class is not a subclass of
    # 'sqlutil.Database'.
    self.database_class = database_class
    self.must_exist = must_exist

  def parse(self, argument) -> "sqlutil.Database":
    """See base class."""
    return self.convert(argument)

  def convert(self, argument: str) -> DatabaseFlag:
    """Returns the value of this argument."""
    if not argument:
      raise absl_app.UsageError("Database flag must be set")
    return DatabaseFlag(self.database_class, argument, self.must_exist)


class EnumFlag:
  """A parsed enum. This is instantiated by EnumParser.convert() and
  used to provide a repr()-friendly method for instantiating databases.

  E.g. repr(FLAGS.enum) will yield "foo" rather than an
  anonymous lambda.
  """

  def __init__(self, enum_class, name: str):
    self.name = name
    self.enum_class = enum_class

  def __call__(self):
    # Value is already an enum, e.g. the default value.
    if isinstance(self.name, self.enum_class):
      return self.name

    try:
      return self.enum_class[self.name.upper()]
    except KeyError as e:
      valid_options = [value.name.lower() for value in self.enum_class]
      raise absl_app.UsageError(
        f"Invalid {self.enum_class.__name__}={e}. "
        f"Valid values={valid_options}"
      )

  def __repr__(self):
    return str(self.name)

  def __str__(self):
    return self.__repr__()


class EnumParser(absl_flags.ArgumentParser):
  """Parser of enums."""

  def __init__(self, enum_class):
    """Create a enum parser."""
    self.enum_class = enum_class

  def parse(self, argument) -> enum.Enum:
    """See base class."""
    return self.convert(argument)

  def convert(self, argument: str) -> EnumFlag:
    """Returns the value of this argument."""
    if not argument:
      raise TypeError("Enum flag must be set")
    return EnumFlag(self.enum_class, argument)
