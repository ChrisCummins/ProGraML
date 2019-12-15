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
"""Utility code for working with sqlalchemy."""
import contextlib
import os
import pathlib
import queue
import sqlite3
import sys
import threading
import time
import typing
from typing import Callable
from typing import List
from typing import Optional

import sqlalchemy as sql
from absl import flags as absl_flags
from sqlalchemy import func
from sqlalchemy import orm
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from labm8.py import humanize
from labm8.py import labdate
from labm8.py import pbutil
from labm8.py import progress
from labm8.py import text
from labm8.py.internal import logging

FLAGS = absl_flags.FLAGS

absl_flags.DEFINE_boolean(
  "sqlutil_echo",
  False,
  "If True, the Engine will log all statements as well as a repr() of their "
  "parameter lists to the engines logger, which defaults to sys.stdout.",
)
absl_flags.DEFINE_boolean(
  "sqlutil_pool_pre_ping",
  True,
  "Enable pessimistic pre-ping to check that database connections are "
  "alive. This adds some overhead, but reduces the risk of "
  '"server has gone away" errors. See:'
  "<https://docs.sqlalchemy.org/en/13/core/pooling.html#disconnect-handling-pessimistic>",
)
absl_flags.DEFINE_integer(
  "mysql_engine_pool_size",
  5,
  "The number of connections to keep open inside the connection pool. A "
  "--mysql_engine_pool_size of 0 indicates no limit",
)
absl_flags.DEFINE_integer(
  "mysql_engine_max_overflow",
  10,
  "The number of connections to allow in connection pool “overflow”, that "
  "is connections that can be opened above and beyond the "
  "--mysql_engine_pool_size setting",
)
absl_flags.DEFINE_boolean(
  "mysql_assume_utf8_charset",
  True,
  "Default to adding the '?charset=utf8' suffix to MySQL database URLs.",
)
absl_flags.DEFINE_boolean(
  "sqlite_enable_foreign_keys",
  True,
  "Enable foreign key support for SQLite. This enforces foreign key "
  "constraints, and enables cascaded update/delete statements. See: "
  "https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#foreign-key-support",
)

# The Query type is returned by Session.query(). This is a convenience for type
# annotations.
Query = orm.query.Query


class DatabaseNotFound(FileNotFoundError):
  """An error that is raised if the requested database cannot be found."""

  def __init__(self, url: str):
    self._url = url

  @property
  def url(self):
    return self._url

  def __repr__(self) -> str:
    return f"Database not found: '{self.url}'"

  def __str__(self) -> str:
    return repr(self)


def Base(*args, **kwargs) -> sql.ext.declarative.DeclarativeMeta:
  """Construct a base class for declarative class definitions."""
  return sql.ext.declarative.declarative_base(*args, **kwargs)


def GetOrAdd(
  session: sql.orm.session.Session,
  model,
  defaults: typing.Dict[str, object] = None,
  **kwargs,
):
  """Instantiate a mapped database object.

  If the object is not in the database,
  add it. Note that no change is written to disk until commit() is called on the
  session.

  Args:
    session: The database session.
    model: The database table class.
    defaults: Default values for mapped objects.
    kwargs: The values for the table row.

  Returns:
    An instance of the model class, with the values specified.
  """
  instance = session.query(model).filter_by(**kwargs).first()
  if not instance:
    params = {
      k: v
      for k, v in kwargs.items()
      if not isinstance(v, sql.sql.expression.ClauseElement)
    }
    params.update(defaults or {})
    instance = model(**params)
    session.add(instance)
    logging.Log(
      logging.GetCallingModuleName(),
      5,
      "New record: %s(%s)",
      model.__name__,
      params,
    )
  return instance


def Get(
  session: sql.orm.session.Session,
  model,
  defaults: typing.Dict[str, object] = None,
  **kwargs,
):
  """Determine if a database object exists.

  Args:
    session: The database session.
    model: The database table class.
    defaults: Default values for mapped objects.
    kwargs: The values for the table row.

  Returns:
    An instance of the model class with the values specified, or None if the
    object is not in the database.
  """
  del defaults
  return session.query(model).filter_by(**kwargs).first()


def CreateEngine(url: str, must_exist: bool = False) -> sql.engine.Engine:
  """Create an sqlalchemy database engine.

  This is a convenience wrapper for creating an sqlalchemy engine, that also
  creates the database if required, and checks that the database exists. This
  means that it is less flexible than SqlAlchemy's create_engine() - only three
  combination of dialects and drivers are supported: sqlite, mysql, and
  postgresql.

  See https://docs.sqlalchemy.org/en/latest/core/engines.html for details.

  Additionally, this implements a custom 'file://' handler, which reads a URL
  from a local file, and returns a connection to the database addressed by the
  URL. Use this if you would like to keep sensitive information such as a MySQL
  database password out of your .bash_history.

  Examples:
    Create in-memory SQLite database:
    >>> engine = CreateEngine('sqlite://')

    Connect to an SQLite database at relative.db:
    >>> engine = CreateEngine('sqlite:///relative.db')

    Connect to an SQLite database at /absolute/path/to/db:
    >>> engine = CreateEngine('sqlite:////absolute/path/to/db')

    Connect to MySQL database:
    >>> engine = CreateEngine(
        'mysql://bob:password@localhost:1234/database?charset=utf8')

    Connect to PostgreSQL database:
    >>> engine.CreateEngine(
      'postgresql://bob:password@localhost:1234/database')

    Connect to a URL specified in the file /tmp/url.txt:
    >>> engine.CreateEngine('file:///tmp/url.txt')

    Connect to a URL specified in the file /tmp/url.txt, with the suffix
    '/database?charset=utf8':
    >>> engine.CreateEngine('file:///tmp/url.txt?/database?charset=utf8')

  Args:
    url: The URL of the database to connect to.
    must_exist: If True, raise DatabaseNotFound if it doesn't exist. Else,
        database is created if it doesn't exist.

  Returns:
    An SQLalchemy Engine instance.

  Raises:
    DatabaseNotFound: If the database does not exist and must_exist is set.
    ValueError: If the datastore backend is not supported.
  """
  engine_args = {}

  # Read and expand a `file://` prefixed URL.
  url = ResolveUrl(url)

  if url.startswith("mysql://"):
    # Support for MySQL dialect.

    # We create a throwaway engine that we use to check if the requested
    # database exists.
    engine = sql.create_engine("/".join(url.split("/")[:-1]))
    database = url.split("/")[-1].split("?")[0]
    query = engine.execute(
      sql.text(
        "SELECT SCHEMA_NAME FROM "
        "INFORMATION_SCHEMA.SCHEMATA WHERE "
        "SCHEMA_NAME = :database",
      ),
      database=database,
    )

    # Engine-specific options.
    engine_args["pool_size"] = FLAGS.mysql_engine_pool_size
    engine_args["max_overflow"] = FLAGS.mysql_engine_max_overflow

    if not query.first():
      if must_exist:
        raise DatabaseNotFound(url)
      else:
        # We can't use sql.text() escaping here because it uses single quotes
        # for escaping. MySQL only accepts backticks for quoting database
        # names.
        engine.execute(f"CREATE DATABASE `{database}`")
    engine.dispose()
  elif url.startswith("sqlite://"):
    # Support for SQLite dialect.

    # This project (phd) deliberately disallows relative paths due to Bazel
    # sandboxing.
    if url != "sqlite://" and not url.startswith("sqlite:////"):
      raise ValueError("Relative path to SQLite database is not allowed")

    if url == "sqlite://":
      if must_exist:
        raise ValueError(
          "must_exist=True not valid for in-memory SQLite database",
        )
    else:
      path = pathlib.Path(url[len("sqlite:///") :])
      if must_exist:
        if not path.is_file():
          raise DatabaseNotFound(url)
      else:
        # Make the parent directory for SQLite database if creating a new
        # database.
        path.parent.mkdir(parents=True, exist_ok=True)
  elif url.startswith("postgresql://"):
    # Support for PostgreSQL dialect.

    engine = sql.create_engine("/".join(url.split("/")[:-1] + ["postgres"]))
    conn = engine.connect()
    database = url.split("/")[-1]
    query = conn.execute(
      sql.text("SELECT 1 FROM pg_database WHERE datname = :database"),
      database=database,
    )
    if not query.first():
      if must_exist:
        raise DatabaseNotFound(url)
      else:
        # PostgreSQL does not let you create databases within a transaction, so
        # manually complete the transaction before creating the database.
        conn.execute(sql.text("COMMIT"))
        # PostgreSQL does not allow single quoting of database names.
        conn.execute(f"CREATE DATABASE {database}")
    conn.close()
    engine.dispose()
  else:
    raise ValueError(f"Unsupported database URL='{url}'")

  # Create the engine.
  engine = sql.create_engine(
    url,
    encoding="utf-8",
    echo=FLAGS.sqlutil_echo,
    pool_pre_ping=FLAGS.sqlutil_pool_pre_ping,
    **engine_args,
  )

  # Create and immediately close a connection. This is because SQLAlchemy engine
  # is lazily instantiated, so for connections such as SQLite, this line
  # actually creates the file.
  engine.connect().close()

  return engine


@sql.event.listens_for(sql.engine.Engine, "connect")
def EnableSqliteForeignKeysCallback(dbapi_connection, connection_record):
  """Enable foreign key constraints for SQLite databases.

  See --sqlite_enable_foreign_keys for details.
  """
  del connection_record
  # This callback listens for *all* database connections, not just SQLite. Check
  # the type before trying to run an SQLite-specific pragma.
  if FLAGS.sqlite_enable_foreign_keys and isinstance(
    dbapi_connection, sqlite3.Connection
  ):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def ResolveUrl(url: str, use_flags: bool = True):
  """Resolve the URL of a database.

  The following modifications are supported:
    * If the url begins with 'file://', the URL is substituted with the
      contents of the file.
    * If --mysql_assume_utf8_charset is set, then '?charset=utf8' suffix is
      appended to URLs which begin with mysql://.
    * Shell variables are expanded.

  Args:
    url: The URL to expand, e.g. `file://path/to/file.txt?arg'
    use_flags: Determine whether behaviour is dictated by the FLAGS variables.
      Set this to False only when resolving database URLs before flags parsing,
      e.g. in enumerating test fixtures.

  Returns:
    The URL as interpreted by reading any URL file.

  Raises:
    ValueError: If the file path is invalid.
    FileNotFoundError: IF the file path does not exist.
  """
  # Substitute shell variables.
  url = os.path.expandvars(url)

  if url.startswith("file://"):
    # Split the URL into the file path, and the optional suffix.
    components = url.split("?")
    path, suffix = components[0], "?".join(components[1:])

    # Strip the file:// prefix from the path.
    path = pathlib.Path(path[len("file://") :])

    if not path.is_absolute():
      raise ValueError("Relative path to file:// is not allowed")

    if not path.is_file():
      raise FileNotFoundError(f"File '{path}' not found")

    # Read the contents of the file, ignoring lines starting with '#'.
    with open(path) as f:
      url = "\n".join(
        x for x in f.read().split("\n") if not x.lstrip().startswith("#")
      ).strip()

    # Append the suffix.
    url += suffix

  if (
    use_flags and url.startswith("mysql://") and FLAGS.mysql_assume_utf8_charset
  ):
    url += "?charset=utf8"

  return url


def ColumnNames(model) -> typing.List[str]:
  """Return the names of all columns in a mapped object.

  Args:
    model: A mapped class.

  Returns:
    A list of string column names in the order that they are declared.
  """
  try:
    inst = sql.inspect(model)
    return [c_attr.key for c_attr in inst.mapper.column_attrs]
  except sql.exc.NoInspectionAvailable as e:
    raise TypeError(str(e))


class Session(orm.session.Session):
  """A subclass of the default SQLAlchemy Session with added functionality.

  An instance of this class is returned by Database.Session().
  """

  def GetOrAdd(
    self, model, defaults: typing.Dict[str, object] = None, **kwargs
  ):
    """Instantiate a mapped database object.

    If the object is not in the database, add it. Note that no change is written
    to disk until commit() is called on the session.

    Args:
      model: The database table class.
      defaults: Default values for mapped objects.
      kwargs: The values for the table row.

    Returns:
      An instance of the model class, with the values specified.
    """
    return GetOrAdd(self, model, defaults, **kwargs)


class Database(object):
  """A base class for implementing databases."""

  SessionType = Session

  def __init__(self, url: str, declarative_base, must_exist: bool = False):
    """Instantiate a database object.

    Example:
      >>> db = Database('sqlite:////tmp/foo.db',
                        sqlalchemy.ext.declarative.declarative_base())

    Args:
      url: The URL of the database to connect to.
      declarative_base: The SQLAlchemy declarative base instance.
      must_exist: If True, raise DatabaseNotFound if it doesn't exist. Else,
        database is created if it doesn't exist.

    Raises:
      DatabaseNotFound: If the database does not exist and must_exist is set.
      ValueError: If the datastore backend is not supported.
    """
    self._url = url
    self.engine = CreateEngine(url, must_exist=must_exist)
    declarative_base.metadata.create_all(self.engine)
    declarative_base.metadata.bind = self.engine

    # Bind the Engine to a session maker, which instantiates our own Session
    # class, which is a subclass of the default SQLAlchemy Session with added
    # functionality.
    self.MakeSession = orm.sessionmaker(bind=self.engine, class_=Session)

  def Close(self) -> None:
    """Close the connection to the database.

    Use this to free up the connection to a database, while keeping the database
    instance around. After calling this method, attempting to run operations on
    this database will raise an error (like a sqlalchemy.exc.OperationalError).

    Usage of this method is generally discouraged - connections are
    automatically closed up when a database instance is garbage collected, so
    there are rarely cases for leaving a database instance around with the
    connection closed. Use at your peril!
    """
    self.engine.dispose()

  def Drop(self, are_you_sure_about_this_flag: bool = False):
    """Drop the database, irreverisbly destroying it.

    Be careful with this! After calling this method an a Database instance, no
    further operations can be made on it, and any Sessions should be discarded.

    Args:
      are_you_sure_about_this_flag: You should be sure.

    Raises:
      ValueError: In case you're not 100% sure.
    """
    if not are_you_sure_about_this_flag:
      raise ValueError("Let's take a minute to think things over")

    if self.url.startswith("mysql://"):
      engine = sql.create_engine("/".join(self.url.split("/")[:-1]))
      database = self.url.split("/")[-1].split("?")[0]
      logging.Log(logging.GetCallingModuleName(), 1, "database %s", database)
      engine.execute(f"DROP DATABASE IF EXISTS `{database}`")
    elif self.url == "sqlite://":
      # In-memory databases do not dropping.
      pass
    elif self.url.startswith("sqlite:///"):
      path = pathlib.Path(self.url[len("sqlite:///") :])
      assert path.is_file()
      path.unlink()
    else:
      raise NotImplementedError(
        f"Unsupported operation DROP for database: '{self.url}'",
      )

  @property
  def url(self) -> str:
    """Return the URL of the database."""
    return self._url

  @contextlib.contextmanager
  def Session(
    self, commit: bool = False, session: Optional[Session] = None
  ) -> Session:
    """Provide a transactional scope around a session.

    The optional session argument may be used for cases where you want to
    optionally re-use an existing session, rather than always creating a new
    session, e.g.:

      class MyDatabase(sqlutil.Database):
        def DoAThing(self, session=None):
          with self.Session(session=session, commit=True):
            # go nuts ...

    Args:
      commit: If true, commit session at the end of scope.
      session: An existing session object to re-use.

    Returns:
      A database session.
    """
    session = session or self.MakeSession()
    try:
      yield session
      if commit:
        session.commit()
    except:
      session.rollback()
      raise
    finally:
      session.close()

  @property
  def Random(self):
    """Get the backend-specific random function.

    This can be used to select a random row from a table, e.g.
        session.query(Table).order_by(db.Random()).first()
    """
    if self.url.startswith("mysql"):
      return func.rand
    else:
      return func.random  # for PostgreSQL, SQLite

  def __repr__(self) -> str:
    return self.url


class TablenameFromClassNameMixin(object):
  """A class mixin which derives __tablename__ from the class name.

  Add this mixin to a mapped table class to automatically set the set the
  __tablename__ property of a class to the lowercase name of the Python class.
  """

  @declarative.declared_attr
  def __tablename__(self):
    return self.__name__.lower()


class TablenameFromCamelCapsClassNameMixin(object):
  """A class mixin which derives __tablename__ from the class name.

  Add this mixin to a mapped table class to automatically set the set the
  __tablename__ property of a class to the name of the Python class with camel
  caps converted to underscores, e.g.

    class FooBar -> table "foo_bar".
  """

  @declarative.declared_attr
  def __tablename__(self):
    return text.CamelCapsToUnderscoreSeparated(self.__name__)


class PluralTablenameFromCamelCapsClassNameMixin(object):
  """A class mixin which derives __tablename__ from the class name.

  Add this mixin to a mapped table class to automatically set the set the
  __tablename__ property of a class to the pluralized name of the Python class
  with camel caps converted to underscores, e.g.

    class FooBar -> table "foo_bars".
  """

  @declarative.declared_attr
  def __tablename__(self):
    pluralised = humanize.Plural(2, self.__name__)
    pluralised = " ".join(pluralised.split()[1:])
    return text.CamelCapsToUnderscoreSeparated(pluralised)


class ProtoBackedMixin(object):
  """A database table backed by protocol buffers.

  This class provides the abstract interface for sqlalchemy table classes which
  support serialization to and from protocol buffers.

  This is only an interface - inheriting classes must still inherit from
  sqlalchemy.ext.declarative.declarative_base().
  """

  proto_t = None

  def SetProto(self, proto: pbutil.ProtocolBuffer) -> None:
    """Set the fields of a protocol buffer with the values from the instance.

    Args:
      proto: A protocol buffer.
    """
    raise NotImplementedError(
      f"{type(self).__name__}.SetProto() not implemented",
    )

  def ToProto(self) -> pbutil.ProtocolBuffer:
    """Serialize the instance to protocol buffer.

    Returns:
      A protocol buffer.
    """
    proto = self.proto_t()
    self.SetProto(proto)
    return proto

  @classmethod
  def FromProto(
    cls, proto: pbutil.ProtocolBuffer,
  ) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto.

    Examples:
      Construct a table instance from proto:
      >>> table = Table(**Table.FromProto(proto))

      Construct a table instance and add to session:
      >>> session.GetOrAdd(Table, **Table.FromProto(proto))

    Args:
      proto: A protocol buffer.

    Returns:
      A dictionary of constructor arguments.
    """
    raise NotImplementedError(
      f"{type(self).__name__}.FromProto() not implemented",
    )

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    """Return a dictionary of instance constructor args from proto file.

    Examples:
      Construct a table instance from proto file:
      >>> table = Table(**Table.FromFile(path))

      Construct a table instance and add to session:
      >>> session.GetOrAdd(Table, **Table.FromFile(path))

    Args:
      path: Path to a proto file.

    Returns:
      An instance.
    """
    proto = pbutil.FromFile(path, cls.proto_t())
    return cls.FromProto(proto)


class OffsetLimitQueryResultsBatch(typing.NamedTuple):
  """The results of an offset-limit batched query."""

  # The current batch number.
  batch_num: int
  # Offset into the results set.
  offset: int
  # Limit is the last row in the results set.
  limit: int
  # The total number of rows in the query if compute_max_rows=True, else None.
  max_rows: int
  # The results of the query.
  rows: typing.List[typing.Any]


def OffsetLimitBatchedQuery(
  query: Query,
  batch_size: int = 1000,
  start_at: int = 0,
  compute_max_rows: bool = False,
) -> typing.Iterator[OffsetLimitQueryResultsBatch]:
  """Split and return the rows resulting from a query in to batches.

  This iteratively runs the query `SELECT * FROM * OFFSET i LIMIT batch_size;`
  with `i` initialized to `start_at` and increasing by `batch_size` per
  iteration. Iteration terminates when the query returns no rows.

  This function is useful for returning row sets from enormous tables, where
  loading the full query results in to memory would take prohibitive time or
  resources.

  Args:
    query: The query to run.
    batch_size: The number of rows to return per batch.
    start_at: The initial offset into the table.
    compute_max_rows: If true

  Returns:
    A generator of OffsetLimitQueryResultsBatch tuples, where each tuple
    contains between 1 <= x <= `batch_size` rows.
  """
  max_rows = None
  if compute_max_rows:
    max_rows = query.count()

  batch_num = 0
  i = start_at
  while True:
    batch_num += 1
    batch = query.offset(i).limit(batch_size).all()
    if batch:
      yield OffsetLimitQueryResultsBatch(
        batch_num=batch_num,
        offset=i,
        limit=i + batch_size,
        max_rows=max_rows,
        rows=batch,
      )
      i += len(batch)
    else:
      break


class ColumnTypes(object):
  """Abstract class containing methods for generating column types."""

  def __init__(self):
    raise TypeError("abstract class")

  @staticmethod
  def BinaryArray(length: int):
    """Return a fixed size binary array column type.

    Args:
      length: The length of the column.

    Returns:
      A column type.
    """
    return sql.Binary(length).with_variant(mysql.BINARY(length), "mysql")

  @staticmethod
  def LargeBinary():
    """Return a fixed size binary array column type.

    Returns:
      A column type.
    """
    return sql.LargeBinary().with_variant(sql.LargeBinary(2 ** 31), "mysql")

  @staticmethod
  def UnboundedUnicodeText():
    """Return an unbounded unicode text column type.

    This isn't truly unbounded, but 2^32 chars should be enough!

    Returns:
      A column type.
    """
    return sql.UnicodeText().with_variant(sql.UnicodeText(2 ** 31), "mysql")

  @staticmethod
  def IndexableString(length: int = None):
    """Return a string that is short enough that it can be used as an index.

    Returns:
      A column type.
    """
    # MySQL InnoDB tables use a default index key prefix length limit of 767.
    # https://dev.mysql.com/doc/refman/5.6/en/innodb-restrictions.html
    MAX_LENGTH = 767
    if length and length > MAX_LENGTH:
      raise ValueError(
        f"IndexableString requested length {length} is greater "
        f"than maximum allowed {MAX_LENGTH}",
      )
    return sql.String(MAX_LENGTH)

  @staticmethod
  def MillisecondDatetime():
    """Return a datetime type with millisecond precision.

    Returns:
      A column type.
    """
    return sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql")


class ColumnFactory(object):
  """Abstract class containing methods for generating columns."""

  @staticmethod
  def MillisecondDatetime(
    nullable: bool = False, default=labdate.GetUtcMillisecondsNow,
  ):
    """Return a datetime column with millisecond precision.

    Returns:
      A column which defaults to UTC now.
    """
    return sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql",),
      nullable=nullable,
      default=default,
    )


def ResilientAddManyAndCommit(db: Database, mapped: typing.Iterable[Base]):
  """Attempt to commit all mapped objects and return those that fail.

  This method creates a session and commits the given mapped objects.
  In case of error, this method will recurse up to O(log(n)) times, committing
  as many objects that can be as possible.

  Args:
    db: The database to add the objects to.
    mapped: A sequence of objects to commit.

  Returns:
    Any items in `mapped` which could not be committed, if any. Relative order
    of items is preserved.
  """
  failures = []

  if not mapped:
    return failures

  mapped = list(mapped)
  try:
    with db.Session(commit=True) as session:
      session.add_all(mapped)
  except sql.exc.SQLAlchemyError as e:
    logging.Log(
      logging.GetCallingModuleName(),
      1,
      "Caught error while committing %d mapped objects: %s",
      len(mapped),
      e,
    )

    # Divide and conquer. If we're committing only a single object, then a
    # failure to commit it means that we can do nothing other than return it.
    # Else, divide the mapped objects in half and attempt to commit as many of
    # them as possible.
    if len(mapped) == 1:
      return mapped
    else:
      mid = int(len(mapped) / 2)
      left = mapped[:mid]
      right = mapped[mid:]
      failures += ResilientAddManyAndCommit(db, left)
      failures += ResilientAddManyAndCommit(db, right)

  return failures


class BufferedDatabaseWriter(threading.Thread):
  """A buffered writer for adding objects to a database.

  Use this class for cases when you are producing lots of mapped objects that
  you would like to commit to a database, but don't require them to be committed
  immediately. By buffering objects and committing them in batches, this class
  minimises the number of SQL statements that are executed, and is faster than
  creating and committing a session for every object.

  This object spawns a separate thread for asynchronously performing database
  writes. Use AddOne() and AddMany() methods to add objects to the write buffer.
  Note that because this is a multithreaded implementation, in-memory SQLite
  databases are not supported.

  The user is responsible for calling Close() to flush the contents of the
  buffer and terminate the thread. Alternatively, use this class as a context
  manager to automatically flush the buffer and terminate the thread:

    with BufferedDatabaseWriter(db, max_buffer_length=128) as writer:
      for chunk in chunks_to_process:
        objs = ProcessChunk(chunk)
        writer.AddMany(objs)
  """

  def __init__(
    self,
    db: Database,
    max_buffer_size: Optional[int] = None,
    max_buffer_length: Optional[int] = None,
    max_seconds_since_flush: Optional[float] = None,
    log_level: int = 2,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    """Constructor.

    Args:
      db: The database to write to.
      max_buffer_size: The maximum size of the buffer before flushing, in bytes.
        The buffer size is the sum of the elements in the write buffer. The size
        of elements is determined using sys.getsizeof(), and has all the caveats
        of this method.
      max_buffer_length: The maximum number of items in the write buffer before
        flushing.
      max_seconds_since_flush: The maximum number of elapsed seconds between
        flushes.
      ctx: progress.ProgressContext = progress.NullContext,
      log_level: The logging level for logging output.
    """
    super(BufferedDatabaseWriter, self).__init__()
    self.db = db
    self.ctx = ctx
    self.log_level = log_level

    self.max_seconds_since_flush = max_seconds_since_flush
    self.max_buffer_size = max_buffer_size
    self.max_buffer_length = max_buffer_length

    # Counters.
    self.flush_count = 0
    self.error_count = 0

    self._buffer = []
    self.buffer_size = 0
    self._last_flush = time.time()

    # Limit the size of the queue so that calls to AddOne() or AddMany() will
    # block if the calling code is too far ahead of the writer.
    queue_size = self.max_buffer_length * 2 if self.max_buffer_length else 1000
    self._queue = queue.Queue(maxsize=queue_size)

    self.start()

  def __enter__(self) -> "Buff":
    """Enter a scoped writer context closes at the end."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exit a scoped writer context closes at the end."""
    del exc_type
    del exc_val
    del exc_tb
    self.Close()

  def AddOne(self, mapped, size: Optional[int] = None) -> None:
    """Add a mapped object.

    Args:
      mapped: The mapped object to write to the database.
      size: The object sizes to use to update the total buffer size. If not
        provided, sys.getsizeof() is used to determine the size.
    """
    size = size or sys.getsizeof(mapped)
    self._queue.put((mapped, size))

  def AddMany(self, mappeds, sizes: Optional[List[int]] = None) -> None:
    """Add many mapped objects.

    Args:
      mappeds: The mapped objects to write to the database.
      sizes: A list of mapped object sizes to use to calculate the buffer size.
        If not provided, sys.getsizeof() is used to determine the size.
    """
    sizes = sizes or [sys.getsizeof(item) for item in mappeds]

    for mapped, size in zip(mappeds, sizes):
      self._queue.put((mapped, size))

  def AddLambdaOp(self, callback: Callable[[Database.SessionType], None]):
    self._queue.put(BufferedDatabaseWriter.LambdaOp(callback))

  def Flush(self) -> None:
    """Flush the buffer.

    This method blocks until the flush has completed.

    In normal use, you can rely on the automated flushing mechanisms to flush
    the write buffer, rather than calling this by hand.
    """
    self._queue.put(BufferedDatabaseWriter.FlushMarker())
    self._queue.join()

  def Close(self):
    """Close the writer thread.

    This method blocks until the buffer has been flushed and the thread
    terminates.
    """
    if not self.is_alive():
      raise TypeError("Close() called on dead BufferedDatabaseWriter")
    self._queue.put(BufferedDatabaseWriter.CloseMarker())
    self._queue.join()
    self.join()

  @property
  def buffer_length(self) -> int:
    """Get the current length of the buffer, in range [0, max_buffer_length]."""
    return len(self._buffer)

  @property
  def seconds_since_last_flush(self) -> float:
    """Get the number of seconds since the buffer was last flushed."""
    return time.time() - self._last_flush

  ##############################################################################
  # Private methods.
  ##############################################################################

  class CloseMarker(object):
    """An object to append to _queue to close the thread."""

    pass

  class FlushMarker(object):
    """An object to append to _queue to flush the buffer."""

    pass

  class LambdaOp(object):
    def __init__(self, callback):
      self.callback = callback

    def __call__(self, session: Database.SessionType):
      self.callback(session)

  def run(self):
    """The thread loop."""
    while True:
      # Block until there is something on the queue. Use max_seconds_since_flush
      # as a timeout to ensure that flushes still occur when the writer is not
      # being used.
      try:
        item = self._queue.get(timeout=self.max_seconds_since_flush)
      except queue.Empty:
        self._Flush()
        continue

      if isinstance(item, BufferedDatabaseWriter.CloseMarker):
        # End of queue. Break out of the loop.
        break
      elif isinstance(item, BufferedDatabaseWriter.FlushMarker):
        # Force a flush.
        self._Flush()
      elif isinstance(item, BufferedDatabaseWriter.LambdaOp):
        # Handle delete op.
        self._buffer.append(item)
        self._MaybeFlush()
      else:
        # Add the object to the buffer.
        mapped, size = item
        self._buffer.append(mapped)
        self.buffer_size += size
        self._MaybeFlush()

      # Register that the item has been processed. This is used by join() to
      # signal to stop blocking.
      self._queue.task_done()

    # Register that the end-of-queue marker has been processed.
    self._Flush()
    self._queue.task_done()

  def _MaybeFlush(self) -> None:
    if (
      (self.max_buffer_size and self.buffer_size >= self.max_buffer_size)
      or (
        self.max_buffer_length and self.buffer_length >= self.max_buffer_length
      )
      or (
        self.max_seconds_since_flush
        and self.seconds_since_last_flush >= self.max_seconds_since_flush
      )
    ):
      self._Flush()

  def _AddMapped(self, mapped) -> None:
    """Add and commit a list of mapped objects."""
    if not mapped:
      return

    failures = ResilientAddManyAndCommit(self.db, mapped)
    if failures:
      self.ctx.Error("Logger failed to commit %d objects", len(failures))
    self.error_count += len(failures)

  def _Flush(self):
    """Flush the buffer."""
    if not self._buffer:
      return

    with self.ctx.Profile(
      self.log_level,
      f"Committed {self.buffer_length} rows "
      f"({humanize.BinaryPrefix(self.buffer_size, 'B')}) to {self.db.url}",
    ), self.db.Session() as session:
      # Iterate through the buffer and handle any lambda ops.
      start_i, end_i = 0, 0
      for end_i, item in enumerate(self._buffer):
        if isinstance(item, BufferedDatabaseWriter.LambdaOp):
          # If we have a lambda op, we flush the contents of the current buffer,
          # then execute the op and continue.
          self._AddMapped(self._buffer[start_i:end_i])
          self._buffer[end_i](session)
          session.commit()
          start_i = end_i + 1
      # Add any remaining mapped objects from the buffer.
      self._AddMapped(self._buffer[start_i:])

      self._buffer = []
      self._last_flush = time.time()
      self.buffer_size = 0
      self.flush_count += 1
