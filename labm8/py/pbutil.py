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
"""Utility code for working with Protocol Buffers."""
import collections
import gzip
import json
import pathlib
import subprocess
import typing

import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format

# A type alias for annotating methods which take or return protocol buffers.
ProtocolBuffer = typing.Any

# A type alias for protocol enum fields.
Enum = int


class ProtoValueError(ValueError):
  """Raised in case of a value error from a proto."""

  pass


class EncodeError(ProtoValueError):
  """Raised in case of error encoding a proto."""

  pass


class DecodeError(ProtoValueError):
  """Raised in case of error decoding a proto."""

  pass


class ProtoWorkerTimeoutError(subprocess.CalledProcessError):
  """Raised is a protobuf worker binary times out."""

  def __init__(
    self, cmd: typing.List[str], timeout_seconds: int, returncode: int,
  ):
    self.cmd = cmd
    self.timeout_seconds = timeout_seconds
    # subprocess.CalledProcessError.str() requires a returncode attribute.
    self.returncode = returncode

  def __repr__(self) -> str:
    return (
      f"Proto worker timeout after {self.timeout_seconds} "
      f"seconds: {' '.join(self.cmd)}"
    )


def FromString(
  string: str, message: ProtocolBuffer, uninitialized_okay: bool = False,
) -> ProtocolBuffer:
  """Read a text format protocol buffer from a string.

  Args:
    string: A text format protocol buffer.
    message: A message instance to read into.
    uninitialized_okay: If True, do not require that decoded messages be
      initialized. If False, DecodeError is raised.

  Returns:
    The parsed message (same as the message argument).

  Raises:
    DecodeError: If the file cannot be decoded to the given message type, or if
      after decoding, the message is not initialized and uninitialized_okay is
      False.
  """
  try:
    google.protobuf.text_format.Merge(string, message)
  except google.protobuf.text_format.ParseError as e:
    raise DecodeError(e)

  if not uninitialized_okay and not message.IsInitialized():
    raise DecodeError(f"Required fields not set")

  return message


def FromFile(
  path: pathlib.Path,
  message: ProtocolBuffer,
  assume_filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
  uninitialized_okay: bool = False,
) -> ProtocolBuffer:
  """Read a protocol buffer from a file.

  This method uses attempts to guess the encoding from the path suffix,
  supporting binary, text, and json formatted messages. The mapping of suffixes
  to formatting is, in order:
      *.txt.gz: Gzipped text.
      *.txt: Text.
      *.pbtxt.gz: Gzipped text.
      *.pbtxt: Text.
      *.json.gz: Gzipped JSON.
      *.json: JSON.
      *.gz: Gzipped encoded string.
      *: Encoded string.

  Args:
    path: Path to the proto file.
    message: A message instance to read into.
    assume_filename: For the purpose of determining the encoding from the file
      extension, use this name rather than the true path.
    uninitialized_okay: If True, do not require that decoded messages be
      initialized. If False, DecodeError is raised.

  Returns:
    The parsed message (same as the message argument).

  Raises:
    FileNotFoundError: If the path does not exist.
    IsADirectoryError: If the path is a directory.
    DecodeError: If the file cannot be decoded to the given message type, or if
      after decoding, the message is not initialized and uninitialized_okay is
      False.
  """
  if not path.is_file():
    if path.is_dir():
      raise IsADirectoryError(f"Path is a directory: '{path}'")
    else:
      raise FileNotFoundError(f"File not found: '{path}'")

  suffixes = (
    pathlib.Path(assume_filename,).suffixes
    if assume_filename
    else path.suffixes
  )
  if suffixes and suffixes[-1] == ".gz":
    suffixes.pop()
    open_function = gzip.open
  else:
    open_function = open

  suffix = suffixes[-1] if suffixes else ""
  try:
    with open_function(path, "rb") as f:
      if suffix == ".txt" or suffix == ".pbtxt":
        # Allow uninitialized fields here because we will catch the error later,
        # allowing us to report the path of the proto.
        FromString(f.read().decode("utf-8"), message, uninitialized_okay=True)
      elif suffix == ".json":
        google.protobuf.json_format.Parse(f.read(), message)
      else:
        message.ParseFromString(f.read())
  except (
    google.protobuf.text_format.ParseError,
    google.protobuf.json_format.ParseError,
  ) as e:
    # The exception raised during parsing depends on the message format. Catch
    # them all under a single DecodeError exception type.
    raise DecodeError(e)

  if not uninitialized_okay and not message.IsInitialized():
    raise DecodeError(f"Required fields not set: '{path}'")

  return message


def ToFile(
  message: ProtocolBuffer,
  path: pathlib.Path,
  exist_ok: bool = True,
  assume_filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
) -> ProtocolBuffer:
  """Write a protocol buffer to a file.

  This method uses attempts to guess the encoding from the path suffix,
  supporting binary, text, and json formatted messages. The mapping of suffixes
  to formatting is, in order:
      *.txt.gz: Gzipped text format.
      *.txt: Text format.
      *.pbtxt.gz: Gzipped text format.
      *.pbtxt: Text format.
      *.json.gz: Gzipped JSON format.
      *.json: JSON format.
      *.gz: Gzipped binary format.
      *: Binary format.

  Args:
    message: A message instance to write to file. The message must be
      initialized, i.e. have all required fields set.
    path: Path to the proto file.
    exist_ok: If True, overwrite existing file.
    assume_filename: For the purpose of determining the encoding from the file
      extension, use this name rather than the true path.

  Returns:
    The parsed message (same as the message argument).

  Raises:
    EncodeError: If the message is not initialized, i.e. it is missing required
      fields.
    FileNotFoundError: If the parent directory of the requested path does not
      exist.
    IsADirectoryError: If the requested path is a directory.
    FileExistsError: If the requested path already exists and exist_ok is False.
  """
  if not exist_ok and path.exists():
    raise FileExistsError(f"Refusing to overwrite {path}")

  # The SerializeToString() method refuses to encode a message which is not
  # initialized, whereas the MessageToString() and MessageToJson() methods do
  # not. This API should be consistent, so we enforce that all formats require
  # the message to be initialized.
  if not message.IsInitialized():
    class_name = type(message).__name__
    raise EncodeError(f"Required fields not set: '{class_name}'")

  suffixes = (
    pathlib.Path(assume_filename,).suffixes
    if assume_filename
    else path.suffixes
  )
  if suffixes and suffixes[-1] == ".gz":
    suffixes.pop()
    open_function = gzip.open
  else:
    open_function = open

  suffix = suffixes[-1] if suffixes else ""
  mode = "wt" if suffix in {".txt", ".pbtxt", ".json"} else "wb"

  with open_function(path, mode) as f:
    if suffix == ".txt" or suffix == ".pbtxt":
      f.write(google.protobuf.text_format.MessageToString(message))
    elif suffix == ".json":
      f.write(
        google.protobuf.json_format.MessageToJson(
          message, preserving_proto_field_name=True,
        ),
      )
    else:
      f.write(message.SerializeToString())

  return message


def ToJson(message: ProtocolBuffer) -> "jsonutil.JSON":
  """Return a JSON encoded representation of a protocol buffer.

  Args:
    message: The message to convert to JSON.

  Returns:
    JSON encoded message.
  """
  return google.protobuf.json_format.MessageToDict(
    message, preserving_proto_field_name=True,
  )


def _TruncatedString(string: str, n: int = 80) -> str:
  """Return the truncated first 'n' characters of a string.

  Args:
    string: The string to truncate.
    n: The maximum length of the string to return.

  Returns:
    The truncated string.
  """
  if len(string) > n:
    return string[: n - 3] + "..."
  else:
    return string


def _TruncateDictionaryStringValues(
  data: "jsonutil.JSON", n: int = 62,
) -> "jsonutil.JSON":
  """Truncate all string values in a nested dictionary.

  Args:
    data: A dictionary.

  Returns:
    The dictionary.
  """
  for key, value in data.items():
    if isinstance(value, collections.Mapping):
      data[key] = _TruncateDictionaryStringValues(data[key])
    elif isinstance(value, str):
      data[key] = _TruncatedString(value, n)
    else:
      data[key] = value
  return data


def PrettyPrintJson(message: ProtocolBuffer, truncate: int = 52) -> str:
  """Return a pretty printed JSON string representation of the message.

  Args:
    message: The message to pretty print.
    truncate: The length to truncate string values. Truncation is disabled if
      this argument is None.

  Returns:
    JSON string.
  """
  data = ToJson(message)
  return json.dumps(
    _TruncateDictionaryStringValues(data) if truncate else data,
    indent=2,
    sort_keys=True,
  )


def RaiseIfNotSet(
  proto: ProtocolBuffer, field: str, err: ValueError,
) -> typing.Any:
  """Check that a proto field is set before returning it.

  Args:
    proto: A message instance.
    field: The name of the field.
    err: The exception class to raise.

  Returns:
    The value of the field.

  Raises:
    ValueError: If the field is not set.
  """
  if not proto.HasField(field):
    raise err(f"datastore field {field} not set")
  elif not getattr(proto, field):
    raise err(f"datastore field {field} not set")
  return getattr(proto, field)


def ProtoIsReadable(
  path: typing.Union[str, pathlib.Path], message: ProtocolBuffer,
) -> bool:
  """Return whether a file is a readable protocol buffer.

  Arguments:
    path: The path of the file to read.
    message: An instance of the message type.

  Returns:
    True if contents of path can be parsed as an instance of message, else
    False.
  """
  try:
    FromFile(pathlib.Path(path), message)
    return True
  except:
    return False


def AssertFieldIsSet(
  proto: ProtocolBuffer, field_name: str, fail_message: str = None,
) -> typing.Optional[typing.Any]:
  """Assert that protocol buffer field is set.

  Args:
    proto: A proto message instance.
    field_name: The name of the field to assert the constraint on.
    fail_message: An optional message to raise the ProtoValueError
      with if the assertion fails. If not provided, a default message is used.

  Returns:
    The value of the field, if the field has a value. Even though a field may
      be set, it may not have a value. For example, if any of a 'oneof' fields
      is set, then this function will return True for the name of the oneof,
      but the return value will be None.

  Raises:
    ValueError: If the requested field does not exist in the proto schema.
    ProtoValueError: If the field is not set.
  """
  if not proto.HasField(field_name):
    proto_class_name = type(proto).__name__
    raise ProtoValueError(
      fail_message or f"Field not set: '{proto_class_name}.{field_name}'",
    )
  return getattr(proto, field_name) if hasattr(proto, field_name) else None


def AssertFieldConstraint(
  proto: ProtocolBuffer,
  field_name: str,
  constraint: typing.Callable[[typing.Any], bool] = lambda x: True,
  fail_message: str = None,
) -> typing.Any:
  """Assert a constraint on the value of a protocol buffer field.

  Args:
    proto: A proto message instance.
    field_name: The name of the field to assert the constraint on.
    constraint: A constraint checking function to call with the value of the
      field. The function must return True if the constraint check passes, else
      False. If no constraint is specified, this callback always returns True.
      This still allows you to use this function to check if a field is set.
    fail_message: An optional message to raise the ProtoValueError
      with if the assertion fails. If not provided, default messages are used.

  Returns:
    The value of the field.

  Raises:
    ValueError: If the requested field does not exist in the proto schema.
    ProtoValueError: If the field is not set, or if the constraint callback
      returns False for the field's value.
  """
  value = AssertFieldIsSet(proto, field_name, fail_message)
  if not constraint(value):
    proto_class_name = type(proto).__name__
    raise ProtoValueError(
      fail_message
      or f"Field fails constraint check: '{proto_class_name}.{field_name}'",
    )
  else:
    return value


def RunProcessMessage(
  cmd: typing.List[str],
  input_proto: ProtocolBuffer,
  timeout_seconds: int = 360,
  env: typing.Dict[str, str] = None,
) -> str:
  """Run the given command, feeding a serialized input proto to stdin.

  Args:
    cmd: The command to execute.
    input_proto: The input message for the command.
    timeout_seconds: The maximum number of seconds to allow the command to run
      for.
    env: A map of environment variables to set, overriding the default
      environment.

  Returns:
    The raw stdout of the command as a byte array.

  Raises;
    ProtoWorkerTimeoutError: If timeout_seconds elapses without the command
      terminating.
    CalledProcessError: If the command terminates with a non-zero returncode.
  """
  # Run the C++ worker process, capturing it's output.
  process = subprocess.Popen(
    ["timeout", "-s9", str(timeout_seconds)] + cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    env=env,
  )
  # Send the input proto to the C++ worker process.
  stdout, _ = process.communicate(input_proto.SerializeToString())

  # TODO: Check signal value, not hardcoded a hardcoded kill signal.
  if process.returncode == -9 or process.returncode == 9:
    raise ProtoWorkerTimeoutError(
      cmd=cmd, timeout_seconds=timeout_seconds, returncode=process.returncode,
    )
  elif process.returncode:
    raise subprocess.CalledProcessError(process.returncode, cmd)

  return stdout


def RunProcessMessageToProto(
  cmd: typing.List[str],
  input_proto: ProtocolBuffer,
  output_proto: ProtocolBuffer,
  timeout_seconds: int = 360,
  env: typing.Dict[str, str] = None,
) -> ProtocolBuffer:
  """Run a command that accepts a protocol buffer as input and produces a
  protocol buffer output.

  Args:
    cmd: The command to execute.
    input_proto: The input message for the command. This is fed to the command's
      stdin as a serialized string.
    output_proto: The output message for the command. The values of this proto
      are set by the stdout of the command.
    timeout_seconds: The maximum number of seconds to allow the command to run
      for.
    env: A map of environment variables to set, overriding the default
      environment.

  Returns:
    The same protocol buffer as output_proto, with the values produced by the
    stdout of the command.

  Raises;
    ProtoWorkerTimeoutError: If timeout_seconds elapses without the command
      terminating.
    CalledProcessError: If the command terminates with a non-zero returncode.
  """
  stdout = RunProcessMessage(
    cmd, input_proto, timeout_seconds=timeout_seconds, env=env,
  )
  output_proto.ParseFromString(stdout)
  return output_proto


def RunProcessMessageInPlace(
  cmd: typing.List[str],
  input_proto: ProtocolBuffer,
  timeout_seconds: int = 360,
  env: typing.Dict[str, str] = None,
) -> ProtocolBuffer:
  """Run the given command, modifying a protocol buffer inplace.

  Args:
    cmd: The command to execute.
    input_proto: The input message for the command. This is fed to the command's
      stdin as a serialized string.
    timeout_seconds: The maximum number of seconds to allow the command to run
      for.
    env: A map of environment variables to set, overriding the default
      environment.

  Returns:
    The same protocol buffer as input_proto, with the values produced by the
    stdout of the command.

  Raises;
    ProtoWorkerTimeoutError: If timeout_seconds elapses without the command
      terminating.
    CalledProcessError: If the command terminates with a non-zero returncode.
  """
  input_proto.ParseFromString(
    RunProcessMessage(
      cmd, input_proto, timeout_seconds=timeout_seconds, env=env,
    ),
  )
  return input_proto


class ProtoBackedMixin(object):
  """A class backed by protocol buffers.

  This mixin provides the abstract interface for classes which support
  serialization of instances to and from protocol buffers.

  Inheriting classes must set the proto_t class attribute, and  implement the
  SetProto() and FromProto() methods.

  Attributes:
    proto_t: The protocol buffer class that backs instances of this class.
  """

  # Inheritinc classes must set this attribute to the Protocol Buffer class.
  proto_t = None

  def SetProto(self, proto: ProtocolBuffer) -> None:
    """Set the fields of a protocol buffer with the values of the instance.

    It is the responsibility of the inheriting class to ensure that all required
    instance variables are recorded as fields in this proto.

    Args:
      proto: A protocol buffer.
    """
    # ABSTRACT METHOD. Inheriting classes must implement!
    raise NotImplementedError(
      f"{type(self).__name__}.SetProto() not implemented",
    )

  @classmethod
  def FromProto(cls, proto: ProtocolBuffer) -> "ProtoBackedMixin":
    """Return an instance of the class from proto.

    It is the responsibility of the inheriting class to ensure that all required
    instance variables are set according to the fields in the proto.

    Args:
      proto: A protocol buffer.

    Returns:
      An instance of the class.
    """
    # ABSTRACT METHOD. Inheriting classes must implement!
    raise NotImplementedError(
      f"{type(self).__name__}.FromProto() not implemented",
    )

  def ToProto(self) -> ProtocolBuffer:
    """Serialize the instance to protocol buffer.

    It is the responsibility of the inheriting class to set the proto_t class
    attribute to the

    Returns:
      A protocol buffer.
    """
    proto = self.proto_t()
    self.SetProto(proto)
    return proto

  @classmethod
  def FromProtoFile(cls, path: pathlib.Path) -> "ProtoBackedMixin":
    """Return an instance of the class from serialized proto file.

    Args:
      path: Path to a proto file.

    Returns:
      An instance.
    """
    return cls.FromProto(FromFile(path, cls.proto_t()))
