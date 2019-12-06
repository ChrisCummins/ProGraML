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
"""JSON parser which supports comments.
"""
import json
import re
import typing

from labm8.py import fs

# A type alias for annotating methods which take or return JSON.
JSON = typing.Union[typing.List[typing.Any], typing.Dict[str, typing.Any]]


def format_json(data, default=None):
  """
  Pretty print JSON.

  Arguments:
      data (dict): JSON blob.

  Returns:
      str: Formatted JSON
  """
  return json.dumps(
    data, sort_keys=True, indent=2, separators=(",", ": "), default=default
  )


def read_file(*components, **kwargs):
  """
  Load a JSON data blob.

  Arguments:
      path (str): Path to file.
      must_exist (bool, otional): If False, return empty dict if file does
          not exist.

  Returns:
      array or dict: JSON data.

  Raises:
      File404: If path does not exist, and must_exist is True.
      InvalidFile: If JSON is malformed.
  """
  must_exist = kwargs.get("must_exist", True)

  if must_exist:
    path = fs.must_exist(*components)
  else:
    path = fs.path(*components)

  try:
    with open(path) as infile:
      return loads(infile.read())
  except ValueError as e:
    raise ValueError(
      "malformed JSON file '{path}'. Message from parser: {err}".format(
        path=fs.basename(path), err=str(e),
      ),
    )
  except IOError as e:
    if not must_exist:
      return {}
    else:
      return e


def write_file(path, data, format=True):
  """
  Write JSON data to file.

  Arguments:
      path (str): Destination.
      data (dict or list): JSON serializable data.
      format (bool, optional): Pretty-print JSON data.
  """
  if format:
    fs.Write(path, format_json(data).encode("utf-8"))
  else:
    fs.Write(path, json.dumps(data).encode("utf-8"))


def loads(text, **kwargs):
  """
  Deserialize `text` (a `str` or `unicode` instance containing a JSON
  document with Python or JavaScript like comments) to a Python object.

  Supported comment types: `// comment` and `# comment`.

  Taken from `commentjson <https://github.com/vaidik/commentjson>`_, written
  by `Vaidik Kapoor <https://github.com/vaidik>`_.

  Copyright (c) 2014 Vaidik Kapoor, MIT license.

  Arguments:
      text (str): serialized JSON string with or without comments.
      **kwargs (optional): all the arguments that
          `json.loads <http://docs.python.org/2/library/json.html#json.loads>`_
          accepts.

  Returns:
      `dict` or `list`: Decoded JSON.
  """
  regex = r"\s*(#|\/{2}).*$"
  regex_inline = (
    r"(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$"
  )
  lines = text.split("\n")

  for index, line in enumerate(lines):
    if re.search(regex, line):
      if re.search(r"^" + regex, line, re.IGNORECASE):
        lines[index] = ""
      elif re.search(regex_inline, line):
        lines[index] = re.sub(regex_inline, r"\1", line)

  return json.loads("\n".join(lines), **kwargs)


def JsonSerializable(val):
  """Return a JSON-serializable version of the object.

  If the object is natively JSON-serializable, then the object is return
  unmodified. Else the string representation of the object is returned.
  """
  try:
    json.dumps(val)
    return val
  except TypeError:
    return str(val)
