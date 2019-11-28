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
"""Hashing and cryptography utils.
"""
import hashlib
import pathlib
import typing


def _checksum(hash_fn, data):
  return hash_fn(data).hexdigest()


def _checksum_str(hash_fn, string, encoding='utf-8'):
  return _checksum(hash_fn, string.encode(encoding))


def _checksum_list(hash_fn, *elems):
  string = ''.join(sorted(str(x) for x in elems))
  return _checksum_str(hash_fn, string)


def _checksum_file(hash_fn, path: typing.Union[str, pathlib.Path]):
  with open(path, 'rb') as infile:
    ret = _checksum(hash_fn, infile.read())
  return ret


def sha1(data):
  """
  Return the sha1 of "data".

  Arguments:
      data (bytes): Data.

  Returns:
      str: Hex encoded.
  """
  return _checksum(hashlib.sha1, data)


def sha1_str(string, encoding='utf-8'):
  """
  Return the sha1 of string "data".

  Arguments:
      string: String.

  Returns:
      str: Hex encoded.
  """
  return _checksum_str(hashlib.sha1, string, encoding=encoding)


def sha1_list(*elems):
  """
  Return the sha1 of all elements of a list.

  Arguments:
      *elems: List of stringifiable data.

  Returns:
      str: Hex encoded.
  """
  return _checksum_list(hashlib.sha1, *elems)


def sha1_file(path: typing.Union[str, pathlib.Path]):
  """
  Return the sha1 of file at "path".

  Arguments:
      path (str): Path to file

  Returns:
      str: Hex encoded.
  """
  return _checksum_file(hashlib.sha1, path)


def md5(data):
  """
  Return the md5 of "data".

  Arguments:
      data (bytes): Data.

  Returns:
      str: Hex encoded.
  """
  return _checksum(hashlib.md5, data)


def md5_str(string, encoding='utf-8'):
  """
  Return the md5 of string "data".

  Arguments:
      string: String.

  Returns:
      str: Hex encoded.
  """
  return _checksum_str(hashlib.md5, string, encoding=encoding)


def md5_list(*elems):
  """
  Return the md5 of all elements of a list.

  Arguments:
      *elems: List of stringifiable data.

  Returns:
      str: Hex encoded.
  """
  return _checksum_list(hashlib.md5, *elems)


def md5_file(path: typing.Union[str, pathlib.Path]):
  """
  Return the md5 of file at "path".

  Arguments:
      path (str): Path to file

  Returns:
      str: Hex encoded.
  """
  return _checksum_file(hashlib.md5, path)


def sha256(data):
  """
  Return the sha256 of "data".

  Arguments:
      data (bytes): Data.

  Returns:
      str: Hex encoded.
  """
  return _checksum(hashlib.sha256, data)


def sha256_str(string, encoding='utf-8'):
  """
  Return the sha256 of string "data".

  Arguments:
      string: String.

  Returns:
      str: Hex encoded.
  """
  return _checksum_str(hashlib.sha256, string, encoding=encoding)


def sha256_list(*elems):
  """
  Return the sha256 of all elements of a list.

  Arguments:
      *elems: List of stringifiable data.

  Returns:
      str: Hex encoded.
  """
  return _checksum_list(hashlib.sha256, *elems)


def sha256_file(path: typing.Union[str, pathlib.Path]):
  """
  Return the sha256 of file at "path".

  Arguments:
      path (str): Path to file

  Returns:
      str: Hex encoded.
  """
  return _checksum_file(hashlib.sha256, path)
