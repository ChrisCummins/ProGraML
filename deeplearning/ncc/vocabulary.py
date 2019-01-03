"""Support for unzipping vocabulary files at runtime."""

import pathlib
import pickle
import shutil
import tempfile
import typing
import zipfile

from absl import flags

from deeplearning.ncc import rgx_utils
from labm8 import decorators


FLAGS = flags.FLAGS


class VocabularyZipFile(object):
  """A compressed vocabulary file.

  Provides access to the unzipped vocabulary files when used as a context
  manager by extracting the zip contents to a temporary directory.
  """

  def __init__(self, compressed_path: typing.Union[str, pathlib.Path]):
    self._uncompressed_path_val = None
    if not compressed_path:
      raise ValueError("Path must be a string")
    self._compressed_path = pathlib.Path(compressed_path)
    if not self._compressed_path.is_file():
      raise FileNotFoundError(f'File not found: {self._compressed_path}')

  # Public properties.

  @decorators.memoized_property
  def dictionary(self) -> typing.Dict[str, int]:
    """Return the vocabulary dictionary."""
    with open(self._dictionary_pickle, 'rb') as f:
      return pickle.load(f)

  @decorators.memoized_property
  def cutoff_stmts(self) -> typing.Set[str]:
    """Return the vocabulary cut off statements."""
    with open(self._cutoff_stmts_pickle, 'rb') as f:
      return set(pickle.load(f))

  @property
  def unknown_token_index(self) -> int:
    """Get the numeric vocabulary index of the "unknown" token.

    The unknown token is used to mark tokens which fall out-of-vocabulary. It
    can also be used as a pad character for sequences.
    """
    unknown_token_index = self.dictionary[rgx_utils.unknown_token]
    return unknown_token_index

  # Private properties.

  @property
  def _dictionary_pickle(self) -> pathlib.Path:
    return self._uncompressed_path / 'vocabulary' / 'dic_pickle'

  @property
  def _cutoff_stmts_pickle(self) -> pathlib.Path:
    return self._uncompressed_path / 'vocabulary' / 'cutoff_stmts_pickle'

  @property
  def _uncompressed_path(self) -> pathlib.Path:
    if not self._uncompressed_path_val:
      raise TypeError("VocabularyZipFile must be used as a context manager")
    return self._uncompressed_path_val

  def __enter__(self):
    self._uncompressed_path_val = pathlib.Path(tempfile.mkdtemp(prefix='phd_'))
    with zipfile.ZipFile(str(self._compressed_path)) as f:
      f.extractall(path=str(self._uncompressed_path_val))
    return self

  def __exit__(self, *args):
    shutil.rmtree(self._uncompressed_path_val)
