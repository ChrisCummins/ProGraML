"""Support for unzipping vocabulary files at runtime."""

import pathlib
import pickle
import shutil
import tempfile
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

  def __init__(self, compressed_path: str):
    self._uncompressed_path = None
    if not compressed_path:
      raise ValueError("Path must be a string")
    self._compressed_path = pathlib.Path(compressed_path)
    if not self._compressed_path.is_file():
      raise FileNotFoundError(f'File not found: {self._compressed_path}')

  @property
  def dictionary_pickle(self) -> pathlib.Path:
    return self.uncompressed_path / 'vocabulary' / 'dic_pickle'

  @property
  def cutoff_stmts_pickle(self) -> pathlib.Path:
    return self.uncompressed_path / 'vocabulary' / 'cutoff_stmts_pickle'

  @property
  def uncompressed_path(self) -> pathlib.Path:
    if not self._uncompressed_path:
      raise TypeError("VocabularyZipFile must be used as a context manager")
    return self._uncompressed_path

  @decorators.memoized_property
  def unknown_token_index(self) -> int:
    """Get the numeric vocabulary index of the "unknown" token.

    The unknown token is used to mark tokens which fall out-of-vocabulary. It
    can also be used as a pad character for sequences.
    """
    with open(self.dictionary_pickle, 'rb') as f:
      dictionary = pickle.load(f)
    unknown_token_index = dictionary[rgx_utils.unknown_token]
    del dictionary
    return unknown_token_index

  def __enter__(self):
    self._uncompressed_path = pathlib.Path(tempfile.mkdtemp(prefix='phd_'))
    with zipfile.ZipFile(str(self._compressed_path)) as f:
      f.extractall(path=str(self._uncompressed_path))
    return self

  def __exit__(self, *args):
    shutil.rmtree(self._uncompressed_path)
