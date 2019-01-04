"""Support for unzipping vocabulary files at runtime."""

import pathlib
import pickle
import re
import shutil
import tempfile
import typing
import zipfile

from absl import flags

from deeplearning.ncc import inst2vec_pb2
from deeplearning.ncc import rgx_utils
from deeplearning.ncc.inst2vec import inst2vec_preprocess as i2v_prep
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

  def EncodeLlvmBytecode(
      self, llvm_bytecode: str,
      options: inst2vec_pb2.EncodeBytecodeOptions = inst2vec_pb2.EncodeBytecodeOptions()
  ) -> inst2vec_pb2.EncodeBytecodeResult:
    """Encode an LLVM bytecode using the given vocabulary.

    Args:
      llvm_bytecode: LLVM bytecode as a string.
      vocab: The vocabulary to use for encoding.
    :return:
    """
    result = inst2vec_pb2.EncodeBytecodeResult(input_bytecode=llvm_bytecode)

    def _MaybeSetUnknownStatement(stmt: str) -> None:
      if options.set_unknown_statements:
        result.unknown_statements.extend([stmt])

    def _MaybeSetBytecodeAfterPreprocessing(
        preprocessed_lines: typing.List[str]) -> None:
      if options.set_bytecode_after_preprocessing:
        result.bytecode_after_preprocessing = '\n'.join(preprocessed_lines)

    llvm_bytecode_lines = llvm_bytecode.split('\n')

    # Source code pre-processing.
    # TODO(cec): Merge i2v_prep.preprocess() and PreprocessLlvmBytecode().
    preprocessed_data, _ = i2v_prep.preprocess(
        [llvm_bytecode_lines])
    llvm_bytecode_lines = preprocessed_data[0]
    llvm_bytecode_lines = PreprocessLlvmBytecode(llvm_bytecode_lines)
    _MaybeSetBytecodeAfterPreprocessing(llvm_bytecode_lines)

    stmt_indexed = []  # Construct indexed sequence

    for i, stmt in enumerate(llvm_bytecode_lines):
      # check whether this is a label, in which case we ignore it
      if re.match(r'((?:<label>:)?(<LABEL>):|; <label>:<LABEL>)', stmt):
        continue

      # check whether this is an unknown
      if stmt in self.cutoff_stmts:
        _MaybeSetUnknownStatement(stmt)
        stmt = rgx_utils.unknown_token

      # lookup and add to list
      if stmt not in self.dictionary.keys():
        _MaybeSetUnknownStatement(stmt)
        stmt = rgx_utils.unknown_token

      stmt_indexed.append(self.dictionary[stmt])

    result.encoded.extend(stmt_indexed)

    return result


def GetStructDict(bytecode_lines: typing.List[str]):
  # Construct a dictionary ["structure name", "corresponding literal structure"]
  _, struct_dict = i2v_prep.construct_struct_types_dictionary_for_file(
      bytecode_lines)

  # If the dictionary is empty
  if not struct_dict:
    for line in bytecode_lines:
      if re.match(
          rgx_utils.struct_name + ' = type (<?\{ .* \}|opaque|{})', line):
        # "Structures' dictionary is empty for file containing type definitions"
        # + data[0] + '\n' + data[1] + '\n' + data + '\n'
        assert False

  return struct_dict


def PreprocessLlvmBytecode(lines: typing.List[str]):
  """Simplify lines of code by stripping them from their identifiers,
  unnamed values, etc. so that LLVM IR statements can be abstracted from them.
  """
  struct_dict = GetStructDict(lines)

  # Remove all "... = type {..." statements since we don't need them anymore
  lines = [
    stmt for stmt in lines if not re.match('.* = type ', stmt)]

  for i in range(len(lines)):

    # Inline structure types in the src file.
    possible_structs = re.findall('(' + rgx_utils.struct_name + ')', lines[i])
    if possible_structs:
      for possible_struct in possible_structs:
        if (possible_struct in struct_dict and
            not re.match(possible_struct + r'\d* = ', lines[i])):
          # Replace them by their value in dictionary.
          lines[i] = re.sub(
              re.escape(possible_struct) + rgx_utils.struct_lookahead,
              struct_dict[possible_struct], lines[i])

    # Replace all local identifiers (%## expressions) by "<%ID>".
    lines[i] = re.sub(rgx_utils.local_id, "<%ID>", lines[i])
    # Replace all local identifiers (@## expressions) by "<@ID>".
    lines[i] = re.sub(rgx_utils.global_id, "<@ID>", lines[i])

    # Replace label declarations by token '<LABEL>'.
    if re.match(r'; <label>:\d+:?(\s+; preds = )?', lines[i]):
      lines[i] = re.sub(r":\d+", ":<LABEL>", lines[i])
      lines[i] = re.sub("<%ID>", "<LABEL>", lines[i])
    elif re.match(rgx_utils.local_id_no_perc + r':(\s+; preds = )?', lines[i]):
      lines[i] = re.sub(rgx_utils.local_id_no_perc + ':', "<LABEL>:", lines[i])
      lines[i] = re.sub("<%ID>", "<LABEL>", lines[i])

    if '; preds = ' in lines[i]:
      s = lines[i].split('  ')
      if s[-1][0] == ' ':
        lines[i] = s[0] + s[-1]
      else:
        lines[i] = s[0] + ' ' + s[-1]

    # Replace unnamed_values with abstract tokens. Abstract tokens map:
    #   integers: <INT>
    #   floating points: <FLOAT> (whether in decimal or hexadecimal notation)
    #   string: <STRING>
    #
    # Hexadecimal notation.
    lines[i] = re.sub(
        r' ' + rgx_utils.immediate_value_float_hexa, " <FLOAT>", lines[i])
    # Decimal / scientific notation.
    lines[i] = re.sub(
        r' ' + rgx_utils.immediate_value_float_sci, " <FLOAT>", lines[i])
    if (re.match("<%ID> = extractelement", lines[i]) is None and
        re.match("<%ID> = extractvalue", lines[i]) is None and
        re.match("<%ID> = insertelement", lines[i]) is None and
        re.match("<%ID> = insertvalue", lines[i]) is None):
      lines[i] = re.sub(
          r'(?<!align)(?<!\[) ' + rgx_utils.immediate_value_int,
          " <INT>", lines[i])

    lines[i] = re.sub(rgx_utils.immediate_value_string, " <STRING>", lines[i])

    # Replace the index type in expressions containing "extractelement" or
    # "insertelement" by token <TYP>.
    if (re.match("<%ID> = extractelement", lines[i]) is not None or
        re.match("<%ID> = insertelement", lines[i]) is not None):
      lines[i] = re.sub(r'i\d+ ', '<TYP> ', lines[i])

  return lines
