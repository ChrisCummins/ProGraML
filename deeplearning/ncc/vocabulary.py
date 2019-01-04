"""Support for unzipping vocabulary files at runtime."""

import collections
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

    # Source code transformation: simple pre-processing
    preprocessed_data, functions_declared_in_files = i2v_prep.preprocess(
        [llvm_bytecode_lines])
    preprocessed_data_with_structure_def = [llvm_bytecode_lines]

    # IR processing (inline structures, abstract statements)

    # Source code transformation: inline structure types
    processed_data, _ = inline_struct_types_txt(
        preprocessed_data, preprocessed_data_with_structure_def)

    # Source code transformation: identifier processing (abstract statements)
    processed_data = abstract_statements_from_identifiers_txt(processed_data)

    file = processed_data[0]

    _MaybeSetBytecodeAfterPreprocessing(file)

    stmt_indexed = []  # Construct indexed sequence

    for i, stmt in enumerate(file):
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


# TODO(cec): Tidy up and test everything below


def inline_struct_types_in_file(data, dic):
  """
  Inline structure types in the whole file
  :param data: list of strings representing the content of one file
  :param dic: dictionary ["structure name", "corresponding literal structure"]
  :return: modified data
  """
  # Remove all "... = type {..." statements since we don't need them anymore
  data = [stmt for stmt in data if not re.match('.* = type ', stmt)]

  # Inline the named structures throughout the file
  for i in range(len(data)):

    possible_struct = re.findall('(' + rgx_utils.struct_name + ')', data[i])
    if len(possible_struct) > 0:
      for s in possible_struct:
        if s in dic and not re.match(s + r'\d* = ', data[i]):
          # Replace them by their value in dictionary
          data[i] = re.sub(re.escape(s) + rgx_utils.struct_lookahead, dic[s],
                           data[i])

  return data


def inline_struct_types_txt(data, data_with_structure_def):
  """
  Inline structure types so that the code has no more named structures but only explicit aggregate types
  And construct a dictionary of these named structures
  :param data: input data as a list of files where each file is a list of strings
  :return: data: modified input data
           dictio: list of dictionaries corresponding to source files,
                   where each dictionary has entries ["structure name", "corresponding literal structure"]
  """
  print('\tConstructing dictionary of structures and inlining structures...')
  dictio = collections.defaultdict(list)

  # Loop on all files in the dataset
  for i in range(len(data)):
    # Construct a dictionary ["structure name", "corresponding literal structure"]
    data_with_structure_def[i], dict_temp = \
      i2v_prep.construct_struct_types_dictionary_for_file(
          data_with_structure_def[i])

    # If the dictionary is empty
    if not dict_temp:
      found_type = False
      for l in data[i]:
        if re.match(
            rgx_utils.struct_name + ' = type (<?\{ .* \}|opaque|{})', l):
          found_type = True
          break
      assert not found_type, "Structures' dictionary is empty for file containing type definitions: \n" + \
                             data[i][0] + '\n' + data[i][1] + '\n' + data[
                               i] + '\n'

    # Use the constructed dictionary to substitute named structures
    # by their corresponding literal structure throughout the program
    data[i] = inline_struct_types_in_file(data[i], dict_temp)

    # Add the entries of the dictionary to the big dictionary
    for k, v in dict_temp.items():
      dictio[k].append(v)

  return data, dictio


def abstract_statements_from_identifiers_txt(data):
  """
  Simplify lines of code by stripping them from their identifiers,
  unnamed values, etc. so that LLVM IR statements can be abstracted from them
  :param data: input data as a list of files where each file is a list of strings
  :return: modified input data
  """
  data = remove_local_identifiers(data)
  data = remove_global_identifiers(data)
  data = remove_labels(data)
  data = replace_unnamed_values(data)
  data = remove_index_types(data)

  return data


def remove_local_identifiers(data):
  """
  Replace all local identifiers (%## expressions) by "<%ID>"
  :param data: input data as a list of files where each file is a list of strings
  :return: modified input data
  """
  print('\tRemoving local identifiers ...')
  for i in range(len(data)):
    for j in range(len(data[i])):
      data[i][j] = re.sub(rgx_utils.local_id, "<%ID>", data[i][j])

  return data


def remove_global_identifiers(data):
  """
  Replace all local identifiers (@## expressions) by "<@ID>"
  :param data: input data as a list of files where each file is a list of strings
  :return: modified input data
  """
  print('\tRemoving global identifiers ...')
  for i in range(len(data)):
    for j in range(len(data[i])):
      data[i][j] = re.sub(rgx_utils.global_id, "<@ID>", data[i][j])

  return data


def remove_labels(data):
  """Replace label declarations by token '<LABEL>'.

  Args:
    data: A list of list of strings to modify.

  Returns:
     The list of list of strings.
  """
  for i in range(len(data)):
    for j in range(len(data[i])):
      if re.match(r'; <label>:\d+:?(\s+; preds = )?', data[i][j]):
        data[i][j] = re.sub(r":\d+", ":<LABEL>", data[i][j])
        data[i][j] = re.sub("<%ID>", "<LABEL>", data[i][j])
      elif re.match(rgx_utils.local_id_no_perc + r':(\s+; preds = )?',
                    data[i][j]):
        data[i][j] = re.sub(rgx_utils.local_id_no_perc + ':', "<LABEL>:",
                            data[i][j])
        data[i][j] = re.sub("<%ID>", "<LABEL>", data[i][j])
      if '; preds = ' in data[i][j]:
        s = data[i][j].split('  ')
        if s[-1][0] == ' ':
          data[i][j] = s[0] + s[-1]
        else:
          data[i][j] = s[0] + ' ' + s[-1]

  return data


def replace_unnamed_values(data):
  """Replace unnamed_values with abstract tokens.

  Abstract tokens map:
    integers: <INT>
    floating points: <FLOAT> (whether in decimal or hexadecimal notation)
    string: <STRING>

  Args:
    data: A list of list of strings to modify.

  Returns:
     The list of list of strings.
  """
  for i in range(len(data)):
    for j in range(len(data[i])):
      # Hexadecimal notation.
      data[i][j] = re.sub(
          r' ' + rgx_utils.immediate_value_float_hexa, " <FLOAT>", data[i][j])
      # Decimal / scientific notation.
      data[i][j] = re.sub(
          r' ' + rgx_utils.immediate_value_float_sci, " <FLOAT>", data[i][j])
      if (re.match("<%ID> = extractelement", data[i][j]) is None and
          re.match("<%ID> = extractvalue", data[i][j]) is None and
          re.match("<%ID> = insertelement", data[i][j]) is None and
          re.match("<%ID> = insertvalue", data[i][j]) is None):
        data[i][j] = re.sub(
            r'(?<!align)(?<!\[) ' + rgx_utils.immediate_value_int,
            " <INT>", data[i][j])

      data[i][j] = re.sub(rgx_utils.immediate_value_string, " <STRING>",
                          data[i][j])

  return data


def remove_index_types(data):
  """Replace the index type in expressions containing "extractelement" or
  "insertelement" by token <TYP>.

  Args:
    data: A list of list of strings to modify.

  Returns:
     The list of list of strings.
  """
  for i in range(len(data)):
    for j in range(len(data[i])):
      if (re.match("<%ID> = extractelement", data[i][j]) is not None or
          re.match("<%ID> = insertelement", data[i][j]) is not None):
        data[i][j] = re.sub(r'i\d+ ', '<TYP> ', data[i][j])

  return data
