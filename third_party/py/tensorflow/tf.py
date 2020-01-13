"""Import Tensorflow.

This module is a drop-in replacement for regular tensorflow. Replace:

    import tensorflow as tf

with:

    from third_party.py.tensorflow import tf

This wrapper is required to workaround a known bug with packaging Tensorflow
as a pip dependency with bazel. See:
github.com/bazelbuild/rules_python/issues/71
"""
import importlib
import pathlib
import sys

try:
  # Try importing Tensorflow the vanilla way. This will succeed once
  # github.com/bazelbuild/rules_python/issues/71 is fixed.
  import tensorflow
except (ImportError, ModuleNotFoundError):
  # That failed, so see if there is a system install of Tensorflow that we
  # can trick python into importing. This should succeed in the
  # chriscummins/phd_base_tf_cpu docker image.
  PYTHON_SITE_PACKAGES = pathlib.Path("/usr/local/lib/python3.7/site-packages")

  try:
    if pathlib.Path(PYTHON_SITE_PACKAGES / "tensorflow").is_dir():
      sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

      import tensorflow
    else:
      raise ModuleNotFoundError
  except (ImportError, ModuleNotFoundError):
    # That failed, so a final hail mary let's try importing the module directly.
    tensorflow = importlib.import_module(
      "tensorflow", PYTHON_SITE_PACKAGES / "tensorflow",
    )

# Import Tensorflow into this module's namespace.
from tensorflow import *

# Pretend that we've imported the regular Tensorflow.
__file__ = tensorflow.__file__
