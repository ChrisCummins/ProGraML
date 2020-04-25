"""Import Tensorflow.

This module is a drop-in replacement for regular tensorflow. Replace:

    import tensorflow as tf

with:

    from third_party.py.tensorflow import tf

This wrapper is required to workaround a known bug with packaging Tensorflow
as a pip dependency with bazel. See:
github.com/bazelbuild/rules_python/issues/71
"""
import pathlib
import sys

try:
  # Try importing Tensorflow the vanilla way. This will succeed once
  # github.com/bazelbuild/rules_python/issues/71 is fixed.
  import tensorflow
except (ImportError, ModuleNotFoundError):
  # That failed, so see if there is a version of the package elsewhere on the
  # system that we can force python into loading.
  extra_site_packages = [
    "/usr/local/lib/python3.7/site-packages",
  ]
  for path in extra_site_packages:
    tensorflow_site_package = pathlib.Path(path) / "tensorflow"
    if tensorflow_site_package.is_dir():
      # Add the additional packages location to the python path.
      sys.path.insert(0, path)
      try:
        import tensorflow

        break
      except (ImportError, ModuleNotFoundError):
        pass
      finally:
        # Restore python path.
        del sys.path[0]

# Import Tensorflow into this module's namespace. If the above import attempts
# failed, this will raise an error.
from tensorflow import *

# Spoof that we've imported the package generically.
__file__ = tensorflow.__file__
