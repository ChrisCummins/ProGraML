"""Test that Tensorflow works."""
import importlib
import os
import sys

from labm8.py import test

FLAGS = test.FLAGS


def test_import_tensorflow():
  print("Python executable:", sys.executable)
  print("Python version:", sys.version)

  import site

  print("Site packages:", site.getsitepackages())

  import pathlib

  assert pathlib.Path(
    "/usr/local/lib/python3.7/site-packages/tensorflow"
  ).is_file()
  print(
    list(
      pathlib.Path(
        "/usr/local/lib/python3.7/site-packages/tensorflow"
      ).iterdir()
    )
  )

  try:
    import tensorflow
  except (ImportError, ModuleNotFoundError):
    tensorflow = importlib.import_module(
      "tensorflow",
      "/usr/local/lib/python3.7/site-packages/tensorflow/__init__.py",
    )

  print("Tensorflow:", tensorflow.__file__)
  print("Tensorflow version:", tensorflow.VERSION)


def test_tensorflow_session():
  import tensorflow as tf

  a = tf.constant(1)
  b = tf.constant(2)
  c = a + b
  with tf.compat.v1.Session() as sess:
    assert sess.run(c) == 3


if __name__ == "__main__":
  test.Main()
