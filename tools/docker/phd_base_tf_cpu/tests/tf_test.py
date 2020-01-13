"""Test that Tensorflow works."""
import sys

from labm8.py import test

FLAGS = test.FLAGS


def test_import_tensorflow():
  print("Python executable:", sys.executable)
  print("Python version:", sys.version)

  from third_party.py.tensorflow import tf as tensorflow

  print("Tensorflow:", tensorflow.__file__)
  print("Tensorflow version:", tensorflow.VERSION)


def test_tensorflow_session():
  from third_party.py.tensorflow import tf

  a = tf.constant(1)
  b = tf.constant(2)
  c = a + b
  with tf.compat.v1.Session() as sess:
    assert sess.run(c) == 3


if __name__ == "__main__":
  test.Main()
