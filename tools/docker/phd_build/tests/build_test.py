"""Test bazel build."""
import subprocess

from labm8.py import test

FLAGS = test.FLAGS


def test_bazel():
  """Test that bazel is installed."""
  subprocess.check_call(["bazel", "version"])


if __name__ == "__main__":
  test.Main()
