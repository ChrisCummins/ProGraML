"""Test that javac is installed."""
import subprocess

from labm8.py import test

FLAGS = test.FLAGS


def test_javac_help():
  """Test that javac is installed."""
  subprocess.check_call(["/usr/bin/javac", "--help"])


if __name__ == "__main__":
  test.Main()
