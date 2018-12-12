"""Bazel-compatible wrapper around a Jupyter notebook server.

This script launches a Jupyter notebook server. It never terminates. See
//notebooks/BUILD for usage instructions.
"""
from absl import app
from absl import flags
from notebook import notebookapp


FLAGS = flags.FLAGS


def main(argv):
  """Main entry point."""
  # TODO(cec): Pass start up options argv to notebookapp.main().
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  notebookapp.main()


if __name__ == '__main__':
  app.run(main)
