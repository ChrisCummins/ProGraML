"""Bazel-compatible wrapper around a Jupyter notebook server.

This script launches a Jupyter notebook server. It never terminates. It should
be executed using the run_notebooks.sh helper script so that it can open and
save files within the project source tree.
"""
import typing

from notebook import notebookapp

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_integer('port', 8888, 'The port to launch the Jupyter server on.')
app.DEFINE_boolean(
    'with_colaboratory', True,
    'Allow server to be used as a local runtime for Google '
    'Colaboratory notebooks.')
app.DEFINE_boolean('browser', False, 'Open a web browser upon server launch.')
app.DEFINE_boolean(
    'generate_jupyter_config_file', False,
    'Generate a default config file and write it to '
    '~/.jupyter/jupyter_notebook_config.py. If this file '
    'already exists, you are prompted to overwrite.')


def main(argv: typing.List[str]):
  """Main entry point."""
  options = [
      f'--NotebookApp.port={FLAGS.port}',
      f'--NotebookApp.open_browser={FLAGS.browser}',
  ]

  # Optionally enable server to be used as a local runtime for Google
  # Colaboratory.
  if FLAGS.with_colaboratory:
    options += [
        "--NotebookApp.nbserver_extensions={'jupyter_http_over_ws':True}",
        "--NotebookApp.allow_origin='https://colab.research.google.com'",
        '--NotebookApp.port_retries=0',
    ]

  if FLAGS.generate_jupyter_config_file:
    options += ['--JupyterApp.generate_config=True']

  # Append any arguments not parsed by absl.
  options += argv[1:]

  app.Log(1, 'Starting Jupyter notebook server with options: %s', options)
  notebookapp.main(options)


if __name__ == '__main__':
  app.RunWithArgs(main)
