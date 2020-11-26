import sys
from pathlib import Path

from absl import app, flags

from programl import PROGRAML_VERSION

flags.DEFINE_boolean(
    "version",
    False,
    "Print the version information and exit.",
)
FLAGS = flags.FLAGS


def init_app(argv, unrecognized_flags_okay: bool = False):
    """Initialize a commandline app.

    Usage:

        from absl import app
        from programl.util.py.init_app import init_app

        def main(argv):
            init_app(argv)
            print("Hello, world")

        app.run(main)

    Args:
        argv: The commandline arguments after parsing with absl.flags.
        unrecognized_flags_okay: If `True`, ignore unrecognized flags.

    Raises:
        UsageError: If there are unknown flags and `unrecognized_flags_okay` is not `True`.
    """
    if len(argv) != 1 and not unrecognized_flags_okay:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")

    if FLAGS.version:
        name = Path(argv[0]).stem
        print(f"{name} version {PROGRAML_VERSION}")
        sys.exit(0)
