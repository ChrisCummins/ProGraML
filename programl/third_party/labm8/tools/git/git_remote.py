"""This module defines utilities for working with git remotes."""
import contextlib
import time

import git
from labm8.py import app

FLAGS = app.FLAGS


@contextlib.contextmanager
def GitRemote(repo: git.Repo, remote_url: str) -> str:
    """Add a git remote and return it's name.

    Args:
      repo: The repo to add the remote to.
      remote_url: The URL of the remote to add.

    Returns:
      The name of the remote.
    """
    remote_name = f"tmp_remote_{int(time.time() * 1e6)}"
    repo.git.remote("add", remote_name, remote_url)
    try:
        yield remote_name
    finally:
        repo.git.remote("remove", remote_name)
