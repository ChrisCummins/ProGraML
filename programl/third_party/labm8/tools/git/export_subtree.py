"""This module defines utility code for exporting a subset of a git repo."""
import pathlib
from typing import List, Optional, Set

import git
from labm8.py import app, humanize, progress
from tools.git import git_remote

FLAGS = app.FLAGS


def ExportSubtree(
    source: git.Repo,
    destination: git.Repo,
    files_of_interest: Set[str],
    head_ref: str = "HEAD",
) -> int:
    thread = SubtreeExporter(source, destination, files_of_interest, head_ref=head_ref)
    progress.Run(thread)
    return thread.exported_commit_count


class SubtreeExporter(progress.Progress):
    """A progressable thread for exporting a subset of a repo's commits."""

    def __init__(
        self,
        source: git.Repo,
        destination: git.Repo,
        files_of_interest: Set[str],
        head_ref: str = "HEAD",
    ):
        """Constructor.

        Args:
          source: The source repository to export from.
          destination: The destination repository to export to.
          files_of_interest: The relpaths of the files to export.
          head_ref: The commit to export up to.
        """
        if destination.is_dirty():
            raise OSError(f"Repo `{destination.working_tree_dir}` is dirty")

        self.source = source
        self.destination = destination
        self.commits_in_order = GetCommitsInOrder(source, head_ref=head_ref)
        self.files_of_interest = files_of_interest

        if not self.commits_in_order:
            raise OSError(1, "Nothing to export!")

        # The number of commits that were exported.
        # In the range 0 <= x <= len(self.ctx.n).
        self.exported_commit_count = 0

        super(SubtreeExporter, self).__init__(
            name=f"{pathlib.Path(destination.working_tree_dir).name} export",
            i=0,
            n=len(self.commits_in_order),
            unit="commits",
        )

    def Run(self) -> None:
        """Run the export."""
        # Add the source repo as a remote to the destination repo so that we can move
        # commits from one to to the other.
        with git_remote.GitRemote(
            self.destination, self.source.working_tree_dir
        ) as remote:
            self.destination.remote(remote).fetch()

            for self.ctx.i, commit in enumerate(self.commits_in_order):
                if MaybeExportCommitSubset(
                    commit, self.destination, self.files_of_interest, ctx=self.ctx
                ):
                    self.exported_commit_count += 1

            self.ctx.i = self.ctx.n  # done


def GetCommitsInOrder(
    repo: git.Repo, head_ref: str = "HEAD", tail_ref: Optional[str] = None
) -> List[git.Commit]:
    """Get a list of all commits, in chronological order from old to new.

    Args:
      repo: The repo to list the commits of.
      head_ref: The starting point for iteration, e.g. the commit closest to
        head.
      tail_ref: The end point for iteration, e.g. the commit closest to tail.
        This commit is NOT included in the returned values.

    Returns:
      A list of git.Commit objects.
    """

    def TailCommitIterator():
        stop_commit = repo.commit(tail_ref)
        for commit in repo.iter_commits(head_ref):
            if commit == stop_commit:
                break
            yield commit

    if tail_ref:
        commit_iter = TailCommitIterator()
    else:
        commit_iter = repo.iter_commits(head_ref)

    try:
        return list(reversed(list(commit_iter)))
    except git.GitCommandError:
        # If HEAD is not found, an exception is raised.
        return []


def _FormatPythonDatetime(dt):
    """Make python datetime compatabile with git commit date args."""
    return dt.replace(tzinfo=None).replace(microsecond=0).isoformat()


def MaybeExportCommitSubset(
    commit: git.Commit,
    repo: git.Repo,
    files_of_interest: Set[str],
    ctx: progress.ProgressContext = progress.NullContext,
) -> Optional[git.Commit]:
    """Filter the parts of the given commit that touch the files_of_interest and
    commit them. If the commit doesn't touch anything interesting, nothing is
    commited.

    Args:
      repo: The git repo to add the commit to.

    Returns:
      A git commit, if one is created, else None.
    """
    try:
        # Apply the diff of the commit to be exported to the repo.
        repo.git.cherry_pick("--no-commit", "--allow-empty", commit)
        unmerged_to_add = set()
    except git.GitCommandError:
        # If cherry pick fails its because of merge conflicts.
        unmerged_paths = {path for path, _ in repo.index.unmerged_blobs().items()}
        unmerged_to_add = {path for path in unmerged_paths if path in files_of_interest}
        unmerged_to_rm = unmerged_paths - unmerged_to_add
        if unmerged_to_add:
            ctx.Log(
                3, "Adding %s", humanize.Plural(len(unmerged_to_add), "unmerged file")
            )
            # We have to remove an unmerged file before adding it again, else
            # the commit will fail with unmerged error.
            repo.index.remove(list(unmerged_to_add))
            repo.index.add(list(unmerged_to_add))
        if unmerged_to_rm:
            ctx.Log(
                3, "Removing %s", humanize.Plural(len(unmerged_to_rm), "unmerged file")
            )
            repo.index.remove(list(unmerged_to_rm))

    # Filter the changed files and exclude those that aren't interesting.
    modified_paths = {path for (path, _), _ in repo.index.entries.items()}
    paths_to_unstage = {
        path for path in modified_paths if path not in files_of_interest
    }
    paths_to_commit = modified_paths - paths_to_unstage

    if paths_to_unstage:
        ctx.Log(
            3,
            "Removing %s",
            humanize.Plural(len(paths_to_unstage), "uninteresting file"),
        )
        repo.index.remove(list(paths_to_unstage))

    if not repo.is_dirty():
        ctx.Log(2, "Skipping empty commit %s", commit)
        repo.git.clean("-xfd")
        return

    # I'm not sure about this one. The idea here is that in cases where there is
    # a merge error, checkout the file directly from the commit that the merge
    # error came from.
    try:
        modified_paths = [
            path.a_path for path in repo.index.diff("HEAD").iter_change_type("M")
        ]
        modified_unmerged_paths = set(modified_paths).union(unmerged_to_add)
        if modified_unmerged_paths:
            repo.git.checkout(commit, *modified_unmerged_paths)
    except (git.BadName, git.GitCommandError):
        pass

    # Append the hexsha of the original commit that this was exported from. This
    # can be used to determine the starting point for incremental exports, by
    # reading the commit messages and parsing this statement.
    message = f"{commit.message}\n[Exported from {commit}]"
    ctx.Log(
        3,
        "Committing %s from %s",
        humanize.Plural(len(paths_to_commit), "change"),
        commit,
    )

    new_commit = repo.index.commit(
        message=message,
        author=commit.author,
        committer=commit.committer,
        author_date=_FormatPythonDatetime(commit.authored_datetime),
        commit_date=_FormatPythonDatetime(commit.committed_datetime),
        skip_hooks=True,
    )
    repo.git.reset("--hard")
    repo.git.clean("-xfd")
    return new_commit
