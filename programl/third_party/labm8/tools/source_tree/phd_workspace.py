"""Library that handles custom stuff for working with this project."""
import datetime
import glob
import os
import pathlib
import re
import shutil
import subprocess
import typing

import git
from labm8.py import app, bazelutil, fs, humanize
from tools.git import export_subtree

FLAGS = app.FLAGS

# A list of relative paths to include in every export of this project. Glob
# patterns are expanded.
_ALWAYS_EXPORTED_FILES = [
    ".bazelrc",  # Not strictly required, but provides consistency.
    "BUILD",  # Top-level BUILD file is always needed.
    "configure",  # Needed to generate config proto.
    "CONTRIBUTING.md",  # Core documentation.
    "INSTALL.md",  # Core documentation.
    "README.md",  # Core documentation.
    "requirements.txt",  # Needed by WORKSPACE.
    "third_party/*.BUILD",  # Implicit dependencies of WORKSPACE file.
    "third_party/bazel/*",  # Needed by WORKSPACE.
    "third_party/py/*.bzl",  # Implicit dependencies of WORKSPACE file.
    "third_party/py/*.tpl",  # Implicit dependencies of WORKSPACE file.
    "tools/bazel",  # Optional, but useful.
    "tools/Brewfile.travis",  # Needed by Travis CI.
    "tools/bzl/*",  # Implicit dependency of WORKSPACE file.
    "tools/flaky_bazel.sh",  # Needed by Travis CI.
    "tools/whoami.sh",  # Debugging utility.
    "tools/workspace_status.sh",  # Needed by .bazelrc
    "WORKSPACE",  # Implicit dependency of everything.
]

# A list of relative paths to files which are excluded from export. Glob
# patterns are NOT supported.
_NEVER_EXPORTED_FILES = []


class PhdWorkspace(bazelutil.Workspace):
    def __init__(self, *args, **kwargs):
        super(PhdWorkspace, self).__init__(*args, **kwargs)
        self._repo = git.Repo(self.workspace_root)

    @property
    def git_repo(self) -> git.Repo:
        return self._repo

    @property
    def version(self) -> typing.Optional[str]:
        version_file = self.workspace_root / "version.txt"
        if version_file.is_file():
            return fs.read(version_file).rstrip()
        else:
            return None

    def GetAlwaysExportedFiles(self) -> typing.Iterable[str]:
        """Get hardcoded additional files to export."""
        relpaths = []
        for p in _ALWAYS_EXPORTED_FILES:
            abspaths = glob.glob(f"{self.workspace_root}/{p}")
            relpaths += [
                os.path.relpath(path, self.workspace_root) for path in abspaths
            ]
        return relpaths

    def GetAuxiliaryExportFiles(self, paths: typing.Set[str]) -> typing.List[str]:
        """Get a list of auxiliary files to export."""

        def GlobToPaths(glob_pattern: str) -> typing.List[str]:
            abspaths = glob.glob(glob_pattern)
            return [os.path.relpath(path, self.workspace_root) for path in abspaths]

        auxiliary_exports = []
        for path in paths:
            dirname = (self.workspace_root / path).parent
            auxiliary_exports += GlobToPaths(f"{dirname}/DEPS.txt")
            auxiliary_exports += GlobToPaths(f"{dirname}/README*")
            auxiliary_exports += GlobToPaths(f"{dirname}/LICENSE*")

        return auxiliary_exports

    def GetPythonRequirementsForTarget(
        self, targets: typing.List[str]
    ) -> typing.List[str]:
        """Get the subset of requirements.txt which is needed for a target."""
        # The set of bazel dependencies for all targets.
        dependencies = set()
        for target in targets:
            bazel = self.BazelQuery([f"deps({target})"], stdout=subprocess.PIPE)
            grep = subprocess.Popen(
                ["grep", "^@requirements_pypi__"],
                stdout=subprocess.PIPE,
                stdin=bazel.stdout,
                universal_newlines=True,
            )
            stdout, _ = grep.communicate()
            if bazel.returncode:
                raise OSError("bazel query failed")
            output = stdout.rstrip()
            if output:
                dependencies = dependencies.union(set(output.split("\n")))

        with open(str(self.workspace_root / "requirements.txt")) as f:
            all_requirements = set(f.readlines())

        needed = []
        all_dependencies = set()
        # This is a pretty hacky approach that tries to match the package component
        # of the generated @pypi__<package>_<vesion> package to the name as it
        # appears in requirements.txt.
        for dependency in dependencies:
            if not dependency.startswith("@requirements_pypi__"):
                continue
            dependency = (
                dependency[len("@requirements_pypi__") :].lower().split("//:")[0]
            )
            all_dependencies.add(dependency)

        def _GetMatchingRequirements(dependency: str, all_requirements):
            requirements = []
            for requirement in all_requirements:
                requirement_to_match = requirement.replace("-", "_").lower().rstrip()
                requirement_to_match = requirement_to_match.split("==")[0]
                requirement_to_match = re.sub(r"#.*", "", requirement_to_match)
                if not requirement_to_match:
                    continue
                if dependency.startswith(requirement_to_match):
                    requirements.append(requirement.rstrip())
            return requirements

        for dependency in all_dependencies:
            requirements = _GetMatchingRequirements(dependency, all_requirements)
            if not requirements:
                continue
            needed += requirements
            # Total hack workaround for the fact that
            # //third_party/py/scipy:BUILD pulls in multiple packages.
            for requirement in requirements:
                if requirement.startswith("scipy"):
                    needed += _GetMatchingRequirements("scikit", all_requirements)

        return list(sorted(set(needed)))

    def FilterExcludedPaths(
        self, paths: typing.List[pathlib.Path]
    ) -> typing.Iterable[pathlib.Path]:
        return [path for path in paths if path not in _NEVER_EXPORTED_FILES]

    def GetAllSourceTreeFiles(
        self,
        targets: typing.List[str],
        excluded_targets: typing.Iterable[str],
        extra_files: typing.List[str],
        file_move_mapping: typing.Dict[str, str],
    ) -> typing.List[pathlib.Path]:
        """Get the full list of source files to export for targets."""
        excluded_targets = set(excluded_targets)

        # Never export `exports_repo` targets.
        excluded_targets = excluded_targets.union('kind("exports_repo", //...)')

        file_set = set(extra_files).union(set(file_move_mapping.values()))
        for target in targets:
            file_set = file_set.union(
                set(self.GetDependentFiles(target, excluded_targets))
            )
            file_set = file_set.union(set(self.GetBuildFiles(target)))
            file_set = file_set.union(
                set(self.GetDependentFiles(target, excluded_targets))
            )
            file_set = file_set.union(
                set(self.GetDependentFiles(target, excluded_targets))
            )

        file_set = file_set.union(set(self.GetAlwaysExportedFiles()))
        file_set = file_set.union(set(self.GetAuxiliaryExportFiles(file_set)))
        filtered_files = self.FilterExcludedPaths(file_set)

        return list(sorted(filtered_files))

    def CopyFilesToDestination(
        self, workspace: bazelutil.Workspace, files: typing.List[str]
    ) -> None:
        for relpath in files:
            print(relpath)

            src_path = self.workspace_root / relpath
            dst_path = workspace.workspace_root / relpath

            if not src_path.is_file():
                raise OSError(f"File `{relpath}` not found")

            dst_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(src_path, dst_path)

    def MoveFilesToDestination(
        self,
        workspace: bazelutil.Workspace,
        file_move_mapping: typing.Dict[str, str],
    ) -> None:
        with fs.chdir(workspace.workspace_root):
            for src_relpath, dst_relpath in file_move_mapping.items():
                print(dst_relpath)

                src_path = self.workspace_root / src_relpath
                dst_path = workspace.workspace_root / dst_relpath
                if not src_path.is_file():
                    raise OSError(f"File `{src_relpath}` not found")

                dst_path.parent.mkdir(exist_ok=True, parents=True)
                # We can't simply `git mv` because in incremental exports, this move
                # may have already been applied. Instead, we manually copy the file
                # from the source workspace, and delete the corresponding file in the
                # destination workspace.
                shutil.copy(src_path, dst_path)
                dst_src_path = workspace.workspace_root / src_relpath
                if dst_src_path.is_file():
                    subprocess.check_call(["git", "rm", src_relpath])

    def ExportToRepo(
        self,
        repo: git.Repo,
        targets: typing.List[str],
        src_files: typing.List[str],
        extra_files: typing.List[str],
        file_move_mapping: typing.Dict[str, str],
    ) -> int:
        """Export the requested targets to the destination directory."""
        # The timestamp for the export.
        timestamp = datetime.datetime.utcnow()

        # Check now that all of the auxiliary files exist.
        for relpath in extra_files:
            if not (self.workspace_root / relpath).is_file():
                raise FileNotFoundError(self.workspace_root / relpath)
        for relpath in file_move_mapping:
            if not (self.workspace_root / relpath).is_file():
                raise FileNotFoundError(self.workspace_root / relpath)

        # Export the git history.
        app.Log(
            1, "Exporting git history for %s files", humanize.Commas(len(src_files))
        )
        for file in src_files:
            print(file)

        exported_commit_count = export_subtree.ExportSubtree(
            source=self.git_repo,
            destination=repo,
            files_of_interest=set(src_files),
        )
        if not exported_commit_count:
            return 0

        # Make manual adjustments.
        exported_workspace = bazelutil.Workspace(pathlib.Path(repo.working_tree_dir))
        self.CopyFilesToDestination(exported_workspace, extra_files)
        self.MoveFilesToDestination(exported_workspace, file_move_mapping)

        if not repo.is_dirty(untracked_files=True):
            return exported_commit_count

        app.Log(1, "Creating automated subtree export commit")
        repo.git.add(".")
        author = git.Actor(name="[Git export bot]", email="/dev/null")
        repo.index.commit(
            f"Automated subtree export at {timestamp.isoformat()}",
            author=author,
            committer=author,
            skip_hooks=True,
        )
        return exported_commit_count
