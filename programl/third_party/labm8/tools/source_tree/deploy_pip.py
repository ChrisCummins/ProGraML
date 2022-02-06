"""Library for deploying releases of bazel targets.

Currently, pip package deployment is supported through the DEPLOY_PIP function.
"""
import contextlib
import pathlib
import shutil
import subprocess
import typing

from labm8.py import app, fs
from labm8.py.internal import workspace_status
from tools.source_tree import phd_workspace

FLAGS = app.FLAGS

app.DEFINE_string("package_name", None, "The name of the package to export.")
app.DEFINE_string(
    "package_root",
    None,
    'The root for finding python targets to export, e.g. "//labm8".',
)
app.DEFINE_string(
    "description", None, "A string with the short description of the package"
)
app.DEFINE_string(
    "release_type",
    "testing",
    "The type of release to upload to pypi. One of {testing,release}.",
)
app.DEFINE_list(
    "classifiers",
    None,
    "A list of strings, containing Python package classifiers",
)
app.DEFINE_list("keywords", None, "A list of strings, containing keywords")
app.DEFINE_string("license", None, "The type of license to use")
app.DEFINE_string(
    "long_description_file",
    None,
    'A label with the long description of the package, e.g. "//labm8/py:README.md"',
)
app.DEFINE_string("url", None, "A homepage for the project.")


def FindPyLibraries(
    workspace: phd_workspace.PhdWorkspace, root_package: str
) -> typing.List[str]:
    """Find the python libraries to export."""
    # Break down of the bazel query, from right to left:
    #    * Glob for all python targets under the root package.
    #    * Exclude tests, and testonly libraries, since they can't be the
    #      dependency of a py_library.
    #    * Select only those that are visible at the top of the repo, i.e. those
    #      with //visibility:public visibility.
    p = workspace.BazelQuery(
        [f"visible(//:version.txt, attr(testonly, 0, kind(py_*, {root_package}/...)))"],
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    stdout, _ = p.communicate()
    if p.returncode:
        raise OSError("bazel query failed")
    targets = stdout.rstrip().split("\n")
    return targets


@contextlib.contextmanager
def DeploymentDirectory(workspace: phd_workspace.PhdWorkspace) -> str:
    """Temporarily create a //deployment package for putting generated files in."""
    try:
        deployment_package = "deployment"
        deployment_directory = workspace.workspace_root / deployment_package
        assert not deployment_directory.is_dir()
        deployment_directory.mkdir()
        yield deployment_package
    finally:
        shutil.rmtree(deployment_directory)


def GetUrlFromCnameFile(workspace: phd_workspace.PhdWorkspace, package_root: str):
    relpath = workspace.MaybeTargetToPath(f"{package_root}:CNAME")
    if not relpath:
        raise OSError("`url` not set and could not find {package_root}:CNAME")
    return fs.Read(workspace.workspace_root / relpath).strip()


def _DoDeployPip(
    package_name: str,
    package_root: str,
    classifiers: typing.List[str],
    description: str,
    keywords: typing.List[str],
    license: str,
    long_description_file: str,
    url: typing.Optional[str],
):
    """Private implementation function."""
    release_type = FLAGS.release_type
    if release_type not in {"release", "testing"}:
        raise OSError("--release_type must be one of: {testing,release}")

    source_path = pathlib.Path(workspace_status.STABLE_UNSAFE_WORKSPACE)
    workspace = phd_workspace.PhdWorkspace(source_path)

    if url is None:
        url = GetUrlFromCnameFile(workspace, package_root)

    with DeploymentDirectory(workspace) as deployment_package:
        app.Log(1, "Assembling package ...")
        targets = FindPyLibraries(workspace, package_root)

        requirements = workspace.GetPythonRequirementsForTarget(targets)
        requirements = [r.split()[0] for r in requirements]

        def QuotedList(items, indent: int = 8):
            """Produce an indented, quoted list."""
            prefix = " " * indent
            list = f",\n{prefix}".join([f'"{x}"' for x in items])
            return f"{prefix}{list},"

        rules = f"""
load(
    "@graknlabs_bazel_distribution//pip:rules.bzl",
    "assemble_pip",
    "deploy_pip"
)

py_library(
    name = "{package_name}",
    deps = [
{QuotedList(targets)}
    ],
)

assemble_pip(
    name = "package",
    author = "Chris Cummins",
    author_email ="chrisc.101@gmail.com",
    classifiers=[
{QuotedList(classifiers)}
    ],
    description="{description}",
    install_requires=[
{QuotedList(requirements)}
    ],
    keywords=[
{QuotedList(keywords)}
    ],
    license="{license}",
    long_description_file="{long_description_file}",
    package_name="{package_name}",
    target=":{package_name}",
    url="{url}",
    version_file="//:version.txt",
)

deploy_pip(
  name = "deploy",
  deployment_properties = "//:deployment.properties",
  target = ":package",
)
"""
        app.Log(2, "Generated rules:\n%s", rules)

        fs.Write(
            workspace.workspace_root / deployment_package / "BUILD",
            rules.encode("utf-8"),
        )

        # Build and deploy separately, to make it easier to diagnose the cause of
        # any error.
        app.Log(1, "Building package ...")
        build = workspace.Bazel("build", [f"//{deployment_package}:package"])
        build.communicate()
        if build.returncode:
            raise OSError("package build failed")

        app.Log(1, "Deploying package ...")
        build = workspace.Bazel(
            "run",
            [f"//{deployment_package}:deploy", "--", release_type],
        )
        build.communicate()
        if build.returncode:
            raise OSError("deployment failed")


def _RunFunctionWithFatalOSError(fn, **kwargs):
    """Run the given function with the given args and make OSError fatal."""
    try:
        fn(**kwargs)
    except OSError as e:
        app.FatalWithoutStackTrace(e)


def DEPLOY_PIP(
    package_name: str = None,
    package_root: str = None,
    description: str = None,
    classifiers: typing.List[str] = None,
    keywords: typing.List[str] = None,
    license: str = None,
    long_description_file: str = None,
    url: str = None,
):
    """Custom entry-point to deploy a pip package.

    This should be called from a bare python script, before flags parsing.

    Args:
      package_name: The name of the pypi package.
      package_root: The root for finding python targets to export,
          e.g. "//labm8".
      description: A string with the short description of the package.
      classifiers: A list of strings, containing Python package classifiers.
      keywords: A list of strings, containing keywords.
      license: The type of license to use.
      long_description_file: A label with the long description of the package,
        e.g. "//labm8/py:README.md".
      url: A homepage for the project.
    """
    app.Run(
        lambda: _RunFunctionWithFatalOSError(
            _DoDeployPip,
            package_name=package_name,
            package_root=package_root,
            description=description,
            classifiers=classifiers,
            keywords=keywords,
            license=license,
            long_description_file=long_description_file,
            url=url,
        )
    )


def main():
    _RunFunctionWithFatalOSError(
        _DoDeployPip,
        package_name=FLAGS.package_name,
        package_root=FLAGS.package_root,
        description=FLAGS.description,
        classifiers=FLAGS.classifiers,
        keywords=FLAGS.keywords,
        license=FLAGS.license,
        long_description_file=FLAGS.long_description_file,
        url=FLAGS.url,
    )


if __name__ == "__main__":
    app.Run(main)
