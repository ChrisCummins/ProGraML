"""A module for launching docker images from within python applications."""
import contextlib
import pathlib
import random
import subprocess
import typing

from labm8.py import app, bazelutil, labtypes


def IsDockerContainer() -> bool:
    """Determine if running inside a docker container."""
    return pathlib.Path("/.dockerenv").is_file()


def _Docker(cmd: typing.List[str], timeout: int = 60):
    """Build a docker process invocation."""
    cmd = ["timeout", "-s9", str(timeout), "docker"] + [str(s) for s in cmd]
    app.Log(2, "$ %s", " ".join(cmd))
    return cmd


class DockerImageRunContext(object):
    """A transient context for running docker images."""

    def __init__(self, image_name: str):
        self.image_name = image_name

    def _CommandLineInvocation(
        self,
        args: typing.List[str],
        flags: typing.Dict[str, str],
        volumes: typing.Dict[typing.Union[str, pathlib.Path], str],
        timeout: int,
        entrypoint: typing.Optional[str],
    ) -> typing.List[str]:
        """Build the command line arguments to execute the requested command.

        Args:
          args: A list of string positional arguments to pass to the docker image.
          flags: A map of flag arguments. The keys are prefixed with '--' and
            concatenated with the keys to produce arguments. E.g. {"foo": 2}
            equates to ["--foo", "2"].
          volumes: A map of shared volumes, e.g. {"/tmp": "/foo"} equates to
            the argument "-v/tmp:/foo". It is the responsibility of the calling
            code to ensure that the host paths are accessible to docker, and have
            the appropriate permissions for the docker user to access / modify. See
            //labm8/py:dockerutil_test for an example.
          timeout: The number of seconds to allow the image to run for before being
            killed. Killed processes will exit with returncode 9.
          entrypoint: An optional entrypoint for the docker image, equivalent to
            ["--entrypoint", entrypoint].

        Returns:
          The command line as a list of strings.
        """
        entrypoint_args = ["--entrypoint", entrypoint] if entrypoint else []
        volume_args = [f"-v{src}:{dst}" for src, dst in (volumes or {}).items()]
        flags_args = labtypes.flatten(
            [[f"--{k}", str(v)] for k, v in (flags or {}).items()],
        )
        return _Docker(
            ["run", "--rm"]
            + entrypoint_args
            + volume_args
            + [self.image_name]
            + args
            + flags_args,
            timeout,
        )

    def CheckCall(
        self,
        args: typing.List[str],
        flags: typing.Dict[str, str] = None,
        volumes: typing.Dict[typing.Union[str, pathlib.Path], str] = None,
        timeout: int = 600,
        entrypoint: str = None,
    ) -> None:
        """Run the docker image with specified args.

        This attempts to emulate the behavior of subproces.check_call() for
        docker images.

        See _CommandLineInvocation() for details on args.
        """
        cmd = self._CommandLineInvocation(args, flags, volumes, timeout, entrypoint)
        subprocess.check_call(cmd)

    def CheckOutput(
        self,
        args: typing.List[str],
        flags: typing.Dict[str, str] = None,
        volumes: typing.Dict[typing.Union[str, pathlib.Path], str] = None,
        timeout: int = 600,
        entrypoint: str = None,
    ) -> str:
        """Run the docker image with specified args and return its output.

        This attempts to emulate the behavior of subproces.check_output() for
        docker images.

        See _CommandLineInvocation() for details on args.
        """
        cmd = self._CommandLineInvocation(args, flags, volumes, timeout, entrypoint)
        return subprocess.check_output(cmd, universal_newlines=True)


class BazelPy3Image(object):
    """A docker image created using bazel's py3_image() rule.

    To use a py3_image with this class, add the py3_image target with a ".tar"
    suffix as a data dependency to the bazel target, e.g.

        load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

        py3_image(
            name = "my_image",
            srcs = ["my_image.py"],
        )

        py_binary(
            name = "my_app",
            srcs = ["my_app.py"],
            data = [
                ":my_image.tar",
            ],
            deps = [
                "//labm8/py:app",
                "//labm8/py:dockerutil",
            ],
        )
    """

    def __init__(self, data_path: str):
        """Constructor.

        Args:
          path: The path to the data, including the name of the workspace.

        Raises:
          FileNotFoundError: If path is not a file.
        """
        super(BazelPy3Image, self).__init__()
        self.data_path = data_path
        self.tar_path = bazelutil.DataPath(f"phd/{data_path}.tar")

        components = self.data_path.split("/")
        self.image_name = f'bazel/{"/".join(components[:-1])}:{components[-1]}'

    def _TemporaryImageName(self) -> str:
        basename = self.data_path.split("/")[-1]
        random_suffix = "".join(random.choice("0123456789abcdef") for _ in range(32))
        return f"phd_{basename}_tmp_{random_suffix}"

    @contextlib.contextmanager
    def RunContext(self) -> DockerImageRunContext:
        subprocess.check_call(
            _Docker(["load", "-i", str(self.tar_path)], timeout=600),
        )
        tmp_name = self._TemporaryImageName()
        subprocess.check_call(
            _Docker(["tag", self.image_name, tmp_name], timeout=60),
        )
        subprocess.check_call(_Docker(["rmi", "--force", self.image_name], timeout=60))
        yield DockerImageRunContext(tmp_name)
        # FIXME(cec): Using the --force flag here is almost certainly the wrong
        # thing, but I'm getting strange errors when trying to untag the image
        # otherwise:
        #   Error response from daemon: conflict: unable to remove repository
        #   reference "phd_..." (must force) - container ... is using its
        #   referenced image ...
        subprocess.check_call(_Docker(["rmi", "--force", tmp_name], timeout=60))
