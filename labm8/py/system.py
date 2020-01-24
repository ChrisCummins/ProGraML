# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilies for grokking the underlying system.

Variables:
  * `HOSTNAME` (str) System hostname.
  * `USERNAME` (str) Username.
  * `UID` (int) User ID.
  * `PID` (int) Process ID.
"""
import getpass
import os
import socket
import subprocess
import sys
import tempfile
import threading
import typing
from sys import platform

from labm8.py import app
from labm8.py import fs

HOSTNAME = socket.gethostname()
USERNAME = getpass.getuser()
UID = os.getuid()
PID = os.getpid()

argv = sys.argv
STDOUT = sys.stdout
STDERR = sys.stderr
PIPE = subprocess.PIPE


class Error(Exception):
  pass


class SubprocessError(Error):
  """
  Error thrown if a subprocess fails.
  """

  pass


class CommandNotFoundError(Exception):
  """
  Error thrown a system command is not found.
  """

  pass


class ScpError(Error):
  """
  Error thrown if scp file transfer fails.
  """

  def __init__(self, stdout, stderr):
    """
    Construct an ScpError.

    Arguments:

        stdout (str): Captured stdout of scp subprocess.
        stderr (str): Captured stderr of scp subprocess.
    """
    self.out = stdout
    self.err = stderr

  def __repr__(self):
    return self.out + "\n" + self.err

  def __str__(self):
    return self.__repr__()


class Subprocess(object):
  """Subprocess abstraction.

  Wrapper around subprocess.Popen() which provides the ability to
  force a timeout after a number of seconds have elapsed.
  """

  def __init__(
    self,
    cmd,
    shell=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    decode_out=True,
  ):
    """
    Create a new subprocess.
    """
    self.cmd = cmd
    self.process = None
    self.stdout = None
    self.stderr = None
    self.shell = shell
    self.decode_out = decode_out

    self.stdout_dest = stdout
    self.stderr_dest = stderr

  def run(self, timeout=-1):
    """
    Run the subprocess.

    Arguments:
        timeout (optional) If a positive real value, then timout after
            the given number of seconds.

    Raises:
        SubprocessError If subprocess has not completed after "timeout"
            seconds.
    """

    def target():
      self.process = subprocess.Popen(
        self.cmd,
        stdout=self.stdout_dest,
        stderr=self.stderr_dest,
        shell=self.shell,
      )
      stdout, stderr = self.process.communicate()

      # Decode output if the user wants, and if there is any.
      if self.decode_out:
        if stdout:
          self.stdout = stdout.decode("utf-8")
        if stderr:
          self.stderr = stderr.decode("utf-8")

    thread = threading.Thread(target=target)
    thread.start()

    if timeout > 0:
      thread.join(timeout)
      if thread.is_alive():
        self.process.terminate()
        thread.join()
        raise SubprocessError(
          ("Reached timeout after {t} seconds".format(t=timeout)),
        )
    else:
      thread.join()

    return self.process.returncode, self.stdout, self.stderr


def is_linux():
  return platform == "linux" or platform == "linux2"


def is_mac():
  return platform == "darwin"


def is_windows():
  return platform == "win32"


def run(command, num_retries=1, timeout=-1, **kwargs):
  """
  Run a command with optional timeout and retries.

  Provides a convenience method for executing a subprocess with
  additional error handling.

  Arguments:
      command (list of str): The command to execute.
      num_retries (int, optional): If the subprocess fails, the number of
        attempts to execute it before failing.
      timeout (float, optional): If positive, the number of seconds to wait
        for subprocess completion before failing.
      **kwargs: Additional args to pass to Subprocess.__init__()

  Returns:
      Tuple of (int, str, str): Where the variables represent
      (exit status, stdout, stderr).

  Raises:
      SubprocessError: If the command fails after the given number of
        retries.
  """
  last_error = None
  for _ in range(num_retries):
    try:
      process = Subprocess(command, **kwargs)
      return process.run(timeout)
    except Exception as err:
      last_error = err

  raise last_error


def sed(match, replacement, path, modifiers=""):
  """Perform sed text substitution.

  This requires GNU sed. On MacOS, install it using:

      $ brew "gnu-sed"

  And then ensure that it is in the PATH before the OS-shipped sed:

      $ export PATH="/usr/local/opt/gnu-sed/libexec/gnubin:$PATH"
  """
  cmd = "sed -r -i 's/%s/%s/%s' %s" % (match, replacement, modifiers, path)

  process = Subprocess(cmd, shell=True)
  ret, out, err = process.run(timeout=60)
  if ret:
    raise SubprocessError("Sed command failed!")


def echo(*args, **kwargs):
  """
  Write a message to a file.

  Arguments:
      args A list of arguments which make up the message. The last argument
          is the path to the file to write to.
  """
  msg = args[:-1]
  path = fs.path(args[-1])
  append = kwargs.pop("append", False)

  if append:
    with open(path, "a") as file:
      print(*msg, file=file, **kwargs)
  else:
    with open(fs.path(path), "w") as file:
      print(*msg, file=file, **kwargs)


def which(program, path=None):
  """
  Returns the full path of shell commands.

  Replicates the functionality of system which (1) command. Looks
  for the named program in the directories indicated in the $PATH
  environment variable, and returns the full path if found.

  Examples:

      >>> system.which("ls")
      "/bin/ls"

      >>> system.which("/bin/ls")
      "/bin/ls"

      >>> system.which("not-a-real-command")
      None

      >>> system.which("ls", path=("/usr/bin", "/bin"))
      "/bin/ls"

  Arguments:

      program (str): The name of the program to look for. Can
        be an absolute path.
      path (sequence of str, optional): A list of directories to
        look for the pgoram in. Default value is system $PATH.

  Returns:

     str: Full path to program if found, else None.
  """
  # If path is not given, read the $PATH environment variable.
  path = path or os.environ["PATH"].split(os.pathsep)
  abspath = True if os.path.split(program)[0] else False
  if abspath:
    if fs.isexe(program):
      return program
  else:
    for directory in path:
      # De-quote directories.
      directory = directory.strip('"')
      exe_file = os.path.join(directory, program)
      if fs.isexe(exe_file):
        return exe_file

  return None


def isprocess(pid, error=False):
  """
  Check that a process is running.

  Arguments:

      pid (int): Process ID to check.

  Returns:

      True if the process is running, else false.
  """
  try:
    # Don't worry folks, no processes are harmed in the making of
    # this system call:
    os.kill(pid, 0)
    return True
  except OSError:
    return False


def exit(status=0):
  """
  Terminate the program with the given status code.
  """
  if status == 0:
    print("Done.", file=sys.stderr)
  else:
    print("Error {0}".format(status), file=sys.stderr)
  sys.exit(status)


def ProcessFileAndReplace(
  path: str,
  process_file_callback: typing.Callable[[str, str], None],
  tempfile_prefix: str = "labm8_system_",
  tempfile_suffix: str = None,
) -> None:
  """Process a file and replace with the generated file.

  This function provides the functionality of inplace file modification for
  functions which take an input file and produce an output file. It does this
  by creating a temporary file which, if the function returns successfully (i.e.
  without exception), will overwrite the original file.

  Args:
    path: The path of the file to process inplace.
    process_file_callback: A function which takes two arguments - the path of
      an input file, and the path of an output file.
    tempfile_prefix: An optional name prefix for the temporary file.
    tempfile_suffix: An optional name suffix for the temporary file.
  """
  with tempfile.NamedTemporaryFile(
    prefix=tempfile_prefix, suffix=tempfile_suffix, delete=False,
  ) as f:
    tmp_path = f.name
    try:
      process_file_callback(path, tmp_path)
      os.rename(tmp_path, path)
    finally:
      if os.path.isfile(tmp_path):
        os.unlink(tmp_path)


def CheckCallOrDie(cmd: typing.List[str]) -> None:
  """Run the given command and exit fatally on error."""
  try:
    app.Log(2, "$ %s", " ".join(cmd))
    subprocess.check_call(cmd)
  except subprocess.CalledProcessError as e:
    app.FatalWithoutStackTrace(
      "Command: `%s` failed with error: %s", " ".join(cmd), e,
    )
