"""This module aims to help evaluate the progress of long-running jobs."""
import threading
from typing import Callable
from typing import Optional
from typing import Union

import tqdm

from labm8.py import app
from labm8.py import prof

FLAGS = app.FLAGS


class Progress(threading.Thread):
  """A job with intermediate progress and a progress bar."""

  def __init__(self, *args, **kwargs):
    """Instantiate a long-running job.

    Args:
      args: Arguments for ProgressBarContext().
      kwargs: Keyword arguments for ProgressBarContext().
    """
    self.ctx = ProgressBarContext(*args, **kwargs)
    super(Progress, self).__init__()

    self.run = self.Run
    self.Start = self.start
    self.IsAlive = self.is_alive

  def Run(self):
    """The run method. This should progress from [self.ctx.i, self.ctx.n]."""
    raise NotImplementedError("abstract class")


def Run(progress: Progress, refresh_time: float = 0.2) -> Progress:
  """Run the given progress job until completion, updating the progress bar."""
  progress.Start()
  while progress.is_alive():
    progress.ctx.Refresh()
    progress.join(refresh_time)
  progress.ctx.Refresh()
  return progress


class ProgressContext(object):
  """A context for logging and profiling."""

  def __init__(self, print_context):
    """Constructor.

    Args:
      print_context: A context for printing.
    """
    self.print_context = print_context

  @app.skip_log_prefix
  def Log(self, *args, **kwargs):
    """Log a message."""
    app.Log(*args, **kwargs, print_context=self.print_context)

  @app.skip_log_prefix
  def Warning(self, *args, **kwargs):
    """Log a warning."""
    app.Warning(*args, **kwargs, print_context=self.print_context)

  @app.skip_log_prefix
  def Error(self, *args, **kwargs):
    """Log an error."""
    app.Error(*args, **kwargs, print_context=self.print_context)

  @app.skip_log_prefix
  def Profile(self, level: int, msg: Union[str, Callable[[int], str]] = ""):
    """Return a profiling context."""
    return prof.Profile(msg, print_to=lambda x: self.Log(level, x),)


class ProgressBarContext(ProgressContext):
  """The context for logging and profiling with a progress bar."""

  def __init__(
    self,
    name: str,
    i: int,
    n: Optional[int] = None,
    unit: str = "",
    vertical_position: int = 0,
  ):
    """Construct a new progress.

    Args:
      name: The name of the job being processed. This is a prefix for the
        progress bar output.
      i: The starting value for the job.
      n: The end value for the job. If not known this may be None. When None,
        no estimated time or "bar" is printed during progress.
      unit: An optional unit for progress bar.
      vertical_position: The vertical position of the progress bar. Use this
        when there are multiple progress bar concurrently.
    """
    self.name = name
    self.i = i
    self.n = n
    self.unit = unit
    self.vertical_position = vertical_position

    # Prepend whitespace to the unit
    unit = f" {unit}" if unit else unit

    # Create the progress bar.
    self.bar = tqdm.tqdm(
      desc=self.name,
      initial=self.i,
      total=self.n,
      unit=unit,
      position=self.vertical_position,
    )
    self.print_context = self.bar.external_write_mode

  def ToProgressContext(self) -> ProgressContext:
    """Construct a progress context from the given progress context with bar.

    This method is useful for sending the progress context to worker processes
    during multimprocessing. By removing self.bar attribute, we need only
    serialize the logging context.
    """
    return ProgressContext(self.print_context)

  def Refresh(self):
    """Refresh the progress bar."""
    self.bar.n = self.i
    self.bar.refresh()
