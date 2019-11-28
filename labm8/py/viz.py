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
"""Plotting and visualization utilities."""
import pathlib
import subprocess
import tempfile
import time
import typing

import matplotlib
import numpy as np
import seaborn as sns
from IPython.core import display
from matplotlib import axes
from matplotlib import pyplot as plt
from scipy import stats

from labm8.py import app
from labm8.py import fs

FLAGS = app.FLAGS


def Finalize(output: typing.Optional[typing.Union[str, pathlib.Path]] = None,
             figsize=None,
             tight=True,
             **savefig_opts):
  """Finalise a plot.

  Display or show the plot, then close it.

  Args:
    output: Path to save figure to. If not given, plot is shown.
    figsize: Figure size in inches.
    **savefig_opts: Any additional arguments to pass to
      plt.savefig(). Only required if output is not None.
  """
  # Set figure size.
  if figsize is not None:
    plt.gcf().set_size_inches(*figsize)

  # Set plot layout.
  if tight:
    plt.tight_layout()

  if output is None:
    plt.show()
  else:
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), **savefig_opts)
    app.Log(1, "Wrote '%s'", output)
  plt.close()


def ShowErrorBarCaps(ax: axes.Axes):
  """Show error bar caps.
  
  Seaborn paper style hides error bar caps. Call this function on an axes
  object to make them visible again.
  """
  for child in ax.get_children():
    if str(child).startswith('Line2D'):
      child.set_markeredgewidth(1)
      child.set_markersize(8)


def RotateXLabels(rotation: int = 90, ax: axes.Axes = None) -> None:
  """Rotate plot X labels anti-clockwise.

  Args:
    rotation: The number of degrees to rotate the labels by.
    ax: The plot axis.
  """
  ax = ax or plt.gca()
  plt.setp(ax.get_xticklabels(), rotation=rotation)


def RotateYLabels(rotation: int = 90, ax: axes.Axes = None):
  """Rotate plot Y labels anti-clockwise.

  Args:
    rotation: The number of degrees to rotate the labels by.
    ax: The plot axis.
  """
  ax = ax or plt.gca()
  plt.setp(ax.get_yticklabels(), rotation=rotation)


def FormatXLabelsAsTimestamps(
    format='%H:%M:%S',
    convert_to_seconds=lambda t: t / 1000,
    ax: axes.Axes = None,
) -> None:
  """Format the X labels

  Args:
    format: The time formatting string.
    convert_to_seconds: A function to convert values to seconds.
    ax: The plot axis.
  """
  ax = ax or plt.gca()
  formatter = matplotlib.ticker.FuncFormatter(
      lambda t, _: time.strftime(format, time.gmtime(convert_to_seconds(t))),
  )
  ax.xaxis.set_major_formatter(formatter)


def Distplot(
    x=None,
    hue=None,
    data=None,
    log_x: bool = False,
    log1p_x: bool = False,
    kde=False,
    bins=None,
    nbins: typing.Optional[int] = None,
    norm_hist=False,
    hue_order=None,
    ax=None,
):
  """An extension of seaborn distribution plots for grouped data.

  Args:
    x: The x label for dataframe.
    hue: An optional categorical grouping for data.
    data: Dataset for plotting.
    log_x: If True, log transform the data before plotting.
    log1p_x: If True, log1p transform the data before plotting.
    kde: Whether to plot a gaussian kernel density estimate.
    bins: Specification of hist bins, or None to use Freedman-Diaconis rule.
    nbins: If bins is None, specify the number of bins to use. If not provided,
      use min(30, Freedman-Diaconis rule).
    norm_hist: If True, the histogram height shows a density rather than a
      count. This is implied if a KDE or fitted density is plotted.
    hue_order: Order to plot the categorical levels in, otherwise the levels are
      inferred from the data objects.
    ax: If provided, plot on this axis.

  Returns:
    Returns the Axes object with the plot for further tweaking.
  """

  # Utility code taken from seaborn. See:
  # https://github.com/mwaskom/seaborn/blob/master/seaborn/distributions.py
  #
  # Copyright (c) 2012-2018, Michael L. Waskom
  # All rights reserved.
  #
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions are met:
  #
  # * Redistributions of source code must retain the above copyright notice,
  #   this list of conditions and the following disclaimer.
  #
  # * Redistributions in binary form must reproduce the above copyright notice,
  #   this list of conditions and the following disclaimer in the documentation
  #   and/or other materials provided with the distribution.
  #
  # * Neither the name of the project nor the names of its
  #   contributors may be used to endorse or promote products derived from
  #   this software without specific prior written permission.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  # POSSIBILITY OF SUCH DAMAGE.

  def iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1

  def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
      return 1
    h = 2 * iqr(a) / (len(a)**(1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
      return int(np.sqrt(a.size))
    else:
      return int(np.ceil((a.max() - a.min()) / h))

  # End of seaborn utility code.

  values_to_plot = data[x]
  if log_x:
    values_to_plot = np.log(values_to_plot)
  elif log1p_x:
    values_to_plot = np.log1p(values_to_plot)

  if bins is None:
    if nbins is None:
      nbins = min(_freedman_diaconis_bins(values_to_plot), 50)
    bins = np.linspace(min(values_to_plot), max(values_to_plot), nbins)

  if ax is None:
    ax = plt.gca()

  if hue is None:
    sns.distplot(
        values_to_plot,
        kde=kde,
        bins=bins,
        label=x,
        norm_hist=norm_hist,
        ax=ax,
    )
  else:
    hue_order = hue_order or sorted(set(data[hue]))
    for h in hue_order:
      sns.distplot(
          values_to_plot[data[hue] == h],
          kde=kde,
          bins=bins,
          label=h,
          norm_hist=norm_hist,
          ax=ax,
      )
    plt.legend()

  return ax


def SummarizeFloats(floats: typing.Iterable[float], nplaces: int = 2) -> str:
  """Summarize a sequence of floats."""
  arr = np.array(list(floats), dtype=np.float32)
  percs = ' '.join([
      f'{p}%={np.percentile(arr, p):.{nplaces}f}' for p in [0, 50, 95, 99, 100]
  ])
  return (
      f'n={len(arr)}, mean={arr.mean():.{nplaces}f}, stdev={arr.std():.{nplaces}f}, '
      f'percentiles=[{percs}]')


def SummarizeInts(ints: typing.Iterable[int]) -> str:
  """Summarize a sequence of ints."""
  arr = np.array(list(ints), dtype=np.int32)
  percs = ' '.join(
      [f'{p}%={np.percentile(arr, p):.0f}' for p in [0, 50, 95, 99, 100]],)
  return (f'n={len(arr)}, mean={arr.mean():.2f}, stdev={arr.std():.2f}, '
          f'percentiles=[{percs}]')


def PlotDot(dot: str) -> None:
  """Compile and display the given dot plot."""
  with tempfile.TemporaryDirectory() as d:
    dot_path = pathlib.Path(d) / 'dot.dot'
    png_path = pathlib.Path(d) / 'dot.png'

    fs.Write(dot_path, dot.encode('utf-8'))
    try:
      subprocess.check_call(
          ['dot', str(dot_path), '-Tpng', '-o',
           str(png_path)])
    except subprocess.CalledProcessError as e:
      raise ValueError(f"Failed to process dotgraph: {dot}")
    display.display(display.Image(filename=f'{d}/dot.png'))
