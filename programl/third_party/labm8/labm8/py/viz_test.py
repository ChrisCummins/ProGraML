# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //labm8/py:viz."""
import pathlib

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from labm8.py import app, test, viz
from matplotlib import pyplot as plt

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def test_plot():
    """Test fixture that makes a plot."""
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)
    plt.plot(t, s)


@test.Parametrize("extension", (".png", ".pdf"))
def test_Finalize_produces_a_file(
    test_plot,
    tempdir: pathlib.Path,
    extension: str,
):
    """That file is produced."""
    del test_plot
    viz.Finalize(tempdir / f"plot{extension}")
    assert (tempdir / f"plot{extension}").is_file()


@test.Parametrize("extension", (".png", ".pdf"))
def test_Finalize_tight(test_plot, tempdir: pathlib.Path, extension: str):
    """That tight keyword."""
    del test_plot
    viz.Finalize(tempdir / f"plot{extension}", tight=True)
    assert (tempdir / f"plot{extension}").is_file()


@test.Parametrize("extension", (".png", ".pdf"))
def test_Finalize_figsize(test_plot, tempdir: pathlib.Path, extension: str):
    """That figsize keyword."""
    del test_plot
    viz.Finalize(tempdir / f"plot{extension}", figsize=(10, 5))
    assert (tempdir / f"plot{extension}").is_file()


def test_Distplot_dataframe():
    """Test plotting dataframe."""
    df = pd.DataFrame({"x": 1, "group": "foo"}, {"x": 2, "group": "bar"})
    viz.Distplot(x="x", hue="group", data=df)


def test_Distplot_with_hue_order():
    """Test plotting with hue order."""
    df = pd.DataFrame({"x": 1, "group": "foo"}, {"x": 2, "group": "bar"})
    viz.Distplot(x="x", hue="group", hue_order=["foo", "bar"], data=df)


def test_Distplot_with_missing_hue_order_values():
    """Plotting with missing hue order is not an error."""
    df = pd.DataFrame({"x": 1, "group": "foo"}, {"x": 2, "group": "bar"})
    viz.Distplot(x="x", hue="group", hue_order=["foo", "bar", "car"], data=df)


if __name__ == "__main__":
    test.Main()
