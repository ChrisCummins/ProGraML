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
"""Unit tests for //labm8/py:labmath."""
import math

import pytest
from labm8.py import app, labmath, test

FLAGS = app.FLAGS


def test_ceil():
    assert isinstance(labmath.ceil(1), int)
    assert 1 == labmath.ceil(1)
    assert 2 == labmath.ceil(1.1)
    assert 3 == labmath.ceil(2.5)
    assert 4 == labmath.ceil(3.9)


def test_ceil_bad_params():
    with test.Raises(TypeError):
        labmath.ceil(None)
    with test.Raises(TypeError):
        labmath.ceil("abc")


def test_floor():
    assert isinstance(labmath.floor(1), int)
    assert 1 == labmath.floor(1)
    assert 1 == labmath.floor(1.1)
    assert 2 == labmath.floor(2.5)
    assert 3 == labmath.floor(3.9)


def test_floor_bad_params():
    with test.Raises(TypeError):
        labmath.floor(None)
    with test.Raises(TypeError):
        labmath.floor("abc")


def test_sqrt_4():
    assert 2 == labmath.sqrt(4)


def test_sqrt():
    assert math.sqrt(1024) == labmath.sqrt(1024)


# mean() tests
def test_mean_empty_array():
    assert 0 == labmath.mean([])


def test_mean_single_item_array():
    assert 1 == labmath.mean([1])


def test_mean():
    assert 2 == labmath.mean([1, 2, 3])
    assert (1 / 3.0) == labmath.mean([1, 1.5, -1.5])
    assert 2 == labmath.mean([2, 2, 2, 2, 2])
    assert 2.5 == labmath.mean([1, 2, 3, 4])


# mean() tests
def test_geomean_empty_array():
    assert 0 == labmath.geomean([])


def test_geomean_single_item_array():
    assert 1 == labmath.geomean([1])


def test_geomean():
    assert pytest.approx(labmath.geomean([1, 2, 3]), 1.8171205928321397)
    assert pytest.approx(labmath.geomean([1, 1.5, 2]), 1.44224957031)
    assert labmath.geomean([2, 2, 2, 2, 2]) == 2
    assert pytest.approx(labmath.geomean([1, 2, 3, 4]), 2.2133638394)
    assert labmath.geomean([0, 1, 2, 3, 4]) == 0


# median() tests
def test_median_empty_array():
    assert 0 == labmath.median([])


def test_median_single_item_array():
    assert 1 == labmath.median([1])


def test_median():
    assert 2 == labmath.median([1, 2, 3])
    assert 1 == labmath.median([1, 1.5, -1.5])
    assert 2.5 == labmath.median([1, 2, 3, 4])


# range() tests
def test_range_empty_array():
    assert 0 == labmath.range([])


def test_range_single_item_array():
    assert 0 == labmath.range([1])


def test_range_123_array():
    assert 2 == labmath.range([1, 2, 3])


# variance() tests
def test_variance_empty_array():
    assert 0 == labmath.variance([])


def test_variance_single_item_array():
    assert 0 == labmath.variance([1])


def test_variance_123_array():
    assert 1 == labmath.variance([1, 2, 3])


# stdev() tests
def test_stdev_empty_array():
    assert 0 == labmath.stdev([])


def test_stdev_single_item_array():
    assert 0 == labmath.stdev([1])


def test_stdev_123_array():
    assert 1 == labmath.stdev([1, 2, 3])


# iqr() tests
def test_filter_iqr():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert 0 == labmath.iqr(a, 0, 1)
    assert [4, 5, 6, 7] == labmath.iqr(a, 0.25, 0.75)
    assert [2, 3, 4, 5, 6, 7] == labmath.iqr(a, 0.1, 0.75)


# filter_iqr() tests
def test_filter_iqr():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert a == labmath.filter_iqr(a, 0, 1)
    assert [4, 5, 6, 7] == labmath.filter_iqr(a, 0.25, 0.75)
    assert [2, 3, 4, 5, 6, 7] == labmath.filter_iqr(a, 0.1, 0.75)


# confinterval() tests
def test_confinterval_empty_array():
    assert (0, 0) == labmath.confinterval([])


def test_confinterval_single_item_array():
    assert (1, 1) == labmath.confinterval([1])


def test_confinterval_123_array():
    assert labmath.confinterval([1, 2, 3,]) == (
        -0.48413771184375287,
        4.4841377118437524,
    )


def test_confinterval_all_same():
    assert (1, 1) == labmath.confinterval([1, 1, 1, 1, 1])


def test_confinterval_c50():
    assert (1.528595479208968, 2.4714045207910322) == labmath.confinterval(
        [1, 2, 3],
        conf=0.5,
    )


def test_confinterval_normal_dist():
    assert (0.86841426592382809, 3.1315857340761717) == labmath.confinterval(
        [1, 2, 3],
        normal_threshold=1,
    )


def test_confinterval_array_mean():
    assert pytest.approx(
        labmath.confinterval([1, 2, 3], conf=0.5, array_mean=2),
        (1.528595479208968, 2.4714045207910322),
    )
    assert pytest.approx(
        labmath.confinterval([1, 2, 3], conf=0.5, array_mean=1),
        (0.528595479209, 1.47140452079),
    )


def test_confinterval_error_only():
    assert pytest.approx(
        labmath.confinterval([1, 2, 3], conf=0.5, error_only=True),
        0.47140452079103223,
    )


if __name__ == "__main__":
    test.Main()
