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
"""Unit tests for //labm8/py:humanize."""
import datetime

import pytest
from labm8.py import app, humanize, labdate, test

FLAGS = app.FLAGS

# Adapted from Google's unittests.
# Copyright 2008 Google Inc. All Rights Reserved.
# <https://github.com/google/google-apputils>


def test_Commas():
    assert "0" == humanize.Commas(0)
    assert "100" == humanize.Commas(100)
    assert "1,000" == humanize.Commas(1000)
    assert "10,000" == humanize.Commas(10000)
    assert "1,000,000" == humanize.Commas(1e6)
    assert "-1,000,000" == humanize.Commas(-1e6)


def test_Plural():
    assert "0 objects" == humanize.Plural(0, "object")
    assert "1 object" == humanize.Plural(1, "object")
    assert "-1 objects" == humanize.Plural(-1, "object")
    assert "42 objects" == humanize.Plural(42, "object")
    assert "42 cats" == humanize.Plural(42, "cat")
    assert "42 glasses" == humanize.Plural(42, "glass")
    assert "42 potatoes" == humanize.Plural(42, "potato")
    assert "42 cherries" == humanize.Plural(42, "cherry")
    assert "42 monkeys" == humanize.Plural(42, "monkey")
    assert "42 oxen" == humanize.Plural(42, "ox", "oxen")
    assert "42 indices" == humanize.Plural(42, "index")
    assert "42 attorneys general" == humanize.Plural(
        42,
        "attorney general",
        "attorneys general",
    )


def test_PluralWord():
    assert "vaxen" == humanize.PluralWord(2, "vax", plural="vaxen")
    assert "cores" == humanize.PluralWord(2, "core")
    assert "group" == humanize.PluralWord(1, "group")
    assert "cells" == humanize.PluralWord(0, "cell")
    assert "degree" == humanize.PluralWord(1.0, "degree")
    assert "helloes" == humanize.PluralWord(3.14, "hello")


def test_WordSeries():
    assert "" == humanize.WordSeries([])
    assert "foo" == humanize.WordSeries(["foo"])
    assert "foo and bar" == humanize.WordSeries(["foo", "bar"])
    assert "foo, bar, and baz" == humanize.WordSeries(["foo", "bar", "baz"])
    assert "foo, bar, or baz" == humanize.WordSeries(
        ["foo", "bar", "baz"],
        conjunction="or",
    )


def test_AddIndefiniteArticle():
    assert "a thing" == humanize.AddIndefiniteArticle("thing")
    assert "an object" == humanize.AddIndefiniteArticle("object")
    assert "a Porsche" == humanize.AddIndefiniteArticle("Porsche")
    assert "an Audi" == humanize.AddIndefiniteArticle("Audi")


def test_DecimalPrefix():
    assert "0 m" == humanize.DecimalPrefix(0, "m")
    assert "1 km" == humanize.DecimalPrefix(1000, "m")
    assert "-1 km" == humanize.DecimalPrefix(-1000, "m")
    assert "10 Gbps" == humanize.DecimalPrefix(10e9, "bps")
    assert "6000 Yg" == humanize.DecimalPrefix(6e27, "g")
    assert "12.1 km" == humanize.DecimalPrefix(12100, "m", precision=3)
    assert "12 km" == humanize.DecimalPrefix(12100, "m", precision=2)
    assert "1.15 km" == humanize.DecimalPrefix(1150, "m", precision=3)
    assert "-1.15 km" == humanize.DecimalPrefix(-1150, "m", precision=3)
    assert "1 k" == humanize.DecimalPrefix(1000, "")
    assert "-10 G" == humanize.DecimalPrefix(-10e9, "")
    assert "12" == humanize.DecimalPrefix(12, "")
    assert "-115" == humanize.DecimalPrefix(-115, "")
    assert "0" == humanize.DecimalPrefix(0, "")

    assert "1.1 s" == humanize.DecimalPrefix(1.12, "s", precision=2)
    assert "-1.1 s" == humanize.DecimalPrefix(-1.12, "s", precision=2)
    assert "nan bps" == humanize.DecimalPrefix(float("nan"), "bps")
    assert "nan" == humanize.DecimalPrefix(float("nan"), "")
    assert "inf bps" == humanize.DecimalPrefix(float("inf"), "bps")
    assert "-inf bps" == humanize.DecimalPrefix(float("-inf"), "bps")
    assert "-inf" == humanize.DecimalPrefix(float("-inf"), "")

    assert "-4 mm" == humanize.DecimalPrefix(-0.004, "m", min_scale=None)
    assert "0 m" == humanize.DecimalPrefix(0, "m", min_scale=None)
    assert "1 µs" == humanize.DecimalPrefix(0.0000013, "s", min_scale=None)
    assert "3 km" == humanize.DecimalPrefix(3000, "m", min_scale=None)
    assert "5000 TB" == humanize.DecimalPrefix(5e15, "B", max_scale=4)
    assert "5 mSWE" == humanize.DecimalPrefix(0.005, "SWE", min_scale=None)
    assert "0.0005 ms" == humanize.DecimalPrefix(
        5e-7,
        "s",
        min_scale=-1,
        precision=2,
    )


def test_BinaryPrefix():
    assert "0 B" == humanize.BinaryPrefix(0, "B")
    assert "1000 B" == humanize.BinaryPrefix(1000, "B")
    assert "1 KiB" == humanize.BinaryPrefix(1024, "B")
    assert "64 GiB" == humanize.BinaryPrefix(2 ** 36, "B")
    assert "65536 Yibit" == humanize.BinaryPrefix(2 ** 96, "bit")
    assert "1.25 KiB" == humanize.BinaryPrefix(1280, "B", precision=3)
    assert "1.2 KiB" == humanize.BinaryPrefix(1280, "B", precision=2)
    assert "1.2 Ki" == humanize.BinaryPrefix(1280, "", precision=2)
    assert "12" == humanize.BinaryPrefix(12, "", precision=2)
    assert "10.0 QPS" == humanize.BinaryPrefix(10, "QPS", precision=3)


def test_DecimalScale():
    assert isinstance(humanize.DecimalScale(0, "")[0], float)
    assert isinstance(humanize.DecimalScale(1, "")[0], float)
    assert (12.1, "km") == humanize.DecimalScale(12100, "m")
    assert (12.1, "k") == humanize.DecimalScale(12100, "")
    assert (0, "") == humanize.DecimalScale(0, "")
    assert (12.1, "km") == humanize.DecimalScale(
        12100,
        "m",
        min_scale=0,
        max_scale=None,
    )
    assert (12100, "m") == humanize.DecimalScale(
        12100,
        "m",
        min_scale=0,
        max_scale=0,
    )
    assert (1.15, "Mm") == humanize.DecimalScale(1150000, "m")
    assert (1, "m") == humanize.DecimalScale(1, "m", min_scale=None)
    assert (450, "mSWE") == humanize.DecimalScale(0.45, "SWE", min_scale=None)
    assert (250, "µm") == humanize.DecimalScale(
        1.0 / (4 * 1000),
        "m",
        min_scale=None,
    )
    assert (0.250, "km") == humanize.DecimalScale(250, "m", min_scale=1)
    assert (12000, "mm") == humanize.DecimalScale(
        12,
        "m",
        min_scale=None,
        max_scale=-1,
    )


def test_BinaryScale():
    assert isinstance(humanize.BinaryScale(0, "")[0], float)
    assert isinstance(humanize.BinaryScale(1, "")[0], float)
    value, unit = humanize.BinaryScale(200000000000, "B")
    assert value == pytest.approx(186.26, 2)
    assert unit == "GiB"

    value, unit = humanize.BinaryScale(3000000000000, "B")
    assert value == pytest.approx(2.728, 3)
    assert unit == "TiB"


def test_PrettyFraction():
    # No rounded integer part
    assert "½" == humanize.PrettyFraction(0.5)
    # Roundeded integer + fraction
    assert "6⅔" == humanize.PrettyFraction(20.0 / 3.0)
    # Rounded integer, no fraction
    assert "2" == humanize.PrettyFraction(2.00001)
    # No rounded integer, no fraction
    assert "0" == humanize.PrettyFraction(0.001)
    # Round up
    assert "1" == humanize.PrettyFraction(0.99)
    # No round up, edge case
    assert "⅞" == humanize.PrettyFraction(0.9)
    # Negative fraction
    assert "-⅕" == humanize.PrettyFraction(-0.2)
    # Negative close to zero (should not be -0)
    assert "0" == humanize.PrettyFraction(-0.001)
    # Smallest fraction that should round down.
    assert "0" == humanize.PrettyFraction(1.0 / 16.0)
    # Largest fraction should round up.
    assert "1" == humanize.PrettyFraction(15.0 / 16.0)
    # Integer zero.
    assert "0" == humanize.PrettyFraction(0)
    # Check that division yields fraction
    assert "⅘" == humanize.PrettyFraction(4.0 / 5.0)
    # Custom spacer.
    assert "2 ½" == humanize.PrettyFraction(2.5, spacer=" ")


def test_Duration():
    assert "2h" == humanize.Duration(7200)
    assert "5d 13h 47m 12s" == humanize.Duration(481632)
    assert "0s" == humanize.Duration(0)
    assert "59s" == humanize.Duration(59)
    assert "1m" == humanize.Duration(60)
    assert "1m 1s" == humanize.Duration(61)
    assert "1h 1s" == humanize.Duration(3601)
    assert "2h-2s" == humanize.Duration(7202, separator="-")


def test_FloatDuration():
    assert "18s" == humanize.Duration(18.0)
    assert "530ms" == humanize.Duration(0.53)
    assert "1ms" == humanize.Duration(0.001)
    assert "1s 10ms" == humanize.Duration(1.01)
    assert "1h 500ms" == humanize.Duration(3600.5)
    assert "1ms" == humanize.Duration(1e-3)
    assert "100µs" == humanize.Duration(1e-4)
    assert "100ns" == humanize.Duration(1e-7)
    # Anything below one nanosecond is rounded away.
    assert "<1ns" == humanize.Duration(1e-10)
    assert "1s" == humanize.Duration(1 + 1e-10)
    assert "1s" == humanize.Duration(datetime.timedelta(seconds=1))


def test_LargeDuration():
    # The maximum seconds and days that can be stored in a datetime.timedelta
    # object, as seconds.  max_days is equal to MAX_DELTA_DAYS in Python's
    # Modules/datetimemodule.c, converted to seconds.
    max_seconds = 3600 * 24 - 1
    max_days = 999999999 * 24 * 60 * 60

    assert "999999999d" == humanize.Duration(max_days)
    assert "999999999d 23h 59m 59s" == humanize.Duration(max_days + max_seconds)
    assert ">=999999999d 23h 59m 60s" == humanize.Duration(
        max_days + max_seconds + 1,
    )


def test_TimeDelta():
    assert "0s" == humanize.TimeDelta(datetime.timedelta())
    assert "2h" == humanize.TimeDelta(datetime.timedelta(hours=2))
    assert "1m" == humanize.TimeDelta(datetime.timedelta(minutes=1))
    assert "5d" == humanize.TimeDelta(datetime.timedelta(days=5))
    assert "1.25s" == humanize.TimeDelta(
        datetime.timedelta(seconds=1, microseconds=250000),
    )
    assert "1.5s" == humanize.TimeDelta(datetime.timedelta(seconds=1.5))
    assert "4d 10h 5m 12.25s" == humanize.TimeDelta(
        datetime.timedelta(
            days=4,
            hours=10,
            minutes=5,
            seconds=12,
            microseconds=250000,
        ),
    )


def test_UnixTimestamp():
    assert "2013-11-17 11:08:27.723524 PST" == humanize.UnixTimestamp(
        1384715307.723524,
        labdate.US_PACIFIC,
    )
    assert "2013-11-17 19:08:27.723524 UTC" == humanize.UnixTimestamp(
        1384715307.723524,
        labdate.UTC,
    )

    # DST part of the timezone should not depend on the current local time,
    # so this should be in PDT (and different from the PST in the first test).
    assert "2013-05-17 15:47:21.723524 PDT" == humanize.UnixTimestamp(
        1368830841.723524,
        labdate.US_PACIFIC,
    )

    assert "1970-01-01 00:00:00.000000 UTC" == humanize.UnixTimestamp(
        0,
        labdate.UTC,
    )


def test_AddOrdinalSuffix():
    assert "0th" == humanize.AddOrdinalSuffix(0)
    assert "1st" == humanize.AddOrdinalSuffix(1)
    assert "2nd" == humanize.AddOrdinalSuffix(2)
    assert "3rd" == humanize.AddOrdinalSuffix(3)
    assert "4th" == humanize.AddOrdinalSuffix(4)
    assert "5th" == humanize.AddOrdinalSuffix(5)
    assert "10th" == humanize.AddOrdinalSuffix(10)
    assert "11th" == humanize.AddOrdinalSuffix(11)
    assert "12th" == humanize.AddOrdinalSuffix(12)
    assert "13th" == humanize.AddOrdinalSuffix(13)
    assert "14th" == humanize.AddOrdinalSuffix(14)
    assert "20th" == humanize.AddOrdinalSuffix(20)
    assert "21st" == humanize.AddOrdinalSuffix(21)
    assert "22nd" == humanize.AddOrdinalSuffix(22)
    assert "23rd" == humanize.AddOrdinalSuffix(23)
    assert "24th" == humanize.AddOrdinalSuffix(24)
    assert "63rd" == humanize.AddOrdinalSuffix(63)
    assert "100000th", humanize.AddOrdinalSuffix(100000)
    assert "100001st", humanize.AddOrdinalSuffix(100001)
    assert "100011th", humanize.AddOrdinalSuffix(100011)
    with test.Raises(ValueError):
        humanize.AddOrdinalSuffix(-1)
    with test.Raises(ValueError):
        humanize.AddOrdinalSuffix(0.5)
    with test.Raises(ValueError):
        humanize.AddOrdinalSuffix(123.001)


def test_NaturalSortKeySimpleWords():
    test = ["pair", "banana", "apple"]
    good = ["apple", "banana", "pair"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeySimpleNums():
    test = ["3333", "2222", "9999", "0000"]
    good = ["0000", "2222", "3333", "9999"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeySimpleDigits():
    test = ["8", "3", "2"]
    good = ["2", "3", "8"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_VersionStrings():
    test = ["1.2", "0.9", "1.1a2", "1.1a", "1", "1.2.1", "0.9.1"]
    good = ["0.9", "0.9.1", "1", "1.1a", "1.1a2", "1.2", "1.2.1"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeySimpleNumLong():
    test = ["11", "9", "1", "200", "19", "20", "900"]
    good = ["1", "9", "11", "19", "20", "200", "900"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeyAlNum():
    test = ["x10", "x9", "x1", "x11"]
    good = ["x1", "x9", "x10", "x11"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeyNumAlNum():
    test = ["4x10", "4x9", "4x11", "5yy4", "3x1", "2x11"]
    good = ["2x11", "3x1", "4x9", "4x10", "4x11", "5yy4"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeyAlNumAl():
    test = ["a9c", "a4b", "a10c", "a1c", "c10c", "c10a", "c9a"]
    good = ["a1c", "a4b", "a9c", "a10c", "c9a", "c10a", "c10c"]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


def test_NaturalSortKeyBigTest_Big():
    test = [
        "1000X Radonius Maximus",
        "10X Radonius",
        "200X Radonius",
        "20X Radonius",
        "20X Radonius Prime",
        "30X Radonius",
        "40X Radonius",
        "Allegia 50 Clasteron",
        "Allegia 500 Clasteron",
        "Allegia 51 Clasteron",
        "Allegia 51B Clasteron",
        "Allegia 52 Clasteron",
        "Allegia 60 Clasteron",
        "Alpha 100",
        "Alpha 2",
        "Alpha 200",
        "Alpha 2A",
        "Alpha 2A-8000",
        "Alpha 2A-900",
        "Callisto Morphamax",
        "Callisto Morphamax 500",
        "Callisto Morphamax 5000",
        "Callisto Morphamax 600",
        "Callisto Morphamax 700",
        "Callisto Morphamax 7000",
        "Callisto Morphamax 7000 SE",
        "Callisto Morphamax 7000 SE2",
        "QRS-60 Intrinsia Machine",
        "QRS-60F Intrinsia Machine",
        "QRS-62 Intrinsia Machine",
        "QRS-62F Intrinsia Machine",
        "Xiph Xlater 10000",
        "Xiph Xlater 2000",
        "Xiph Xlater 300",
        "Xiph Xlater 40",
        "Xiph Xlater 5",
        "Xiph Xlater 50",
        "Xiph Xlater 500",
        "Xiph Xlater 5000",
        "Xiph Xlater 58",
    ]
    good = [
        "10X Radonius",
        "20X Radonius",
        "20X Radonius Prime",
        "30X Radonius",
        "40X Radonius",
        "200X Radonius",
        "1000X Radonius Maximus",
        "Allegia 50 Clasteron",
        "Allegia 51 Clasteron",
        "Allegia 51B Clasteron",
        "Allegia 52 Clasteron",
        "Allegia 60 Clasteron",
        "Allegia 500 Clasteron",
        "Alpha 2",
        "Alpha 2A",
        "Alpha 2A-900",
        "Alpha 2A-8000",
        "Alpha 100",
        "Alpha 200",
        "Callisto Morphamax",
        "Callisto Morphamax 500",
        "Callisto Morphamax 600",
        "Callisto Morphamax 700",
        "Callisto Morphamax 5000",
        "Callisto Morphamax 7000",
        "Callisto Morphamax 7000 SE",
        "Callisto Morphamax 7000 SE2",
        "QRS-60 Intrinsia Machine",
        "QRS-60F Intrinsia Machine",
        "QRS-62 Intrinsia Machine",
        "QRS-62F Intrinsia Machine",
        "Xiph Xlater 5",
        "Xiph Xlater 40",
        "Xiph Xlater 50",
        "Xiph Xlater 58",
        "Xiph Xlater 300",
        "Xiph Xlater 500",
        "Xiph Xlater 2000",
        "Xiph Xlater 5000",
        "Xiph Xlater 10000",
    ]
    test.sort(key=humanize.NaturalSortKey)
    assert test == good


if __name__ == "__main__":
    test.Main()
