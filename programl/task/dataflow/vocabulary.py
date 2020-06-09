# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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
"""This module defines the logic for loading vocabularies from file."""
import csv
import pathlib

from labm8.py import app
from labm8.py import humanize


def LoadVocabulary(
  dataset_root: pathlib.Path,
  model_name: str,
  max_items: int = 0,
  target_cumfreq: float = 1.0,
):
  vocab_csv = dataset_root / "vocab" / f"{model_name}.csv"
  vocab = {}
  cumfreq = 0
  with open(vocab_csv) as f:
    vocab_file = csv.reader(f.readlines(), delimiter="\t")

    for i, row in enumerate(vocab_file, start=-1):
      if i == -1:  # Skip the header.
        continue
      (cumfreq, _, _, text) = row
      cumfreq = float(cumfreq)
      vocab[text] = i
      if cumfreq >= target_cumfreq:
        app.Log(2, "Reached target cumulative frequency: %.3f", target_cumfreq)
        break
      if max_items and i >= max_items - 1:
        app.Log(2, "Reached max vocab size: %d", max_items)
        break

  app.Log(
    1,
    "Selected %s-element vocabulary achieving %.2f%% node text coverage",
    humanize.Commas(len(vocab)),
    cumfreq * 100,
  )
  return vocab
