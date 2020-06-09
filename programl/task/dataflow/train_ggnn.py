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
"""Train a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib

from labm8.py import app
from programl.task.dataflow import ggnn
from programl.task.dataflow import vocabulary

app.DEFINE_integer(
  "batch_size",
  50000,
  "The number of nodes in a graph. "
  "On our system, we observed that a batch size of 50,000 nodes requires "
  "about 5.2GB of GPU VRAM.",
)
app.DEFINE_boolean(
  "limit_max_data_flow_steps",
  True,
  "If set, limit the size of dataflow-annotated graphs used to only those with "
  "data_flow_steps <= message_passing_step_count",
)
app.DEFINE_boolean(
  "cdfg",
  False,
  "If set, use the CDFG representation for programs. Defaults to ProGraML "
  "representations.",
)
app.DEFINE_integer(
  "max_vocab_size",
  0,
  "If > 0, limit the size of the vocabulary to this number.",
)
app.DEFINE_float(
  "target_vocab_cumfreq", 1.0, "The target cumulative frequency that."
)
FLAGS = app.FLAGS


def Main():
  """Main entry point."""

  path = pathlib.Path(FLAGS.path)

  vocab = vocabulary.LoadVocabulary(
    path,
    model_name="cdfg" if FLAGS.cdfg else "programl",
    max_items=FLAGS.max_vocab_size,
    target_cumfreq=FLAGS.target_vocab_cumfreq,
  )

  # CDFG doesn't use positional embeddings.
  if FLAGS.cdfg:
    FLAGS.use_position_embeddings = False

  if FLAGS.test_only:
    log_dir = FLAGS.restore_from
  else:
    log_dir = ggnn.TrainDataflowGGNN(
      path=path,
      analysis=FLAGS.analysis,
      vocab=vocab,
      limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
      train_graph_counts=[int(x) for x in FLAGS.train_graph_counts],
      val_graph_count=FLAGS.val_graph_count,
      val_seed=FLAGS.val_seed,
      batch_size=FLAGS.batch_size,
      use_cdfg=FLAGS.cdfg,
      run_id=FLAGS.run_id,
      restore_from=FLAGS.restore_from,
    )

  if FLAGS.test:
    ggnn.TestDataflowGGNN(
      path=path,
      log_dir=log_dir,
      analysis=FLAGS.analysis,
      vocab=vocab,
      limit_max_data_flow_steps=FLAGS.limit_max_data_flow_steps,
      batch_size=FLAGS.batch_size,
      use_cdfg=FLAGS.cdfg,
    )


if __name__ == "__main__":
  app.Run(Main)
