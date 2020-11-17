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
"""This module defines a class for monitoring whole-system performance.

When run as a script, this monitors performance prints to stdout every
--perf_monitor_frequency seconds.
"""
import json
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep, strftime, time

import GPUtil
import numpy as np
import psutil
from absl import app, flags, logging

flags.DEFINE_float(
    "perf_monitor_frequency",
    10,
    "The number of seconds between updates to performance monitor",
)
FLAGS = flags.FLAGS


class PerformanceMonitor(Thread):
    """A class for monitoring system performance in a background thread.

    An instance of this class runs in a separate thread, passively recording
    system stats at a fixed interval frequency and updating rolling averages.
    These stats can be printed or saved to file whenever as you please.

    This class can be used in a 'with' context, for example:

        with PerformanceMonitor(frequency=2) as perf:
          # ... do heavy work
        print(perf.stats)

    Or by manually calling Stop() when you are done:

        perf = PerforamnceMonitor():
        # ... do work
        print(perf.stats)
        perf.Stop()

    Once stopped, a PerformanceMonitor can not be restarted.

    At every set frequency interval, the rolling averages are computed. To reset
    the rolling averages, call Reset().

    To passively record or log stats, a callback function (or list of callback
    functions) can be provided which will be called after every observation:

        def SaveStatsAndReset(perf):
          # For every 10 observations, write stats to file and reset averages.
          if not perf.observation_count % 10:
            with open(f'/tmp/perf_log_{int(time())}', 'w' as f:
              json.dump(perf.stats)
            perf.Reset()

        callbacks = [
          lambda p: print(p.stats),
          SaveStats,
        ]

        perf = PerformanceMonitor(on_observation=callbacks)
        with perf:
          # ... do heavy work
    """

    def __init__(self, frequency: int = None, on_observation=None):
        """Constructor.

        Args:
          frequency: The period of time to wait between recording observations,
            in seconds. A lower frequency will result in more accurate estimates of
            statistic averages, at the expense of higher cost. Memory overhead is
            constant with frequency.
          on_observation: A callback, or list of callbacks, which are called after
            every observation. Each callback takes a single argument, the instance
            of this class.
        """
        super(PerformanceMonitor, self).__init__()
        self.observation_frequency = frequency or FLAGS.perf_monitor_frequency
        self.on_observation = on_observation or []

        self.stopped = False

        # Initialized in Reset().
        self.stats = None
        self.prev_disk_counters = None
        self.prev_net_counters = None
        self.last_record_time = None

        self.Reset()
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Stop()

    @property
    def observation_count(self) -> int:
        return self.stats["observation_count"]

    def Reset(self) -> None:
        """"Reset the rolling average counters."""
        self.stats = {
            "observation_count": 0,
            "observation_frequency_sec": self.observation_frequency,
        }
        for _ in GPUtil.getGPUs():
            self.stats.get("gpus", []).append({})
        self.prev_disk_counters = psutil.disk_io_counters(perdisk=False)
        self.prev_net_counters = psutil.net_io_counters(pernic=False)
        self.last_record_time = time()

    def Update(self, key, value, default=0, data=None):
        """Update a rolling average.

        Args:
          key: The name of the metric being updated.
          value: The newly observed value.
          default: The default value for the first observation.
          data: The data dictionary to update.
        """
        data = data or self.stats
        current_average = data.get(key, default)
        data[key] = (self.observation_count * current_average + value) / (
            self.observation_count + 1
        )

    def MakeObservation(self) -> None:
        """Record a new observation and update internal state."""
        # CPU.
        cpu_loads = np.array(psutil.cpu_percent(percpu=True)) / 100
        self.Update("cpu_load", np.average(cpu_loads))
        self.Update("cpu_load_max", np.max(cpu_loads))
        self.Update("cpu_freq_mhz", psutil.cpu_freq().current)

        # Memory.
        self.Update("memory_util", psutil.virtual_memory().percent / 100)
        self.Update("swap_util", psutil.swap_memory().percent / 100)

        # Counter-based stats.
        elapsed = time() - self.last_record_time
        disk_counters = psutil.disk_io_counters(perdisk=False)
        net_counters = psutil.net_io_counters(pernic=False)

        # Disk counters.
        self.Update(
            "disk_reads_per_sec",
            (disk_counters.read_count - self.prev_disk_counters.read_count) / elapsed,
        )
        self.Update(
            "disk_writes_per_sec",
            (disk_counters.write_count - self.prev_disk_counters.write_count) / elapsed,
        )

        self.Update(
            "disk_read_mb_per_sec",
            (
                (disk_counters.read_bytes - self.prev_disk_counters.read_bytes)
                / (1024 * 1024)
            )
            / elapsed,
        )
        self.Update(
            "disk_write_mb_per_sec",
            (
                (disk_counters.write_bytes - self.prev_disk_counters.write_bytes)
                / (1024 * 1024)
            )
            / elapsed,
        )

        # Network counters.
        self.Update(
            "net_packets_recv_per_sec",
            (net_counters.packets_recv - self.prev_net_counters.packets_recv) / elapsed,
        )
        self.Update(
            "net_packets_sent_per_sec",
            (net_counters.packets_sent - self.prev_net_counters.packets_sent) / elapsed,
        )

        self.Update(
            "net_data_recv_mb_per_sec",
            (
                (net_counters.bytes_recv - self.prev_net_counters.bytes_recv)
                / (1024 * 1024)
            )
            / elapsed,
        )
        self.Update(
            "net_data_sent_mb_per_sec",
            (
                (net_counters.bytes_sent - self.prev_net_counters.bytes_sent)
                / (1024 * 1024)
            )
            / elapsed,
        )

        # Update counters.
        self.last_record_time = time()
        self.prev_disk_counters = disk_counters
        self.prev_net_counters = net_counters

        # GPU stats.
        for gpu_data, gpu in zip(self.stats.get("gpus", []), GPUtil.getGPUs()):
            self.Update("load", gpu.load, data=gpu_data)
            self.Update("memory_util", gpu.memoryUtil, data=gpu_data)
            self.Update("temperature", gpu.temperature, data=gpu_data)

        self.stats["observation_count"] += 1

        # Call the user-provided callback, or list of callbacks.
        if callable(self.on_observation):
            self.on_observation(self)
        else:
            for callback in self.on_observation:
                callback(self)

    def run(self):
        """Thread loop. Terminate by calling Stop()."""
        sleep(self.observation_frequency)
        while not self.stopped:
            self.MakeObservation()
            sleep(self.observation_frequency)

    def WriteJson(self, path: Path):
        """Save monitor stats to file as JSON data."""
        with open(str(path), "w") as f:
            json.dump(self.stats, f, indent=2, sort_keys=True)

    def Stop(self):
        """Stop performance monitor. Returns immediately."""
        self.stopped = True


def PrintToStdoutCallback(monitor: PerformanceMonitor):
    print(
        datetime.now(),
        ": ",
        json.dumps(monitor.stats, indent=2, sort_keys=True),
        sep="",
    )


def WriteJsonToFileCallback(
    json_dir: Path,
    every: int = 1,
    reset=False,
    filename_format: str = "%y:%m:%dT%H:%M:%S.json",
):
    def _WriteJson(monitor: PerformanceMonitor):
        if monitor.observation_count % every:
            return
        outpath = json_dir / strftime(filename_format)
        monitor.WriteJson(outpath)
        logging.info("Wrote performance log to %s", outpath)
        if reset:
            monitor.Reset()

    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    return _WriteJson


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    PerformanceMonitor(on_observation=PrintToStdoutCallback)


if __name__ == "__main__":
    app.run(main)
