// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "labm8/cpp/fsutil.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/strutil.h"
#include "programl/proto/program_graph.pb.h"

const char* usage = R"(Create vocabulary files from node texts.

This reads the training ProgramGraphs and computes a frequency table of node
texts using inst2vec, ProGraML, and CDFG. The tables are then saved to file in
descending order of frequency. Each line is tab separated in the format:

  <cumulative textfreq> <cumulative nodefreq>  <count>  <node_text>

Where <cumulative text> is in the range [0, 1] and describes the propotion
of total node texts that are described by the current and prior lines.
<cumulative nodefreq> extends this to the proportion of total nodes, including
those without a text representation. <count> is the number of matching node
texts, and <node_text> is the unique text value.)";

DEFINE_string(path, (labm8::fsutil::GetHomeDirectoryOrDie() / "programl/dataflow").string(),
              "The directory to write generated files to.");
DEFINE_int32(limit, 0,
             "If --limit > 0, limit the number of input graphs processed to "
             "this number.");

namespace programl {
namespace task {
namespace dataflow {

namespace fs = boost::filesystem;
using absl::flat_hash_map;
using std::pair;
using std::set;
using std::vector;

vector<fs::path> EnumerateProgramGraphFiles(const fs::path& root) {
  vector<fs::path> files;
  for (auto it : fs::directory_iterator(root)) {
    if (labm8::HasSuffixString(it.path().string(), ".ProgramGraph.pb")) {
      files.push_back(it.path());
    }
  }
  return files;
}

inline std::chrono::milliseconds Now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
}

inline void PrintProgess(const std::chrono::milliseconds& startTime, size_t i, size_t n) {
  std::chrono::milliseconds now = Now();
  int msPerGraph = ((now - startTime) / i).count();
  std::cout << "\r\033[K" << i << " of " << n << " files processed (" << msPerGraph
            << " ms / file, " << std::setprecision(3) << (i / static_cast<float>(n)) * 100 << "%)"
            << std::flush;
}

template <typename T>
void SerializeFrequencyTable(const flat_hash_map<T, size_t>& freq, size_t totalNodeCount,
                             std::ostream* ostream) {
  size_t totalCount = 0;
  for (const auto& it : freq) {
    totalCount += it.second;
  }

  // Flatten to a set, ordered by decreasing frequency.
  typedef std::function<bool(pair<T, size_t>, pair<T, size_t>)> Comparator;
  Comparator sortBy = [](pair<T, size_t> elem1, pair<T, size_t> elem2) {
    return elem1.second > elem2.second;
  };
  set<pair<T, int>, Comparator> ordered(freq.begin(), freq.end(), sortBy);

  (*ostream) << "cumulative_frequency" << '\t' << "cumulative_node_frequency" << '\t' << "count"
             << '\t' << "text" << std::endl;

  size_t cumCount = 0;
  for (const auto& it : ordered) {
    cumCount += it.second;
    (*ostream) << std::setprecision(4) << (cumCount / static_cast<double>(totalCount)) << '\t'
               << (cumCount / static_cast<double>(totalNodeCount)) << '\t' << it.second << '\t'
               << it.first << std::endl;
  }
}

template <typename T>
void SerializeFrequencyTable(const flat_hash_map<T, size_t>& freq, size_t totalNodeCount,
                             const string& path) {
  std::ofstream out(path);
  SerializeFrequencyTable(freq, totalNodeCount, &out);
  std::cout << "Wrote vocab with " << freq.size() << " elements to " << path << std::endl;
}

void CreateVocabularyFiles(const fs::path& root) {
  std::chrono::milliseconds startTime = Now();
  const vector<fs::path> graphs =
      programl::task::dataflow::EnumerateProgramGraphFiles(root / "graphs");
  size_t totalNodeCount = 0;

  // Frequency tables.
  flat_hash_map<string, size_t> inst2vecPreprocessed;
  flat_hash_map<int64_t, size_t> inst2vec;
  flat_hash_map<string, size_t> programl;
  flat_hash_map<string, size_t> cdfg;

  const size_t n =
      FLAGS_limit ? std::min(size_t(graphs.size()), size_t(FLAGS_limit)) : graphs.size();

  for (size_t i = 0; i < n; ++i) {
    ProgramGraph graph;
    std::ifstream graphFile(graphs[i].string());
    if (!graph.ParseFromIstream(&graphFile)) {
      LOG(ERROR) << "Corrupt file: " << graphs[i].string();
    }

    totalNodeCount += graph.node_size();

    for (int j = 0; j < graph.node_size(); ++j) {
      const auto& node = graph.node(j);

      // inst2vecPreprocessed
      const auto& pp = node.features().feature().find("inst2vec_preprocessed");
      if (pp != node.features().feature().end()) {
        inst2vecPreprocessed[(*pp).second.bytes_list().value(0)] += 1;
      }
      // inst2vec
      const auto& emb = node.features().feature().find("inst2vec_embedding");
      if (emb != node.features().feature().end()) {
        inst2vec[(*emb).second.int64_list().value(0)] += 1;
      }
      // ProGraML
      ++programl[node.text()];
      // CDFG
      if (node.type() == Node::INSTRUCTION) {
        ++cdfg[node.text()];
      }

      if (i && !(i % 16)) {
        PrintProgess(startTime, i, n);
      }
    }
  }

  PrintProgess(startTime, n, n);
  std::cout << std::endl;

  fs::create_directory(root / "vocab");
  SerializeFrequencyTable(inst2vecPreprocessed, totalNodeCount,
                          (root / "vocab" / "inst2vec_preprocessed.csv").string());
  SerializeFrequencyTable(inst2vec, totalNodeCount, (root / "vocab" / "inst2vec.csv").string());
  SerializeFrequencyTable(programl, totalNodeCount, (root / "vocab" / "programl.csv").string());
  SerializeFrequencyTable(cdfg, totalNodeCount, (root / "vocab" / "cdfg.csv").string());
}

}  // namespace dataflow
}  // namespace task
}  // namespace programl

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);
  if (argc > 1) {
    std::cerr << "fatal: Unrecognized arguments" << std::endl;
    return 4;
  }

  const fs::path path(FLAGS_path);
  programl::task::dataflow::CreateVocabularyFiles(path);
  LOG(INFO) << "done";

  return 0;
}
