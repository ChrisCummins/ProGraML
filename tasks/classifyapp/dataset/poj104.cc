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
#include "tasks/classifyapp/dataset/poj104.h"

#include <stdio.h>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "boost/filesystem.hpp"
#include "ctpl.h"
#include "labm8/cpp/crypto.h"
#include "labm8/cpp/fsutil.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/statusor.h"
#include "programl/ir/llvm/clang.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"
#include "programl/util/filesystem_cache.h"
#include "subprocess/subprocess.hpp"
#include "tbb/parallel_for.h"

using labm8::Status;
namespace fs = boost::filesystem;

namespace programl {
namespace task {
namespace classifyapp {

static fs::path tempdir;

static ir::llvm::Clang compiler("-xc++ -std=c++11");

namespace {

void CleanUp() { fs::remove_all(tempdir); }

void PreprocessSrc(string* src) {
  // Clean up declaration of main function. Many are missing a return type
  // declaration, or use incorrect void return type.
  size_t n = src->find("void main");
  if (n != string::npos) {
    src->replace(n, 9, "int main ");
  }

  n = src->find("\nmain");
  if (n != string::npos) {
    src->insert(n + 1, "int ");
  }

  if (!src->compare(0, 4, "main")) {
    src->insert(0, "int ");
  }

  // Prepend a header with common includes and values.
  src->insert(0,
              "#include <cstdio>\n"
              "#include <cstdlib>\n"
              "#include <cmath>\n"
              "#include <cstring>\n"
              "#include <iostream>\n"
              "#include <algorithm>\n"
              "#define LEN 512\n"
              "#define MAX_LENGTH 512\n"
              "using namespace std;\n");
}

Status ProcessSourceFile(const fs::path& root, const fs::path& path, const fs::path& outpath,
                         const uint64_t srcId) {
  string fileContents;
  CHECK(labm8::fsutil::ReadFile(path, &fileContents).ok());
  string src = labm8::StripNonUtf8(fileContents);
  PreprocessSrc(&src);

  const string relpath = path.string().substr(root.string().size() + 1);

  // Files are organized by label.
  const string stringLabel = relpath.substr(0, relpath.find('/'));
  const int label = std::stoi(stringLabel);

  std::ofstream srcOut(
      absl::StrFormat("%s/src/%s.%d.SourceFile.pb", outpath.string(), stringLabel, srcId));
  {
    SourceFile srcMessage;
    srcMessage.set_relpath(relpath);
    srcMessage.set_language(SourceFile::CXX);
    srcMessage.set_text(src);
    srcMessage.SerializeToOstream(&srcOut);
  }

  IrList irs;
  RETURN_IF_ERROR(compiler.Compile(src, &irs));

  {
    std::ofstream irsOut(
        absl::StrFormat("%s/ir/%s.%d.IrList.pb", outpath.string(), stringLabel, srcId));
    irs.SerializeToOstream(&irsOut);
  }

  for (int i = 0; i < irs.ir_size(); ++i) {
    std::ofstream irOut(
        absl::StrFormat("%s/ir/%s.%d.%d.ll", outpath.string(), stringLabel, srcId, i));
    irOut << irs.ir(i).text();

    // Add program label.
    ProgramGraph graph;
    RETURN_IF_ERROR(ir::llvm::BuildProgramGraph(irs.ir(i).text(), &graph));
    Feature feature;
    feature.mutable_int64_list()->add_value(label);
    graph.mutable_features()->mutable_feature()->insert({"poj104_label", feature});

    std::ofstream graphOut(absl::StrFormat("%s/graphs/%s.%d.%d.ProgramGraph.pb", outpath.string(),
                                           stringLabel, srcId, i));
    graph.SerializeToOstream(&graphOut);
  }

  return Status::OK;
}

vector<fs::path> EnumerateFiles(const fs::path& root) {
  vector<fs::path> files;
  for (auto it : fs::recursive_directory_iterator(root)) {
    if (!fs::is_regular_file(it)) {
      continue;
    }
    files.push_back(it.path());
  }
  return files;
}

std::pair<int, int> GetLabelFromPath(const fs::path& path) {
  const size_t labelStart = path.string().rfind('/') + 1;
  const size_t labelEnd = path.string().find('.', labelStart);
  const size_t srcStart = labelEnd + 1;
  const size_t srcEnd = path.string().find('.', srcStart);
  const string label = path.string().substr(labelStart, labelEnd - labelStart);
  const string src = path.string().substr(srcStart, srcEnd - srcStart);
  return {std::stoi(label), std::stoi(src)};
}

absl::flat_hash_map<int, absl::flat_hash_map<int, vector<fs::path>>> EnumerateFilesByLabelAndSource(
    const fs::path& path) {
  const vector<fs::path> files = EnumerateFiles(path);

  absl::flat_hash_map<int, absl::flat_hash_map<int, vector<fs::path>>> grouped;
  for (const auto& path : files) {
    std::pair<int, int> v = GetLabelFromPath(path);
    grouped[v.first][v.second].push_back(path);
  }

  return grouped;
}

void SymlinkSources(const vector<fs::path>& src, const fs::path& dstDir) {
  for (const auto& path : src) {
    const fs::path dst = dstDir / path.filename();
    const fs::path rel = fs::relative(path, dstDir).string();
    fs::create_symlink(rel, dst);
  }
}

Status StratifiedTrainValTestSplit(const fs::path& path) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto random = std::default_random_engine(seed);

  const fs::path trainDir = path / "train";
  const fs::path valDir = path / "val";
  const fs::path testDir = path / "test";

  const auto files = EnumerateFilesByLabelAndSource(path / "graphs");

  for (const auto& labelsMap : files) {
    vector<vector<fs::path>> sources;
    for (const auto& srcMap : labelsMap.second) {
      sources.push_back(srcMap.second);
    }

    std::shuffle(sources.begin(), sources.end(), random);

    // 3:1:1 split of {train,val,test}.
    size_t total = sources.size();
    size_t train = static_cast<size_t>((total / 5.0) * 3);
    size_t val = static_cast<size_t>(total / 5.0);

    for (size_t i = 0; i < total; ++i) {
      if (i < train) {
        SymlinkSources(sources[i], trainDir);
      } else if (i < train + val) {
        SymlinkSources(sources[i], valDir);
      } else {
        SymlinkSources(sources[i], testDir);
      }
    }
  }

  return Status::OK;
}

}  // namespace

size_t CreatePoj104Dataset(const string& url, const fs::path& outputPath) {
  fs::create_directories(outputPath);
  fs::create_directory(outputPath / "src");
  fs::create_directory(outputPath / "ir");
  fs::create_directory(outputPath / "graphs");

  fs::create_directory(outputPath / "train");
  fs::create_directory(outputPath / "val");
  fs::create_directory(outputPath / "test");

  tempdir = fs::temp_directory_path() / fs::unique_path();
  CHECK(fs::create_directory(tempdir));
  std::atexit(CleanUp);

  util::FilesystemCache fileCache;

  const fs::path archive = fileCache[{"poj104.tar.gz"}];
  if (!fs::is_regular_file(archive)) {
    LOG(INFO) << "Downloading dataset from " << url << " ...";
    string wget = "wget '" + url + "' -O " + archive.string();
    CHECK(!system(wget.c_str())) << "failed: $ " << wget;
    CHECK(fs::is_regular_file(archive));
  }
  string sha1 = labm8::crypto::Sha1(archive).ValueOrDie();

  LOG(INFO) << "Extracting dataset archive ...";
  string tar = "tar -xf " + archive.string() + " -C " + tempdir.string();
  CHECK(!system(tar.c_str())) << "failed: $ " << tar;

  Repo repo;
  repo.set_url("https://sites.google.com/site/treebasedcnn/");
  repo.set_sha1(sha1);
  absl::CivilSecond ct(2014, 9, 18, 0, 0, 0);
  absl::Time created = absl::FromCivil(ct, absl::UTCTimeZone());
  repo.set_created_ms_timestamp(
      absl::ToInt64Milliseconds(absl::time_internal::ToUnixDuration(created)));
  {
    std::ofstream repoOut((outputPath / "src" / "Repo.pb").string());
    repo.SerializeToOstream(&repoOut);
  }

  fs::path root = tempdir / "ProgramData";
  CHECK(fs::is_directory(root));

  std::chrono::milliseconds startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());

  const vector<fs::path> files = EnumerateFiles(root);
  CHECK(files.size());
  size_t totalFiles = files.size();

  LOG(INFO) << "Processing " << totalFiles << " files ...";

  std::atomic_uint64_t fileCount{0};
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, files.size()), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t index = r.begin(); index != r.end(); ++index) {
          ProcessSourceFile(root, files[index], outputPath, index);
          ++fileCount;
          uint64_t f = fileCount;
          if (f && !(f % 8)) {
            std::chrono::milliseconds now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch());
            int msPerGraph = ((now - startTime) / f).count();
            std::cout << "\r\033[K" << f << " of " << totalFiles << " files processed ("
                      << msPerGraph << " ms / src, " << std::setprecision(3)
                      << (f / static_cast<float>(totalFiles)) * 100 << "%)" << std::flush;
          }
        }
      });
  std::cerr << "\n";

  LOG(INFO) << "Processed " << files.size() << " sources";

  LOG(INFO) << "Creating train/val/test symlinks";
  StratifiedTrainValTestSplit(outputPath);

  return files.size();
}

}  // namespace classifyapp
}  // namespace task
}  // namespace programl
