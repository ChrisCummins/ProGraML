#include "labm8/cpp/fsutil.h"

#include <fstream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/test.h"

namespace fs = boost::filesystem;

namespace labm8 {
namespace fsutil {

class FsutilTest : public Test {};

TEST_F(FsutilTest, GetHomeDirectoryOrDie) {
  fs::path home = GetHomeDirectoryOrDie();
  ASSERT_TRUE(fs::is_directory(home));
}

TEST_F(FsutilTest, ReadFile) {
  fs::path path = (GetTempDir() / "file.txt");
  {
    std::ofstream out(path.string());
    out << "Hello, world";
  }

  string contents;
  Status status = ReadFile(path, &contents);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(contents, "Hello, world");
}

}  // namespace fsutil
}  // namespace labm8

TEST_MAIN();
