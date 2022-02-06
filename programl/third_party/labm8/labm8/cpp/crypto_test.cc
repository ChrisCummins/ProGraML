#include "labm8/cpp/crypto.h"

#include <fstream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/test.h"

namespace fs = boost::filesystem;

namespace labm8 {
namespace crypto {
namespace {

class CryptoTest : public labm8::Test {
 public:
  CryptoTest()
      : missingFile_(labm8::GetTestTempdir() / "missing.txt"),
        emptyFile_(labm8::GetTestTempdir() / "empty.txt"),
        helloFile_(labm8::GetTestTempdir() / "hello.txt") {
    std::ofstream empty(emptyFile_.string());
    std::ofstream hello(helloFile_.string());
    (void)empty;
    hello << "hello\n";
  }

 protected:
  const fs::path missingFile_;
  const fs::path emptyFile_;
  const fs::path helloFile_;
};

TEST_F(CryptoTest, Sha1MissingFile) {
  auto ret = Sha1(missingFile_);
  ASSERT_FALSE(ret.ok());
}

TEST_F(CryptoTest, Sha1EmptyFile) {
  auto ret = Sha1(emptyFile_);

  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.ValueOrDie(), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
}

TEST_F(CryptoTest, Sha1HelloFile) {
  auto ret = Sha1(helloFile_);

  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.ValueOrDie(), "f572d396fae9206628714fb2ce00f72e94f2258f");
}

TEST_F(CryptoTest, Sha256MissingFile) {
  auto ret = Sha256(missingFile_);
  ASSERT_FALSE(ret.ok());
}

TEST_F(CryptoTest, Sha256EmptyFile) {
  auto ret = Sha256(emptyFile_);

  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.ValueOrDie(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_F(CryptoTest, Sha256HelloFile) {
  auto ret = Sha256(helloFile_);

  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.ValueOrDie(), "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03");
}

}  // namespace
}  // namespace crypto
}  // namespace labm8

TEST_MAIN();
