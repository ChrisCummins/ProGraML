#include "labm8/cpp/string.h"

#include "labm8/cpp/test.h"

namespace labm8 {
namespace {

TEST(TrimTest, Null) {
  string s;
  Trim(s);
  EXPECT_TRUE(s.empty());
}

TEST(TrimTest, EmptyString) {
  string s = "";
  Trim(s);
  EXPECT_EQ(s, "");
}

TEST(TrimTest, NoWhitespace) {
  string s = "hello";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, LeadingWhitespace) {
  string s = "  hello";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, TrailingWhitespace) {
  string s = "hello  ";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(TrimTest, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  Trim(s);
  EXPECT_EQ(s, "hello");
}

TEST(CopyAndTrimTest, Null) {
  string s;
  EXPECT_TRUE(CopyAndTrim(s).empty());
}

TEST(CopyAndTrimTest, EmptyString) {
  string s = "";
  EXPECT_EQ(CopyAndTrim(s), "");
}

TEST(CopyAndTrimTest, NoWhitespace) {
  string s = "hello";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, LeadingWhitespace) {
  string s = "  hello";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, TrailingWhitespace) {
  string s = "hello  ";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(CopyAndTrimTest, LeadingAndTrailingWhitespace) {
  string s = "  hello   ";
  EXPECT_EQ(CopyAndTrim(s), "hello");
}

TEST(ToCamelCase, EmptyString) { EXPECT_EQ(ToCamelCase(""), ""); }

TEST(ToCamelCase, Hello) {
  EXPECT_EQ(ToCamelCase("hello"), "Hello");
  EXPECT_EQ(ToCamelCase("Hello"), "Hello");
  EXPECT_EQ(ToCamelCase("HELLO"), "Hello");
}

TEST(ToCamelCase, LeadingWhitespace) { EXPECT_EQ(ToCamelCase("   hello"), "Hello"); }

TEST(ToCamelCase, TrailingWhitespace) { EXPECT_EQ(ToCamelCase("hello   "), "Hello"); }

TEST(ToCamelCase, MultipleComponents) { EXPECT_EQ(ToCamelCase("hello world"), "HelloWorld"); }

TEST(ToCamelCase, MultipleInnerWhitespace) {
  EXPECT_EQ(ToCamelCase("hello    world"), "HelloWorld");
}

TEST(ReplaceChar, Null) {
  string s;
  ReplaceChar(s, 'a', 'b');
  EXPECT_TRUE(s.empty());
}

TEST(ReplaceChar, EmptyString) {
  string s = "";
  ReplaceChar(s, 'a', 'b');
  EXPECT_EQ(s, "");
}

TEST(ReplaceChar, Occurrences) {
  string s = "abcdeabc";
  ReplaceChar(s, 'a', 'b');
  EXPECT_EQ(s, "bbcdebbc");
}

TEST(ReplaceChar, NoOccurrences) {
  string s = "bcdebc";
  ReplaceChar(s, 'a', 'b');
  EXPECT_EQ(s, "bcdebc");
}

TEST(CopyAndReplaceChar, EmptyString) { EXPECT_EQ(CopyAndReplaceChar("", 'a', 'b'), ""); }

TEST(CopyAndReplaceChar, Occurrences) {
  EXPECT_EQ(CopyAndReplaceChar("abcdeabc", 'a', 'b'), "bbcdebbc");
}

TEST(CopyAndReplaceChar, NoOccurrences) {
  EXPECT_EQ(CopyAndReplaceChar("bcdebc", 'a', 'b'), "bcdebc");
}

TEST(ReplaceSubstr, Null) {
  string s;
  ReplaceSubstr(s, "a", "b");
  EXPECT_TRUE(s.empty());
}

TEST(ReplaceSubstr, EmptyString) {
  string s = "";
  ReplaceSubstr(s, "a", "b");
  EXPECT_EQ(s, "");
}

TEST(ReplaceSubstr, Occurrences) {
  string s = "abcdeabc";
  ReplaceSubstr(s, "a", "b");
  EXPECT_EQ(s, "bbcdebbc");
}

TEST(ReplaceSubstr, NoOccurrences) {
  string s = "bcdebc";
  ReplaceSubstr(s, "a", "b");
  EXPECT_EQ(s, "bcdebc");
}

TEST(CopyAndReplaceSubstr, EmptyString) { EXPECT_EQ(CopyAndReplaceSubstr("", "a", "b"), ""); }

TEST(CopyAndReplaceSubstr, Occurrences) {
  EXPECT_EQ(CopyAndReplaceSubstr("abcdeabc", "a", "b"), "bbcdebbc");
}

TEST(CopyAndReplaceSubstr, NoOccurrences) {
  EXPECT_EQ(CopyAndReplaceSubstr("bcdebc", "a", "b"), "bcdebc");
}

TEST(TruncateWithEllipsis, EmptyString) {
  string s;
  TruncateWithEllipsis(s, 10);
  EXPECT_EQ(s, "");
}

}  // namespace
}  // namespace labm8

TEST_MAIN();
