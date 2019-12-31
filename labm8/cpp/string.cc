#include "labm8/cpp/string.h"

#include <algorithm>
#include <boost/algorithm/string/replace.hpp>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace labm8 {

// Trim a string from the left in-place.
void TrimLeft(string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// Trim a string from the end in-place.
void TrimRight(string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// Trim a string from both ends in-place.
string &Trim(string &s) {
  TrimLeft(s);
  TrimRight(s);
  return s;
}

// Trim a string from the left.
string CopyAndTrimLeft(string s) {
  TrimLeft(s);
  return s;
}

// Trim a string from the right.
string CopyAndTrimRight(string s) {
  TrimRight(s);
  return s;
}

// Trim a string from both ends.
string CopyAndTrim(string s) {
  Trim(s);
  return s;
}

// Returns whether full_string ends with suffix.
bool EndsWith(const string &full_string, const string &suffix) {
  if (full_string.length() >= suffix.length()) {
    return full_string.compare(full_string.length() - suffix.length(),
                               suffix.length(), suffix) == 0;
  } else {
    return false;
  }
}

// Convert a string to CamelCase. E.g. "hello world" -> "HelloWorld".
string ToCamelCase(const string &full_string) {
  // Split string into a vector of space separated components.
  auto split_on_whitespace = absl::StrSplit(full_string, ' ');
  std::vector<string> space_separated_components;
  for (auto &s : split_on_whitespace)
    space_separated_components.push_back(string(s));

  string camel_case = "";
  for (auto component : space_separated_components) {
    // Convert starting letter to uppercase.
    component[0] = std::toupper(component[0]);
    // Convert all other letters to lowercase.
    for (size_t i = 1; i < component.size(); ++i) {
      component[i] = std::tolower(component[i]);
    }
    // Append component to output string.
    absl::StrAppend(&camel_case, component);
  }

  return camel_case;
}

string ReplaceChar(string &s, const char src, const char dst) {
  std::replace(s.begin(), s.end(), src, dst);
  return s;
}

string CopyAndReplaceChar(const string &s, const char src, const char dst) {
  string output_string(s);
  return ReplaceChar(output_string, src, dst);
}

string ReplaceSubstr(string &s, const string &src, const string &dst) {
  boost::replace_all(s, src, dst);
  return s;
}

string CopyAndReplaceSubstr(const string &s, const string &src,
                            const string &dst) {
  return boost::replace_all_copy(s, src, dst);
}

void TruncateWithEllipsis(string &s, const size_t &n) {
  if (n <= 3) {
    return;
  }

  if (s.size() > n) {
    s.resize(n);
    s[n - 1] = '.';
    s[n - 2] = '.';
    s[n - 3] = '.';
  }
}

}  // namespace labm8
