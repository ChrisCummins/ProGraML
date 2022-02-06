// Adapted from Google's protocol buffer sources. Original comment header:
//
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "labm8/cpp/status.h"

#include <stdio.h>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

namespace labm8 {
namespace {

TEST(Status, Empty) {
  labm8::Status status;
  EXPECT_EQ(labm8::error::OK, labm8::Status::OK.error_code());
  EXPECT_EQ(labm8::error::OK, labm8::Status::OK.code());
  EXPECT_EQ("OK", labm8::Status::OK.ToString());
}

TEST(Status, GenericCodes) {
  EXPECT_EQ(labm8::error::OK, labm8::Status::OK.error_code());
  EXPECT_EQ(labm8::error::OK, labm8::Status::OK.code());
  EXPECT_EQ(labm8::error::CANCELLED, labm8::Status::CANCELLED.error_code());
  EXPECT_EQ(labm8::error::CANCELLED, labm8::Status::CANCELLED.code());
  EXPECT_EQ(labm8::error::UNKNOWN, labm8::Status::UNKNOWN.error_code());
  EXPECT_EQ(labm8::error::UNKNOWN, labm8::Status::UNKNOWN.code());
}

TEST(Status, ConstructorZero) {
  labm8::Status status(labm8::error::OK, "msg");
  EXPECT_TRUE(status.ok());
  EXPECT_EQ("OK", status.ToString());
}

TEST(Status, CheckOK) {
  labm8::Status status;
  CHECK_OK(status);
  CHECK_OK(status) << "Failed";
  DCHECK_OK(status) << "Failed";
}

TEST(Status, ErrorMessage) {
  labm8::Status status(labm8::error::INVALID_ARGUMENT, "");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ("", status.error_message().ToString());
  EXPECT_EQ("", status.message().ToString());
  EXPECT_EQ("INVALID_ARGUMENT", status.ToString());
  status = labm8::Status(labm8::error::INVALID_ARGUMENT, "msg");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ("msg", status.error_message().ToString());
  EXPECT_EQ("msg", status.message().ToString());
  EXPECT_EQ("INVALID_ARGUMENT:msg", status.ToString());
  status = labm8::Status(labm8::error::OK, "msg");
  EXPECT_TRUE(status.ok());
  EXPECT_EQ("", status.error_message().ToString());
  EXPECT_EQ("", status.message().ToString());
  EXPECT_EQ("OK", status.ToString());
}

TEST(Status, Copy) {
  labm8::Status a(labm8::error::UNKNOWN, "message");
  labm8::Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  labm8::Status a(labm8::error::UNKNOWN, "message");
  labm8::Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, AssignEmpty) {
  labm8::Status a(labm8::error::UNKNOWN, "message");
  labm8::Status b;
  a = b;
  ASSERT_EQ(string("OK"), a.ToString());
  ASSERT_TRUE(b.ok());
  ASSERT_TRUE(a.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(labm8::Status::OK, labm8::Status()); }

TEST(Status, EqualsSame) {
  const labm8::Status a = labm8::Status(labm8::error::CANCELLED, "message");
  const labm8::Status b = labm8::Status(labm8::error::CANCELLED, "message");
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const labm8::Status a = labm8::Status(labm8::error::CANCELLED, "message");
  const labm8::Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const labm8::Status a = labm8::Status(labm8::error::CANCELLED, "message");
  const labm8::Status b = labm8::Status(labm8::error::UNKNOWN, "message");
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const labm8::Status a = labm8::Status(labm8::error::CANCELLED, "message");
  const labm8::Status b = labm8::Status(labm8::error::CANCELLED, "another");
  ASSERT_NE(a, b);
}

TEST(Status, FormatString) {
  const labm8::Status a = labm8::Status(labm8::error::INVALID_ARGUMENT, "a = {}", 5);
  ASSERT_EQ(a.error_message(), "a = 5");
  ASSERT_EQ(a.error_code(), labm8::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace labm8

TEST_MAIN();
