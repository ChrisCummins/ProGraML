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

// Author: kenton@google.com (Kenton Varda) and others
//
// Contains basic types and utilities used by the rest of the library.

#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "labm8/cpp/macros.h"
#include "labm8/cpp/platform_macros.h"
#include "labm8/cpp/port.h"

#ifndef PROTOBUF_USE_EXCEPTIONS
#if defined(_MSC_VER) && defined(_CPPUNWIND)
#define PROTOBUF_USE_EXCEPTIONS 1
#elif defined(__EXCEPTIONS)
#define PROTOBUF_USE_EXCEPTIONS 1
#else
#define PROTOBUF_USE_EXCEPTIONS 0
#endif
#endif

#if PROTOBUF_USE_EXCEPTIONS
#include <exception>
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>  // for TARGET_OS_IPHONE
#endif

#ifdef __GNUC__
// Provided at least since GCC 3.0.
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define PREDICT_TRUE(x) (x)
#endif

#ifdef __GNUC__
// Provided at least since GCC 3.0.
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#else
#define PREDICT_FALSE(x) (x)
#endif

namespace std {}

namespace labm8 {

// ===================================================================
// from google3/util/utf8/public/unilib.h

class StringPiece;
namespace internal {

// Checks if the buffer contains structurally-valid UTF-8.  Implemented in
// structurally_valid.cc.
bool IsStructurallyValidUTF8(const char *buf, int len);

inline bool IsStructurallyValidUTF8(const std::string &str) {
  return IsStructurallyValidUTF8(str.data(), static_cast<int>(str.length()));
}

// Returns initial number of bytes of structually valid UTF-8.
int UTF8SpnStructurallyValid(const StringPiece &str);

// Coerce UTF-8 byte string in src_str to be
// a structurally-valid equal-length string by selectively
// overwriting illegal bytes with replace_char (typically ' ' or '?').
// replace_char must be legal printable 7-bit Ascii 0x20..0x7e.
// src_str is read-only.
//
// Returns pointer to output buffer, src_str.data() if no changes were made,
//  or idst if some bytes were changed. idst is allocated by the caller
//  and must be at least as big as src_str
//
// Optimized for: all structurally valid and no byte copying is done.
//
char *UTF8CoerceToStructurallyValid(const StringPiece &str, char *dst,
                                    char replace_char);

}  // namespace internal

// ===================================================================
// Shutdown support.

// Shut down the entire protocol buffers library, deleting all static-duration
// objects allocated by the library or by generated .pb.cc files.
//
// There are two reasons you might want to call this:
// * You use a draconian definition of "memory leak" in which you expect
//   every single malloc() to have a corresponding free(), even for objects
//   which live until program exit.
// * You are writing a dynamically-loaded library which needs to clean up
//   after itself when the library is unloaded.
//
// It is safe to call this multiple times.  However, it is not safe to use
// any other part of the protocol buffers library after
// ShutdownProtobufLibrary() has been called. Furthermore this call is not
// thread safe, user needs to synchronize multiple calls.
void ShutdownProtobufLibrary();

namespace internal {

// Register a function to be called when ShutdownProtocolBuffers() is called.
void OnShutdown(void (*func)());
// Run an arbitrary function on an arg
void OnShutdownRun(void (*f)(const void *), const void *arg);

template <typename T>
T *OnShutdownDelete(T *p) {
  OnShutdownRun([](const void *pp) { delete static_cast<const T *>(pp); }, p);
  return p;
}

}  // namespace internal

#if PROTOBUF_USE_EXCEPTIONS
class FatalException : public std::exception {
 public:
  FatalException(const char *filename, int line, const std::string &message)
      : filename_(filename), line_(line), message_(message) {}
  virtual ~FatalException() throw();

  virtual const char *what() const throw();

  const char *filename() const { return filename_; }
  int line() const { return line_; }
  const std::string &message() const { return message_; }

 private:
  const char *filename_;
  const int line_;
  const std::string message_;
};
#endif

// This is at the end of the file instead of the beginning to work around a bug
// in some versions of MSVC.
using std::string;

}  // namespace labm8
