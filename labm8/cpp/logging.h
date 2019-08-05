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

#pragma once

#include "labm8/cpp/macros.h"
#include "labm8/cpp/port.h"
#include "labm8/cpp/stringpiece.h"

// ===================================================================
// emulates google3/base/logging.h

namespace labm8 {

enum LogLevel {
  LOGLEVEL_DEBUG,    // Debugging information.
  LOGLEVEL_INFO,     // Informational.
  LOGLEVEL_WARNING,  // Warns about issues that, although not technically a
                     // problem now, could cause problems in the future.  For
                     // example, a // warning will be printed when parsing a
                     // message that is near the message size limit.
  LOGLEVEL_ERROR,    // An error occurred which should never happen during
                     // normal use.
  LOGLEVEL_FATAL,    // An error occurred from which the library cannot
                     // recover.  This usually indicates a programming error
                     // in the code which calls the library, especially when
                     // compiled in debug mode.

#ifdef NDEBUG
  LOGLEVEL_DFATAL = LOGLEVEL_ERROR
#else
  LOGLEVEL_DFATAL = LOGLEVEL_FATAL
#endif
};

class StringPiece;
class Status;
class uint128;

namespace internal {

class LogFinisher;

class LogMessage {
 public:
  LogMessage(LogLevel level, const char *filename, int line);
  ~LogMessage();

  LogMessage &operator<<(const std::string &value);
  LogMessage &operator<<(const char *value);
  LogMessage &operator<<(char value);
  LogMessage &operator<<(int value);
  LogMessage &operator<<(uint value);
  LogMessage &operator<<(long value);
  LogMessage &operator<<(unsigned long value);
  LogMessage &operator<<(long long value);
  LogMessage &operator<<(unsigned long long value);
  LogMessage &operator<<(double value);
  LogMessage &operator<<(void *value);
  LogMessage &operator<<(const StringPiece &value);
  LogMessage &operator<<(const labm8::Status &status);
  LogMessage &operator<<(const uint128 &value);

 private:
  friend class LogFinisher;
  void Finish();

  LogLevel level_;
  const char *filename_;
  int line_;
  std::string message_;
};

// Used to make the entire "LOG(BLAH) << etc." expression have a void return
// type and print a newline after each message.
class LogFinisher {
 public:
  void operator=(LogMessage &other);
};

template <typename T>
bool IsOk(T status) {
  return status.ok();
}
template <>
inline bool IsOk(bool status) {
  return status;
}

}  // namespace internal

// Undef everything in case we're being mixed with some other Google library
// which already defined them itself.  Presumably all Google libraries will
// support the same syntax for these so it should not be a big deal if they
// end up using our definitions instead.
#undef LOG
#undef LOG_IF

#undef CHECK
#undef CHECK_OK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_GT
#undef CHECK_GE
#undef CHECK_NOTNULL

#undef DLOG
#undef DCHECK
#undef DCHECK_OK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LT
#undef DCHECK_LE
#undef DCHECK_GT
#undef DCHECK_GE

#define LOG(LEVEL)                                                  \
  ::labm8::internal::LogFinisher() = ::labm8::internal::LogMessage( \
      ::labm8::LOGLEVEL_##LEVEL, __FILE__, __LINE__)
#define LOG_IF(LEVEL, CONDITION) !(CONDITION) ? (void)0 : LOG(LEVEL)

#define CHECK(EXPRESSION) \
  LOG_IF(FATAL, !(EXPRESSION)) << "CHECK failed: " #EXPRESSION ": "
#define CHECK_OK(A) CHECK(::labm8::internal::IsOk(A))
#define CHECK_EQ(A, B) CHECK((A) == (B))
#define CHECK_NE(A, B) CHECK((A) != (B))
#define CHECK_LT(A, B) CHECK((A) < (B))
#define CHECK_LE(A, B) CHECK((A) <= (B))
#define CHECK_GT(A, B) CHECK((A) > (B))
#define CHECK_GE(A, B) CHECK((A) >= (B))

namespace internal {
template <typename T>
T *CheckNotNull(const char * /* file */, int /* line */, const char *name,
                T *val) {
  if (val == nullptr) {
    LOG(FATAL) << name;
  }
  return val;
}
}  // namespace internal
#define CHECK_NOTNULL(A)                              \
  ::labm8::internal::CheckNotNull(__FILE__, __LINE__, \
                                  "'" #A "' must not be nullptr", (A))

#ifdef NDEBUG

#define DLOG(LEVEL) LOG_IF(LEVEL, false)

#define DCHECK(EXPRESSION) \
  while (false) CHECK(EXPRESSION)
#define DCHECK_OK(E) DCHECK(::labm8::internal::IsOk(E))
#define DCHECK_EQ(A, B) DCHECK((A) == (B))
#define DCHECK_NE(A, B) DCHECK((A) != (B))
#define DCHECK_LT(A, B) DCHECK((A) < (B))
#define DCHECK_LE(A, B) DCHECK((A) <= (B))
#define DCHECK_GT(A, B) DCHECK((A) > (B))
#define DCHECK_GE(A, B) DCHECK((A) >= (B))

#else  // NDEBUG

#define DLOG LOG

#define DCHECK CHECK
#define DCHECK_OK CHECK_OK
#define DCHECK_EQ CHECK_EQ
#define DCHECK_NE CHECK_NE
#define DCHECK_LT CHECK_LT
#define DCHECK_LE CHECK_LE
#define DCHECK_GT CHECK_GT
#define DCHECK_GE CHECK_GE

#endif  // !NDEBUG

typedef void LogHandler(LogLevel level, const char *filename, int line,
                        const std::string &message);

// The protobuf library sometimes writes warning and error messages to
// stderr.  These messages are primarily useful for developers, but may
// also help end users figure out a problem.  If you would prefer that
// these messages be sent somewhere other than stderr, call SetLogHandler()
// to set your own handler.  This returns the old handler.  Set the handler
// to nullptr to ignore log messages (but see also LogSilencer, below).
//
// Obviously, SetLogHandler is not thread-safe.  You should only call it
// at initialization time, and probably not from library code.  If you
// simply want to suppress log messages temporarily (e.g. because you
// have some code that tends to trigger them frequently and you know
// the warnings are not important to you), use the LogSilencer class
// below.
LogHandler *SetLogHandler(LogHandler *new_func);

// Create a LogSilencer if you want to temporarily suppress all log
// messages.  As long as any LogSilencer objects exist, non-fatal
// log messages will be discarded (the current LogHandler will *not*
// be called).  Constructing a LogSilencer is thread-safe.  You may
// accidentally suppress log messages occurring in another thread, but
// since messages are generally for debugging purposes only, this isn't
// a big deal.  If you want to intercept log messages, use SetLogHandler().
class LogSilencer {
 public:
  LogSilencer();
  ~LogSilencer();
};

}  // namespace labm8
