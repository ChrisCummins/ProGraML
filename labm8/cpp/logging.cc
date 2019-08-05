// This code is adapted from Google's protocol buffer sources.
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

#include "labm8/cpp/logging.h"

#include <atomic>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "labm8/cpp/string.h"

// ===================================================================
// emulates google3/base/logging.cc

// If the minimum logging level is not set, we default to logging messages for
// all levels.
#ifndef MIN_LOG_LEVEL
#define MIN_LOG_LEVEL LOGLEVEL_DEBUG
#endif

namespace labm8 {
namespace internal {

void DefaultLogHandler(LogLevel level, const char *filename, int line,
                       const string &message) {
  if (level < MIN_LOG_LEVEL) {
    return;
  }
  static const char *level_names[] = {"D", "I", "W", "E", "F"};

  string timestamp =
      absl::FormatTime("%Y-%m-%d %H:%M:%S", absl::Now(), absl::TimeZone());

  // We use fprintf() instead of cerr because we want this to work at static
  // initialization time.
  fprintf(stderr, "%s %s [%s:%d] %s\n", level_names[level], timestamp.c_str(),
          filename, line, message.c_str());
  fflush(stderr);  // Needed on MSVC.
}

void NullLogHandler(LogLevel /* level */, const char * /* filename */,
                    int /* line */, const string & /* message */) {
  // Nothing.
}

static LogHandler *log_handler_ = &DefaultLogHandler;
static std::atomic<int> log_silencer_count_ = ATOMIC_VAR_INIT(0);

LogMessage &LogMessage::operator<<(const string &value) {
  message_ += value;
  return *this;
}

LogMessage &LogMessage::operator<<(const char *value) {
  message_ += value;
  return *this;
}

LogMessage &LogMessage::operator<<(const StringPiece &value) {
  message_ += value.ToString();
  return *this;
}

// FIXME:
//
// LogMessage& LogMessage::operator<<(
//    const ::phd::Status& status) {
//  message_ += status.ToString();
//  return *this;
//}
//
// LogMessage& LogMessage::operator<<(const uint128& value) {
//  std::ostringstream str;
//  str << value;
//  message_ += str.str();
//  return *this;
//}

// Since this is just for logging, we don't care if the current locale changes
// the results -- in fact, we probably prefer that.  So we use snprintf()
// instead of Simple*toa().
#undef DECLARE_STREAM_OPERATOR
#define DECLARE_STREAM_OPERATOR(TYPE, FORMAT)                      \
  LogMessage &LogMessage::operator<<(TYPE value) {                 \
    /* 128 bytes should be big enough for any of the primitive */  \
    /* values which we print with this, but well use snprintf() */ \
    /* anyway to be extra safe. */                                 \
    char buffer[128];                                              \
    snprintf(buffer, sizeof(buffer), FORMAT, value);               \
    /* Guard against broken MSVC snprintf(). */                    \
    buffer[sizeof(buffer) - 1] = '\0';                             \
    message_ += buffer;                                            \
    return *this;                                                  \
  }

DECLARE_STREAM_OPERATOR(char, "%c")

DECLARE_STREAM_OPERATOR(int, "%d")

DECLARE_STREAM_OPERATOR(unsigned int, "%u")

DECLARE_STREAM_OPERATOR(long, "%ld")

DECLARE_STREAM_OPERATOR(unsigned long, "%lu")

DECLARE_STREAM_OPERATOR(double, "%g")

DECLARE_STREAM_OPERATOR(void *, "%p")

DECLARE_STREAM_OPERATOR(long long, "%lld")

DECLARE_STREAM_OPERATOR(unsigned long long, "%llu")

#undef DECLARE_STREAM_OPERATOR

LogMessage::LogMessage(LogLevel level, const char *filename, int line)
    : level_(level), filename_(filename), line_(line) {}

LogMessage::~LogMessage() {}

void LogMessage::Finish() {
  bool suppress = false;

  if (level_ != LOGLEVEL_FATAL) {
    suppress = log_silencer_count_ > 0;
  }

  if (!suppress) {
    log_handler_(level_, filename_, line_, message_);
  }

  if (level_ == LOGLEVEL_FATAL) {
#if PROTOBUF_USE_EXCEPTIONS
    throw FatalException(filename_, line_, message_);
#else
    abort();
#endif
  }
}

void LogFinisher::operator=(LogMessage &other) { other.Finish(); }

}  // namespace internal

LogHandler *SetLogHandler(LogHandler *new_func) {
  LogHandler *old = internal::log_handler_;
  if (old == &internal::NullLogHandler) {
    old = nullptr;
  }
  if (new_func == nullptr) {
    internal::log_handler_ = &internal::NullLogHandler;
  } else {
    internal::log_handler_ = new_func;
  }
  return old;
}

LogSilencer::LogSilencer() { ++internal::log_silencer_count_; };

LogSilencer::~LogSilencer() { --internal::log_silencer_count_; };

}  // namespace labm8
