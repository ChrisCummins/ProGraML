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

// Author: kenton@google.com (Kenton Varda)

#include "labm8/cpp/common.h"

#include <errno.h>
#include <stdio.h>
#include <atomic>
#include <sstream>
#include <vector>

#include <pthread.h>

#include "labm8/cpp/callback.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/mutex.h"
#include "labm8/cpp/once.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/stringpiece.h"
// #include "labm8/cpp/strutil.h"
// #include "labm8/cpp/int128.h"

namespace labm8 {

// ===================================================================
// emulates google3/base/callback.cc

Closure::~Closure() {}

namespace internal {
FunctionClosure0::~FunctionClosure0() {}
}  // namespace internal

void DoNothing() {}

// ===================================================================
// emulates google3/util/endian/endian.h
//
// TODO(xiaofeng): PROTOBUF_LITTLE_ENDIAN is unfortunately defined in
// google/protobuf/io/coded_stream.h and therefore can not be used here.
// Maybe move that macro definition here in the furture.
uint32 ghtonl(uint32 x) {
  union {
    uint32 result;
    uint8 result_array[4];
  };
  result_array[0] = static_cast<uint8>(x >> 24);
  result_array[1] = static_cast<uint8>((x >> 16) & 0xFF);
  result_array[2] = static_cast<uint8>((x >> 8) & 0xFF);
  result_array[3] = static_cast<uint8>(x & 0xFF);
  return result;
}

// ===================================================================
// Shutdown support.

namespace internal {

struct ShutdownData {
  ~ShutdownData() {
    std::reverse(functions.begin(), functions.end());
    for (auto pair : functions) pair.first(pair.second);
  }

  static ShutdownData *get() {
    static auto *data = new ShutdownData;
    return data;
  }

  std::vector<std::pair<void (*)(const void *), const void *>> functions;
  Mutex mutex;
};

static void RunZeroArgFunc(const void *arg) {
  void (*func)() = reinterpret_cast<void (*)()>(const_cast<void *>(arg));
  func();
}

void OnShutdown(void (*func)()) {
  OnShutdownRun(RunZeroArgFunc, reinterpret_cast<void *>(func));
}

void OnShutdownRun(void (*f)(const void *), const void *arg) {
  auto shutdown_data = ShutdownData::get();
  MutexLock lock(&shutdown_data->mutex);
  shutdown_data->functions.push_back(std::make_pair(f, arg));
}

}  // namespace internal

void ShutdownProtobufLibrary() {
  // This function should be called only once, but accepts multiple calls.
  static bool is_shutdown = false;
  if (!is_shutdown) {
    delete internal::ShutdownData::get();
    is_shutdown = true;
  }
}

#if PROTOBUF_USE_EXCEPTIONS
FatalException::~FatalException() throw() {}

const char *FatalException::what() const throw() { return message_.c_str(); }
#endif

}  // namespace labm8
