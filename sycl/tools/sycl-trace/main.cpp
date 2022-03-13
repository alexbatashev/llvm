//==------------ main.cpp - SYCL Tracing Tool ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "launch.hpp"
#include "llvm/Support/CommandLine.h"

#include <bitset>
#include <iostream>
#include <string>

using namespace llvm;

enum ModeKind { PI, ZE, CU };
enum PrintFormatKind { PRETTY_COMPACT, PRETTY_VERBOSE, CLASSIC };

enum LogDomainKinds {
  LOG_UNKNOWN = 0,
  LOG_PROGRAM_MANAGER = 1,
  LOG_SCHEDULER = 2,
  LOG_DEVICE = 3
};

enum LogLevelKinds { LOG_ERR, LOG_WARN, LOG_INFO, LOG_NONE };

int main(int argc, char **argv, char *env[]) {
  cl::list<ModeKind> Modes(
      cl::desc("Available tracing modes:"),
      cl::values(
          // TODO graph dot
          clEnumValN(PI, "plugin", "Trace Plugin Interface calls"),
          clEnumValN(ZE, "level_zero", "Trace Level Zero calls"),
          clEnumValN(ZE, "cuda", "Trace CUDA Driver API calls")));
  cl::opt<PrintFormatKind> PrintFormat(
      "print-format", cl::desc("Print format"),
      cl::values(
          clEnumValN(PRETTY_COMPACT, "compact", "Human readable compact"),
          clEnumValN(PRETTY_VERBOSE, "verbose", "Human readable verbose"),
          clEnumValN(
              CLASSIC, "classic",
              "Similar to SYCL_PI_TRACE, only compatible with PI layer")));
  cl::bits<LogDomainKinds> LogDomains(
      cl::desc("Available log domains:"),
      cl::values(
          clEnumValN(LOG_UNKNOWN, "log-unknown", "Unknown log source"),
          clEnumValN(LOG_PROGRAM_MANAGER, "log-prog-manager",
                     "Logs from Program Manager"),
          clEnumValN(LOG_SCHEDULER, "log-sched", "Logs from scheduler"),
          clEnumValN(LOG_DEVICE, "log-device", "Logs related to devices")));
  cl::opt<LogLevelKinds> LogLevel(
      "log-level", cl::desc("Log verbosity level"),
      cl::values(clEnumValN(LOG_ERR, "err", "Critical errors only"),
                 clEnumValN(LOG_WARN, "warn", "Warnings"),
                 clEnumValN(LOG_INFO, "info", "Debug info from SYCL runtime"),
                 clEnumValN(LOG_NONE, "none", "Do not print logs")));
  cl::opt<std::string> TargetExecutable(
      cl::Positional, cl::desc("<target executable>"), cl::Required);
  cl::list<std::string> Argv(cl::ConsumeAfter,
                             cl::desc("<program arguments>..."));

  cl::ParseCommandLineOptions(argc, argv);

  std::vector<std::string> NewEnv;

  {
    size_t I = 0;
    while (env[I] != nullptr)
      NewEnv.emplace_back(env[I++]);
  }

  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.so");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_pi_trace_collector.so");
  NewEnv.push_back("XPTI_TRACE_ENABLE=1");

  const auto EnablePITrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_PI_ENABLE=1");
  };
  const auto EnableZETrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_ZE_ENABLE=1");
    NewEnv.push_back("SYCL_PI_LEVEL_ZERO_ENABLE_TRACING=1");
    NewEnv.push_back("ZE_ENABLE_TRACING_LAYER=1");
  };
  const auto EnableCUTrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_CU_ENABLE=1");
    NewEnv.push_back("SYCL_PI_CUDA_ENABLE_TRACING=1");
  };

  for (auto Mode : Modes) {
    switch (Mode) {
    case PI:
      EnablePITrace();
      break;
    case ZE:
      EnableZETrace();
      break;
    case CU:
      EnableZETrace();
      break;
    }
  }

  if (PrintFormat == CLASSIC) {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=classic");
  } else if (PrintFormat == PRETTY_VERBOSE) {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=verbose");
  } else {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=compact");
  }

  if (LogLevel == LOG_ERR) {
    NewEnv.push_back("SYCL_TRACE_LOG_LEVEL=err");
  } else if (LogLevel == LOG_WARN) {
    NewEnv.push_back("SYCL_TRACE_LOG_LEVEL=warn");
  } else if (LogLevel == LOG_INFO) {
    NewEnv.push_back("SYCL_TRACE_LOG_LEVEL=info");
  } else if (LogLevel == LOG_NONE) {
    NewEnv.push_back("SYCL_TRACE_LOG_LEVEL=none");
  }

  std::bitset<4> LogMask{LogDomains.getBits() == 0 ? 0b1111
                                                   : LogDomains.getBits()};
  NewEnv.push_back("SYCL_TRACE_LOG_MASK=" + LogMask.to_string());

  if (Modes.size() == 0) {
    EnablePITrace();
    EnableZETrace();
    EnableCUTrace();
  }

  std::vector<std::string> Args;

  Args.push_back(TargetExecutable);
  std::copy(Argv.begin(), Argv.end(), std::back_inserter(Args));

  int Err = launch(TargetExecutable.c_str(), Args, NewEnv);

  if (Err) {
    std::cerr << "Failed to launch target application. Error code " << Err
              << "\n";
    return Err;
  }

  return 0;
}
