//==---------------------- collector.cpp -----------------------------------==//
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"
#include "xpti/xpti_trace_framework.hpp"

#include <CL/sycl/detail/spinlock.hpp>

#include <bitset>
#include <cassert>
#include <iostream>
#include <mutex>

sycl::detail::SpinLock GlobalLock;

bool HasZEPrinter = false;
bool HasCUPrinter = false;
bool HasPIPrinter = false;

void zePrintersInit();
void zePrintersFinish();
void cuPrintersInit();
void cuPrintersFinish();
void piPrintersInit();
void piPrintersFinish();

enum class XPTIEventsExtension {
  LogError = XPTI_EVENT(0),
  LogWarn = XPTI_EVENT(1),
  LogInfo = XPTI_EVENT(2)
};

enum class XPTITracePointsExtension {
  // Trace point for log events
  Log = XPTI_TRACE_POINT_BEGIN(0),
};

uint16_t LogTracePointT;
uint16_t LogInfoT;
uint16_t LogWarnT;
uint16_t LogErrorT;

enum LogDomainKinds {
  LOG_UNKNOWN = 0,
  LOG_PROGRAM_MANAGER = 1,
  LOG_SCHEDULER = 2,
  LOG_DEVICE = 3
};

enum class LogLevelKinds { Error = 0, Warning = 1, Info = 2 };

static bool LogVerbose = false;

static std::bitset<4> LogDomains;
static LogLevelKinds LogLevel;

XPTI_CALLBACK_API void piCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void zeCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void cuCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void logCallback(uint16_t TraceType,
                                   xpti::trace_event_data_t *Parent,
                                   xpti::trace_event_data_t *Event,
                                   uint64_t Instance, const void *UserData);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE")) {
    piPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         piCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         piCallback);
  } else if (std::string_view(StreamName) ==
                 "sycl.experimental.level_zero.debug" &&
             std::getenv("SYCL_TRACE_ZE_ENABLE")) {
    zePrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         zeCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         zeCallback);
  } else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
             std::getenv("SYCL_TRACE_CU_ENABLE")) {
    cuPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         cuCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         cuCallback);
  } else if (std::string_view(StreamName) == "sycl.debug" &&
             std::getenv("SYCL_TRACE_LOG_LEVEL") &&
             std::string_view{std::getenv("SYCL_TRACE_LOG_LEVEL")} != "none") {
    const char *MaskPtr = std::getenv("SYCL_TRACE_LOG_MASK");
    std::cout << "DOMAIN MASK " << MaskPtr << "\n";
    if (MaskPtr)
      LogDomains = std::bitset<4>(MaskPtr);
    std::string_view LevelView{std::getenv("SYCL_TRACE_LOG_LEVEL")};

    if (LevelView == "err")
      LogLevel = LogLevelKinds::Error;
    else if (LevelView == "warn")
      LogLevel = LogLevelKinds::Warning;
    else if (LevelView == "info")
      LogLevel = LogLevelKinds::Info;

    const char *PrintFormat = std::getenv("SYCL_TRACE_PRINT_FORMAT");
    if (PrintFormat) {
      std::string_view FormatView{PrintFormat};
      if (FormatView == "verbose") {
        LogVerbose = true;
      }
    }
    uint16_t StreamID = xptiRegisterStream(StreamName);
    LogTracePointT = xptiRegisterUserDefinedTracePoint(
        "sycl_dpcpp", static_cast<uint8_t>(XPTITracePointsExtension::Log));
    LogInfoT = xptiRegisterUserDefinedEventType(
        "sycl_dpcpp", static_cast<uint8_t>(XPTIEventsExtension::LogInfo));
    LogWarnT = xptiRegisterUserDefinedEventType(
        "sycl_dpcpp", static_cast<uint8_t>(XPTIEventsExtension::LogWarn));
    LogErrorT = xptiRegisterUserDefinedEventType(
        "sycl_dpcpp", static_cast<uint8_t>(XPTIEventsExtension::LogError));

    xptiRegisterCallback(StreamID, LogTracePointT, logCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE"))
    piPrintersFinish();
  else if (std::string_view(StreamName) ==
               "sycl.experimental.level_zero.debug" &&
           std::getenv("SYCL_TRACE_ZE_ENABLE"))
    zePrintersFinish();
  else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
           std::getenv("SYCL_TRACE_CU_ENABLE"))
    cuPrintersFinish();
}

XPTI_CALLBACK_API void logCallback(uint16_t /*TraceType*/,
                                   xpti::trace_event_data_t * /*Parent*/,
                                   xpti::trace_event_data_t *Event,
                                   uint64_t /*Instance*/,
                                   const void *UserData) {
  assert(Event);

  std::lock_guard _{GlobalLock};

  if (Event->event_type == LogInfoT) {
    if (LogLevel < LogLevelKinds::Info)
      return;
  } else if (Event->event_type == LogWarnT) {
    if (LogLevel < LogLevelKinds::Warning)
      return;
  }

  std::string DomainName = "unknown";

  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  bool HasDomain = false;
  bool HasDomainName = false;
  for (const auto &Item : *Metadata) {
    std::string_view Name{xptiLookupString(Item.first)};
    if (Name == "domain") {
      uint32_t DomainID = 100;
      xpti::object_data_t RawData = xptiLookupObject(Item.second);
      DomainID = *reinterpret_cast<const uint32_t *>(RawData.data);
      if (!LogDomains[DomainID])
        return;
      HasDomain = true;
    } else if (Name == "domain_name") {
      DomainName = std::string{xpti::readMetadata(Item).data()};
      HasDomainName = true;
    }
    if (HasDomain && HasDomainName)
      break;
  }

  if (Event->event_type == LogInfoT) {
    std::cout << "[INFO:" << DomainName;
  } else if (Event->event_type == LogWarnT) {
    std::cout << "[WARN:" << DomainName;
  } else if (Event->event_type == LogErrorT) {
    std::cout << "[ERROR:" << DomainName;
  }

  if (LogVerbose) {
    std::cout << ":";
    const xpti::payload_t *Payload = xptiQueryPayload(Event);

    if (Payload->source_file != nullptr &&
        !std::string_view{Payload->source_file}.empty()) {
      std::cout << Payload->source_file << ":" << Payload->line_no;
      if (Payload->name) {
        std::cout << "->" << Payload->name;
      }
      std::cout << "]";
    } else if (Payload->name &&
               std::string_view{Payload->name}.find("unknown") > 0) {
      std::cout << Payload->name << "]";
    } else {
      std::cout << "unknown]";
    }

    std::cout << "\n";
  } else {
    std::cout << "] ";
  }

  std::cout << static_cast<const char *>(UserData) << std::endl;
}
