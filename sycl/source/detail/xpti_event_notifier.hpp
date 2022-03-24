//==---------- xpti_event_notifier.hpp ----- PI Event Listener for XPTI ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "xpti/xpti_trace_framework.hpp"
#include <CL/sycl/detail/pi.h>
#include <detail/config.hpp>
#include <detail/global_handler.hpp>
#include <detail/plugin.hpp>
#include <CL/sycl/detail/common.hpp>
#include <detail/xpti_registry.hpp>
#include <mutex>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting
// traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class XPTIEventNotifier {
public:
  void initPIOverrides() {
#ifdef XPTI_TRACE_ENABLE
    if (SYCLConfig<SYCL_XPTI_ENABLE_EXTRA_INSTRUMENTATION>::get()) {
      std::vector<plugin> &Plugins = GlobalHandler::instance().getPlugins();
      for (plugin &Plugin : Plugins) {
        if (Plugin.getBackend() == backend::opencl) {
          wrapAPIs<backend::opencl>(Plugin);
        } else if (Plugin.getBackend() == backend::ext_oneapi_level_zero) {
          wrapAPIs<backend::ext_oneapi_level_zero>(Plugin);
        } else if (Plugin.getBackend() == backend::ext_oneapi_cuda) {
          wrapAPIs<backend::ext_oneapi_cuda>(Plugin);
        } else if (Plugin.getBackend() == backend::ext_oneapi_hip) {
          wrapAPIs<backend::ext_oneapi_hip>(Plugin);
        } else if (Plugin.getBackend() == backend::ext_intel_esimd_emulator) {
          wrapAPIs<backend::ext_intel_esimd_emulator>(Plugin);
        }
      }

      MWorker = std::thread{[this]() {
        while (!MStopped) {
          if (!xptiTraceEnabled()) {
            std::this_thread::yield();
            continue;
          }
          std::lock_guard Lock{MLock};

          auto It = std::remove_if(MQueue.begin(), MQueue.end(), [this](EventInfo Info) {
            pi_int32 Status;
            MEventGetInfo[EventInfo.Backend](Info.Event, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS, sizeof(pi_int32), &Status, nullptr);
            if (Status == PI_EVENT_COMPLETE) {
              uint64_t ID;
              xpti::trace_event_data_t *Event = xptiMakeEvent("event_status", nullptr, xpti::trace_signal, xpti_at::active, &ID);
              if (EventInfo.KernelName.has_value()) {
                xpti::addMetadata(Event, "kernel_name", *EventInfo.KernelName);
              }

              pi_uint64 Timestamp = 0;

              MEventProfilingInfo[Info.Backend](Info.Event, PI_PROFILING_INFO_COMMAND_QUEUED, sizeof(pi_uint64), &Timestamp, nullptr);
              xpti::addMeaddMetadata(Event, "event_queued", Timestamp);

              MEventProfilingInfo[Info.Backend](Info.Event, PI_PROFILING_INFO_COMMAND_SUBMIT, sizeof(pi_uint64), &Timestamp, nullptr);
              xpti::addMeaddMetadata(Event, "event_submit", Timestamp);

              MEventProfilingInfo[Info.Backend](Info.Event, PI_PROFILING_INFO_COMMAND_START, sizeof(pi_uint64), &Timestamp, nullptr);
              xpti::addMeaddMetadata(Event, "event_start", Timestamp);

              MEventProfilingInfo[Info.Backend](Info.Event, PI_PROFILING_INFO_COMMAND_END, sizeof(pi_uint64), &Timestamp, nullptr);
              xpti::addMeaddMetadata(Event, "event_end", Timestamp);

              xptiNotifySubscribers(GEventsStreamID, xpti::trace_signal, nullptr, Event, ID, nullptr);

              MEventRelease[Info.Backend](Info.Event);
            }

            return Status == PI_EVENT_COMPLETE;

            });

          MQueue.erase(It, MQueue.end());
          Lock.unlock();

          std::this_thread::yield();
        }
      }};
    }
#endif // XPTI_TRACE_ENABLE
  }
private:
  struct EventInfo {
    backend Backend;
    pi_event Event;
    std::optional<std::string> KernelName;
  };

  template <backend Backend>
  void wrapAPIs(plugin &Plugin) {
    MEventRelease[Backend] = Plugin.getPiPlugin().PiFunctionTable.piEventRelease;
    MEventRetain[Backend] = Plugin.getPiPlugin().PiFunctionTable.piEventRetain;
    MEventGetInfo[Backend] = Plugin.getPiPlugin().PiFunctionTable.piEventGetInfo;
    MEventGetProfilingInfo[Backend] = Plugin.getPiPlugin().PiFunctionTable.piEventGetProfilingInfo;

    MQueueCreate[Backend] = Plugin.getPiPlugin().PiFunctionTable.piQueueCreate;
    Plugin.getPiPlugin().PiFunctionTable.piQueueCreate = &XPTIEventNotifier::piQueueCreate<Backend>;

    MEnqueueKernel[Backend] = Plugin.getPiPlugin().PiFunctionTable.piEnqueueKernelLaunch;
    Plugin.getPiPlugin().PiFunctionTable.piEnqueueKernelLaunch = &XPTIEventNotifier::piEnqueueKernelLaunch<Backend>;
  }

  template <backend Backend> static pi_result piQueueCreate(pi_context context, pi_device device, pi_queue_properties properties, pi_queue *queue) {
    properties |= PI_QUEUE_PROFILING_ENABLE;
    return GlobalHandler::instance().getXPTIEventNotifier().MQueueCreate[Backend](context, device, properties, queue);
  }

  template <backend Backend>
  static pi_result piEnqueueKernelLaunch(
    pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
    pi_result Res = GlobalHandler::instance().getXPTIEventNotifier().MEnqueueKernel[Backend](queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);

    if (!xptiTraceEnabled())
      return Res;

    std::lock_guard Lock{GlobalHandler::instance().getXPTIEventNotifier().MLock};
    GlobalHandler::instance().getXPTIEventNotifier().MEventRetain[Backend](*event);
    GlobalHandler::instance().getXPTIEventNotifier().MQueue.emplace_back(Backend, *event);

    return Res;
  }

  using EnqueueKernelPtr = pi_result (*)(
    pi_queue , pi_kernel , pi_uint32 ,
    const size_t *, const size_t *,
    const size_t *, pi_uint32 ,
    const pi_event *, pi_event *);
  using QueueCreatePtr = pi_result (*)(pi_context context, pi_device device,
                                      pi_queue_properties properties,
                                      pi_queue *queue);
  using EventRetainPtr = pi_result (*)(pi_event);
  using EventReleasePtr = pi_result (*)(pi_event);
  using EventGetInfoPtr = pi_result (*)(pi_event event, pi_event_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret);
  using EventGetProfilingInfoPtr = pi_result (*)(pi_event event,
                                                pi_profiling_info param_name,
                                                size_t param_value_size,
                                                void *param_value,
                                                size_t *param_value_size_ret);

  std::unordered_map<backend, EnqueueKernelPtr> MEnqueueKernel;
  std::unordered_map<backend, QueueCreatePtr> MQueueCreate;
  std::unordered_map<backend, EventRetainPtr> MEventRetain;
  std::unordered_map<backend, EventReleasePtr> MEventRelease;
  std::unordered_map<backend, EventGetInfoPtr> MEventGetInfo;
  std::unordered_map<backend, EventGetProfilingInfoPtr> MEventGetProfilingInfo;

  std::mutex MLock;
  std::vector<EventInfo> MQueue;
  std::thread MWorker;
  std::atomic_bool MStopped{false};
};

}
}
}
