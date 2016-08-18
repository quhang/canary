/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file src/message/launch_message.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Messages used between the launcher and the controller.
 */

#ifndef CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_
#define CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_

#include <map>
#include <set>
#include <string>
#include <utility>

#include "message/message.h"

/*
 * All the messages in this file are exchanged between the launcher and the
 * controller. The launcher, which might run in a cluster resource manager, can
 * do coarse-grained execution management, e.g. deciding whether an application
 * should pause or give resources to higher priority applications, or shutting
 * down workers to give resources to other cloud frameworks.
 */

namespace canary {
namespace message {

//! Launches an application.
struct LaunchApplication {
  std::string binary_location;
  std::string application_parameter;
  //! Fix the number of workers.
  int fix_num_worker = -1;
  //! The first barrier stage.
  int first_barrier_stage = -1;
  int priority_level = -1;
  //! Placement algorithm to use.
  std::string placement_algorithm;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(binary_location, application_parameter);
    archive(fix_num_worker, first_barrier_stage, priority_level);
    archive(placement_algorithm);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, LAUNCH_APPLICATION, LaunchApplication);
//! Responds to an application launching command.
struct LaunchApplicationResponse {
  int application_id = -1;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, LAUNCH_APPLICATION_RESPONSE,
                 LaunchApplicationResponse);

//! Pauses an application.
struct PauseApplication {
  int application_id = -1;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, PAUSE_APPLICATION, PauseApplication);
//! Responds to an application pausing command.
struct PauseApplicationResponse {
  int application_id = -1;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, PAUSE_APPLICATION_RESPONSE,
                 PauseApplicationResponse);

//! Resumes an application.
struct ResumeApplication {
  int application_id = -1;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, RESUME_APPLICATION, ResumeApplication);
//! Responds to an application resuming command.
struct ResumeApplicationResponse {
  int application_id = -1;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, RESUME_APPLICATION_RESPONSE,
                 ResumeApplicationResponse);

//! Controls the running priority of an application.
struct ControlApplicationPriority {
  int application_id = -1;
  int priority_level = -1;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, priority_level);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, CONTROL_APPLICATION_PRIORITY,
                 ControlApplicationPriority);
//! Responds to an application priority control command.
struct ControlApplicationPriorityResponse {
  int application_id = -1;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, CONTROL_APPLICATION_PRIORITY_RESPONSE,
                 ControlApplicationPriorityResponse);

//! Requests running stats of an application.
struct RequestApplicationStat {
  int application_id = -1;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, REQUEST_APPLICATION_STAT,
                 RequestApplicationStat);
//! Responds with the running stats of an application.
struct RequestApplicationStatResponse {
  int application_id = -1;
  bool succeed = false;
  std::string error_message;
  //! The total of cycles used in seconds.
  double cycles = 0;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message, cycles);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, REQUEST_APPLICATION_STAT_RESPONSE,
                 RequestApplicationStatResponse);

//! Shuts down workers after migrating out running partitions on those workers.
struct RequestShutdownWorker {
  std::set<int> shutdown_worker_id_set;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(shutdown_worker_id_set);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, REQUEST_SHUTDOWN_WORKER,
                 RequestShutdownWorker);
//! Responds to worker shutdown command.
struct RequestShutdownWorkerResponse {
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, REQUEST_SHUTDOWN_WORKER_RESPONSE,
                 RequestShutdownWorkerResponse);

struct RequestWorkerStat {
  template <typename Archive>
  void serialize(Archive&) {  // NOLINT
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, REQUEST_WORKER_STAT, RequestWorkerStat);
struct RequestWorkerStatResponse {
  bool succeed = false;
  std::string error_message;
  // WorkerId -> (canary_usage, other_usage).
  std::map<int, std::pair<double, double>> cpu_util_percentage_map;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(succeed, error_message, cpu_util_percentage_map);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, REQUEST_WORKER_STAT_RESPONSE,
                 RequestWorkerStatResponse);

//! Triggers the scheduling algorithm.
struct TriggerScheduling {
  std::string scheduling_algorithm;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(scheduling_algorithm);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, TRIGGER_SCHEDULING, TriggerScheduling);
//! Response of the scheduling algorithm execution.
struct TriggerSchedulingResponse {
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, TRIGGER_SCHEDULING_RESPONSE,
                 TriggerSchedulingResponse);

}  // namespace message
}  // namespace canary

#endif  // CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_
