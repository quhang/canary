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
 * @brief Class LaunchMessage.
 */

#ifndef CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_
#define CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_

#include <string>

#include "message/message.h"

namespace canary {
namespace message {

//! Launches an application.
struct LaunchApplication {
  std::string binary_location;
  std::string application_parameter;
  int fix_num_worker = -1;
  int first_barrier_stage = -1;
  int priority_level = -1;

  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(binary_location, application_parameter);
    archive(fix_num_worker, first_barrier_stage, priority_level);
  }
};
REGISTER_MESSAGE(LAUNCH_COMMAND, LAUNCH_APPLICATION, LaunchApplication);

//! Responds to an application launching command.
struct LaunchApplicationResponse {
  int application_id;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, LAUNCH_APPLICATION_RESPONSE,
                 LaunchApplicationResponse);

//! Launches an application.
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
  int application_id;
  bool succeed = false;
  std::string error_message;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, succeed, error_message);
  }
};
REGISTER_MESSAGE(LAUNCH_RESPONSE_COMMAND, RESUME_APPLICATION_RESPONSE,
                 ResumeApplicationResponse);

//! Controls the priority of an application.
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

//! Responds to an application priority tuning.
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

}  // namespace message
}  // namespace canary

#endif  // CANARY_SRC_MESSAGE_LAUNCH_MESSAGE_H_
