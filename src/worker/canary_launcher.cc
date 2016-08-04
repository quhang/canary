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
 * @file src/worker/canary_launcher.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryLauncher.
 */

#include <algorithm>
#include <cereal/archives/xml.hpp>  // NOLINT
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "shared/canary_internal.h"

#include "message/launch_message.h"
#include "shared/initialize.h"
#include "shared/network.h"

// --launch_application=./app/logistic_loop/liblogistic_loop.so iterations=1
DEFINE_string(launch_application, "",
              "Launch an application given its binary location.");
// Auxiliary info for launching an application.
DEFINE_int32(launch_num_worker, -1,
             "Specify the number of worker. -1 means any.");
DEFINE_int32(launch_first_barrier, -1, "Specify the first barrier stage.");
DEFINE_int32(launch_priority, 100, "Specify the priority level.");

// --resume_application=1
DEFINE_int32(resume_application, -1,
             "Resume an application given its application id.");

// --control_application=1 --control_priority=99
DEFINE_int32(control_application, -1,
             "Control an application's priority given its application id.");
// Auxiliary info for controlling an application.
DEFINE_int32(control_priority, 100, "Specify the priority level.");

namespace {
//! Connects to the controller and returns the channel socket fd.
int ConnectToController() {
  using namespace canary;  // NOLINT
  struct addrinfo hints;
  network::initialize_addrinfo(&hints, false);
  struct addrinfo* available_addresses = nullptr;
  const int errorcode =
      getaddrinfo(FLAGS_controller_host.c_str(), FLAGS_launch_service.c_str(),
                  &hints, &available_addresses);
  CHECK_EQ(errorcode, 0) << gai_strerror(errorcode);
  CHECK_NOTNULL(available_addresses);
  const int result_fd =
      socket(available_addresses->ai_family, available_addresses->ai_socktype,
             available_addresses->ai_protocol);
  PCHECK(result_fd >= 0);
  const int status = connect(result_fd, available_addresses->ai_addr,
                             available_addresses->ai_addrlen);
  freeaddrinfo(available_addresses);
  PCHECK(status == 0);
  return result_fd;
}

//! Disconnects with the controller.
void DisconnectWithController(int socket_fd) {
  using namespace canary;  // NOLINT
  network::close_socket(socket_fd);
}

//! Sends a launch message and waits for response.
template <typename LaunchResponseMessageType, typename LaunchMessageType>
LaunchResponseMessageType LaunchAndWaitResponse(
    const LaunchMessageType& launch_message) {
  using namespace canary;  // NOLINT
  const int socket_fd = ConnectToController();
  CHECK_NE(socket_fd, -1);
  struct evbuffer* buffer =
      message::SerializeMessageWithControlHeader(launch_message);
  do {
    PCHECK(evbuffer_write(buffer, socket_fd) != -1);
  } while (evbuffer_get_length(buffer) != 0);
  evbuffer_free(buffer);
  buffer = evbuffer_new();
  while (evbuffer_read(buffer, socket_fd, -1) > 0) {
    if (struct evbuffer* whole_message =
            message::SegmentControlMessage(buffer)) {
      DisconnectWithController(socket_fd);
      auto header = CHECK_NOTNULL(message::ExamineControlHeader(whole_message));
      CHECK(header->category_group ==
            message::MessageCategoryGroup::LAUNCH_RESPONSE_COMMAND);
      LaunchResponseMessageType response_message;
      CHECK(header->category ==
            message::get_message_category(response_message));
      message::RemoveControlHeader(whole_message);
      message::DeserializeMessage(whole_message, &response_message);
      return std::move(response_message);
    }
  }
  LOG(FATAL) << "Launch response is not received!";
  return LaunchResponseMessageType();
}
}  // namespace

int main(int argc, char** argv) {
  using namespace canary;  // NOLINT
  InitializeCanaryWorker(&argc, &argv);
  if (!FLAGS_launch_application.empty()) {
    std::stringstream ss;
    {
      cereal::XMLOutputArchive archive(ss);
      for (int i = 1; i < argc; ++i) {
        std::string token(argv[i]);
        auto iter = std::find(token.begin(), token.end(), '=');
        CHECK(iter != token.end())
            << "Application argument needs to be specified as key=value";
        archive(cereal::make_nvp(std::string(token.begin(), iter),
                                 std::string(std::next(iter), token.end())));
      }
    }
    message::LaunchApplication launch_application;
    launch_application.binary_location = FLAGS_launch_application;
    launch_application.application_parameter = ss.str();
    launch_application.fix_num_worker = FLAGS_launch_num_worker;
    launch_application.first_barrier_stage = FLAGS_launch_first_barrier;
    launch_application.priority_level = FLAGS_launch_priority;
    auto response = LaunchAndWaitResponse<message::LaunchApplicationResponse>(
        launch_application);
    if (response.succeed) {
      printf("Launching application (id=%d) succeeded!\n",
             response.application_id);
    } else {
      printf("Launching application failed!\n%s\n",
             response.error_message.c_str());
    }
  } else if (FLAGS_resume_application != -1) {
    message::ResumeApplication resume_application;
    resume_application.application_id = FLAGS_resume_application;
    auto response = LaunchAndWaitResponse<message::ResumeApplicationResponse>(
        resume_application);
    if (response.succeed) {
      printf("Resuming application (id=%d) succeeded!\n",
             response.application_id);
    } else {
      printf("Resuming application failed!\n%s\n",
             response.error_message.c_str());
    }
  } else if (FLAGS_control_application != -1) {
    message::ControlApplicationPriority control_application;
    control_application.application_id = FLAGS_control_application;
    control_application.priority_level = FLAGS_control_priority;
    auto response =
        LaunchAndWaitResponse<message::ControlApplicationPriorityResponse>(
            control_application);
    if (response.succeed) {
      printf("Controlling application's priority (id=%d) succeeded!\n",
             response.application_id);
    } else {
      printf("Controlling application's priority failed!\n%s\n",
             response.error_message.c_str());
    }
  }
  return 0;
}
