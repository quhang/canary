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
 * @file src/launcher/launcher_helper.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class LauncherHelper.
 */

#ifndef CANARY_SRC_LAUNCHER_LAUNCHER_HELPER_H_
#define CANARY_SRC_LAUNCHER_LAUNCHER_HELPER_H_

#include "shared/canary_internal.h"

#include "message/launch_message.h"
#include "shared/network.h"

namespace canary {

/**
 * Helper class to connect to Canary controller.
 */
class LauncherHelper {
 public:
  LauncherHelper() {}
  virtual ~LauncherHelper() {}
  //! Connects to the controller and returns the channel socket fd.
  bool ConnectToController() {
    CHECK_EQ(socket_fd_, -1);
    struct addrinfo hints;
    network::initialize_addrinfo(&hints, false);
    struct addrinfo* available_addresses = nullptr;
    // Caution: accesses global variables.
    const int errorcode =
        getaddrinfo(FLAGS_controller_host.c_str(), FLAGS_launch_service.c_str(),
                    &hints, &available_addresses);
    CHECK_EQ(errorcode, 0) << gai_strerror(errorcode);
    CHECK_NOTNULL(available_addresses);
    socket_fd_ =
        socket(available_addresses->ai_family, available_addresses->ai_socktype,
               available_addresses->ai_protocol);
    PCHECK(socket_fd_ >= 0);
    const int status = connect(socket_fd_, available_addresses->ai_addr,
                               available_addresses->ai_addrlen);
    freeaddrinfo(available_addresses);
    PCHECK(status == 0);
    return true;
  }
  //! Disconnects with the controller.
  void DisconnectWithController() {
    CHECK_NE(socket_fd_, -1);
    network::close_socket(socket_fd_);
    socket_fd_ = -1;
  }

  //! Sends a launch message and waits for response.
  template <typename LaunchResponseMessageType, typename LaunchMessageType>
  LaunchResponseMessageType LaunchAndWaitResponse(
      const LaunchMessageType& launch_message) {
    if (socket_fd_ == -1) {
      CHECK(ConnectToController());
    }
    CHECK_NE(socket_fd_, -1);
    struct evbuffer* buffer =
        message::SerializeMessageWithControlHeader(launch_message);
    do {
      PCHECK(evbuffer_write(buffer, socket_fd_) != -1);
    } while (evbuffer_get_length(buffer) != 0);
    evbuffer_free(buffer);
    buffer = evbuffer_new();
    while (evbuffer_read(buffer, socket_fd_, -1) > 0) {
      if (struct evbuffer* whole_message =
              message::SegmentControlMessage(buffer)) {
        DisconnectWithController();
        auto header =
            CHECK_NOTNULL(message::ExamineControlHeader(whole_message));
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

 private:
  int socket_fd_ = -1;
};

}  // namespace canary
#endif  // CANARY_SRC_LAUNCHER_LAUNCHER_HELPER_H_
