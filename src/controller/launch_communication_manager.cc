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
 * @file src/controller/launch_communication_manager.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class LaunchCommunicationManager.
 */

#include "controller/launch_communication_manager.h"

#include <string>

#include "message/message_include.h"

namespace canary {

void LaunchCommunicationManager::Initialize(
    network::EventMainThread* event_main_thread,
    LaunchReceiveCommandInterface* command_receiver,
    const std::string& launch_service) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  command_receiver_ = CHECK_NOTNULL(command_receiver);
  event_base_ = event_main_thread_->get_event_base();
  // Registers the launching listening port.
  listening_socket_ = network::allocate_and_bind_listen_socket(launch_service);
  CHECK_NE(listening_socket_, -1) << "Cannot listen on port " << launch_service
                                  << "!";
  // Starts the listening service.
  listening_event_ = CHECK_NOTNULL(
      evconnlistener_new(event_base_, &DispatchAcceptEvent, this,
                         LEV_OPT_CLOSE_ON_FREE, kBacklog, listening_socket_));
  evconnlistener_set_error_cb(listening_event_, DispatchAcceptErrorEvent);
  // Sets the initialization flag.
  is_initialized_ = true;
  VLOG(1) << "Launch communication manager is initialized.";
}

void LaunchCommunicationManager::Finalize() {
  CHECK_EQ(event_base_loopbreak(event_base_), 0);
}

void LaunchCommunicationManager::SendLaunchResponseCommand(
    LaunchCommandId launch_command_id, struct evbuffer* buffer) {
  CHECK(message::CheckIsIntegrateControlMessage(buffer));
  CHECK(message::ExamineControlHeader(buffer)->category_group ==
        message::MessageCategoryGroup::LAUNCH_RESPONSE_COMMAND);
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::AppendLaunchResponseCommandSendingQueue, this,
                launch_command_id, buffer));
}

/*
 * Public static methods used to dispatch event callbacks.
 */

void LaunchCommunicationManager::DispatchAcceptEvent(
    struct evconnlistener* listener, int socket_fd, struct sockaddr* address,
    int socklen, void* arg) {
  auto manager = reinterpret_cast<SelfType*>(arg);
  manager->CallbackAcceptEvent(listener, socket_fd, address, socklen);
}

void LaunchCommunicationManager::DispatchAcceptErrorEvent(
    struct evconnlistener*, void*) {
  LOG(FATAL) << "Failed to accept connection ("
             << network::get_error_message(network::get_last_error_number())
             << ").";
}

void LaunchCommunicationManager::DispatchReadEvent(int socket_fd,
                                                   short,  // NOLINT
                                                   void* arg) {
  auto launch_command_record = reinterpret_cast<LauncherRecord*>(arg);
  CHECK_EQ(socket_fd, launch_command_record->socket_fd);
  launch_command_record->manager->CallbackReadEvent(
      launch_command_record->launch_command_id, launch_command_record);
}

void LaunchCommunicationManager::DispatchWriteEvent(int socket_fd,
                                                    short,  // NOLINT
                                                    void* arg) {
  auto launch_command_record = reinterpret_cast<LauncherRecord*>(arg);
  CHECK_EQ(socket_fd, launch_command_record->socket_fd);
  launch_command_record->manager->CallbackWriteEvent(launch_command_record);
}

/*
 * Core logic to handle accept/read/write events.
 */

void LaunchCommunicationManager::CallbackAcceptEvent(
    struct evconnlistener* listener, int socket_fd, struct sockaddr*, int) {
  CHECK_EQ(listener, listening_event_);
  const LaunchCommandId launch_command_id = launch_command_id_allocator_++;
  VLOG(1) << "Received launching command (id=" << get_value(launch_command_id)
          << ").";
  InitializeLauncherRecord(launch_command_id, socket_fd);
}

void LaunchCommunicationManager::CallbackReadEvent(
    LaunchCommandId launch_command_id, LauncherRecord* launch_command_record) {
  struct evbuffer* receive_buffer = launch_command_record->receive_buffer;
  const int socket_fd = launch_command_record->socket_fd;
  int status = 0;
  while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
    while (struct evbuffer* whole_message =
               message::SegmentControlMessage(receive_buffer)) {
      command_receiver_->ReceiveLaunchCommand(launch_command_id, whole_message);
    }
  }
  if (status == 0 || (status == -1 && !network::is_blocked())) {
    CleanUpLauncherRecord(launch_command_record);
  }
}

void LaunchCommunicationManager::CallbackWriteEvent(
    LauncherRecord* launch_command_record) {
  launch_command_record->send_buffer = network::send_as_much(
      launch_command_record->socket_fd, launch_command_record->send_buffer,
      &launch_command_record->send_queue);
  // If not all data are sent.
  if (launch_command_record->send_buffer != nullptr) {
    // Channel is blocked or has error.
    if (network::is_blocked()) {
      CHECK_EQ(event_add(launch_command_record->write_event, nullptr), 0);
    } else {
      CleanUpLauncherRecord(launch_command_record);
    }
  }
}

void LaunchCommunicationManager::AppendLaunchResponseCommandSendingQueue(
    LaunchCommandId launch_command_id, struct evbuffer* buffer) {
  auto iter = launch_command_id_to_status_.find(launch_command_id);
  if (iter == launch_command_id_to_status_.end()) {
    evbuffer_free(buffer);
    LOG(WARNING) << "Launch command response is dropped!";
    return;
  } else {
    LauncherRecord* launch_command_record = &iter->second;
    launch_command_record->send_queue.push_back(buffer);
    event_add(launch_command_record->write_event, nullptr);
  }
}

void LaunchCommunicationManager::InitializeLauncherRecord(
    LaunchCommandId launch_command_id, int socket_fd) {
  CHECK(launch_command_id_to_status_.find(launch_command_id) ==
        launch_command_id_to_status_.end());
  auto& launch_command_record = launch_command_id_to_status_[launch_command_id];
  launch_command_record.launch_command_id = launch_command_id;
  launch_command_record.socket_fd = socket_fd;
  launch_command_record.read_event =
      CHECK_NOTNULL(event_new(event_base_, socket_fd, EV_READ | EV_PERSIST,
                              &DispatchReadEvent, &launch_command_record));
  CHECK_EQ(event_add(launch_command_record.read_event, nullptr), 0);
  launch_command_record.write_event =
      CHECK_NOTNULL(event_new(event_base_, socket_fd, EV_WRITE,
                              &DispatchWriteEvent, &launch_command_record));
  launch_command_record.send_buffer = nullptr;
  launch_command_record.receive_buffer = CHECK_NOTNULL(evbuffer_new());
  launch_command_record.manager = this;
}

void LaunchCommunicationManager::CleanUpLauncherRecord(
    LauncherRecord* launch_command_record) {
  if (launch_command_record->read_event) {
    event_free(launch_command_record->read_event);
    launch_command_record->read_event = nullptr;
  }
  if (launch_command_record->write_event) {
    event_free(launch_command_record->write_event);
    launch_command_record->write_event = nullptr;
  }
  // Caution socket must be closed after events are freed.
  if (launch_command_record->socket_fd >= 0) {
    network::close_socket(launch_command_record->socket_fd);
    launch_command_record->socket_fd = -1;
  }
  if (launch_command_record->send_buffer) {
    evbuffer_free(launch_command_record->send_buffer);
    launch_command_record->send_buffer = nullptr;
  }
  if (launch_command_record->receive_buffer) {
    evbuffer_free(launch_command_record->receive_buffer);
    launch_command_record->receive_buffer = nullptr;
  }
  for (auto buffer : launch_command_record->send_queue) {
    if (buffer) {
      evbuffer_free(buffer);
    }
  }
  launch_command_record->send_queue.clear();
  const auto launch_command_id = launch_command_record->launch_command_id;
  launch_command_id_to_status_.erase(launch_command_id);
}

}  // namespace canary
