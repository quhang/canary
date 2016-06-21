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
 * @file src/worker/worker_communication_manager.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerCommunicationManager.
 */

#include "worker/worker_communication_manager.h"

namespace canary {

void WorkerCommunicationManager::Initialize(
    network::EventMainThread* event_main_thread,
    WorkerReceiveCommandInterface* command_receiver,
    WorkerReceiveDataInterface* data_receiver) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  command_receiver_ = command_receiver;
  event_base_ = event_main_thread_->get_event_base();
  route_service_ = FLAGS_worker_service;
  data_router_.Initialize(event_main_thread_, data_receiver, route_service_);
  // Initialize host and service, and initiate connection.
  controller_record.host = FLAGS_controller_host;
  controller_record.service = FLAGS_controller_service;
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::CallbackInitiateEvent, this));
  // Sets the initialization flag.
  is_initialized_ = true;
}

/*
 * Public static methods used to dispatch event callbacks.
 * Controller channel related.
 */

void WorkerCommunicationManager::DispatchConnectEvent(int, short,  // NOLINT
                                                      void* arg) {
  auto manager = reinterpret_cast<SelfType*>(arg);
  manager->CallbackConnectEvent();
}

void WorkerCommunicationManager::DispatchReadEvent(int, short,  // NOLINT
                                                   void* arg) {
  auto manager = reinterpret_cast<SelfType*>(arg);
  manager->CallbackReadEvent();
}

void WorkerCommunicationManager::DispatchWriteEvent(int, short,  // NOLINT
                                                    void* arg) {
  auto manager = reinterpret_cast<SelfType*>(arg);
  manager->CallbackWriteEvent();
}

/*
 * Core logic to handle connect/read/write events.
 */
void WorkerCommunicationManager::CallbackInitiateEvent() {
  controller_record.socket_fd = network::allocate_and_connect_socket(
      controller_record.host, controller_record.service);
  CHECK_GE(controller_record.socket_fd, 0);
  // Triggered when the channel is ready to write.
  CHECK_EQ(event_base_once(event_base_, controller_record.socket_fd, EV_WRITE,
                           DispatchConnectEvent, this, nullptr),
           0);
}

void WorkerCommunicationManager::CallbackConnectEvent() {
  const int error_flag =
      network::get_socket_error_number(controller_record.socket_fd);
  if (error_flag != 0) {
    // The socket is in error state, so has to reopen it.
    network::close_socket(controller_record.socket_fd);
    controller_record.socket_fd = -1;
    event_main_thread_->AddDelayInjectedEvent(
        std::bind(&SelfType::CallbackInitiateEvent, this));
  } else {
    InitializeControllerRecord();
  }
}

void WorkerCommunicationManager::InitializeControllerRecord() {
  const int socket_fd = controller_record.socket_fd;
  CHECK_NE(socket_fd, -1);
  controller_record.is_ready = false;
  controller_record.read_event = CHECK_NOTNULL(event_new(
      event_base_, socket_fd, EV_READ | EV_PERSIST, &DispatchReadEvent, this));
  CHECK_EQ(event_add(controller_record.read_event, nullptr), 0);
  controller_record.write_event = CHECK_NOTNULL(
      event_new(event_base_, socket_fd, EV_WRITE, &DispatchWriteEvent, this));
  controller_record.send_buffer = nullptr;
  controller_record.receive_buffer = evbuffer_new();
}

void WorkerCommunicationManager::CallbackReadEvent() {
  struct evbuffer* receive_buffer = controller_record.receive_buffer;
  const int socket_fd = controller_record.socket_fd;

  message::ControlHeader message_header;
  int status = 0;
  while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
    while (struct evbuffer* whole_message =
               message_header.SegmentMessage(receive_buffer)) {
      ProcessIncomingMessage(message_header, whole_message);
    }
  }
  if (status == 0 || (status == 1 && !network::is_blocked())) {
    LOG(FATAL) << "Controller connection is down!";
  }
}

//! Sends data to the controller.
// Sync call.
void WorkerCommunicationManager::CallbackWriteEvent() {
  controller_record.send_buffer = network::send_as_much(
      controller_record.socket_fd, controller_record.send_buffer,
      &controller_record.send_queue);
  if (controller_record.send_buffer != nullptr) {
    // Channel is blocked or has error.
    if (network::is_blocked()) {
      CHECK_EQ(event_add(controller_record.write_event, nullptr), 0);
    } else {
      LOG(FATAL) << "Controller connection is down!";
    }
  }
}

/*
 * Process incoming messages.
 */

// Macro used to dispatch a message type to a method.
#define PROCESS_MESSAGE(TYPE, METHOD)                                         \
  case MessageCategory::TYPE: {                                               \
    auto message =                                                            \
        message::ControlHeader::UnpackMessage<MessageCategory::TYPE>(buffer); \
    METHOD(message);                                                          \
    break;                                                                    \
  }
void WorkerCommunicationManager::ProcessIncomingMessage(
    const message::ControlHeader& message_header, struct evbuffer* buffer) {
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  using message::ControlHeader;
  // If assigned worker_id.
  // Differentiante between control messages and command messages.
  switch (message_header.get_category_group()) {
    case MessageCategoryGroup::DATA_PLANE_CONTROL:
      switch (message_header.get_category()) {
        PROCESS_MESSAGE(ASSIGN_WORKER_ID, ProcessAssignWorkerIdMessage);
        PROCESS_MESSAGE(UPDATE_PARTITION_MAP_AND_WORKER,
                        ProcessUpdatePartitionMapAndWorkerMessage);
        PROCESS_MESSAGE(UPDATE_PARTITION_MAP_ADD_APPLICATION,
                        ProcessUpdatePartitionMapAddApplicationMessage);
        PROCESS_MESSAGE(UPDATE_PARTITION_MAP_DROP_APPLICATION,
                        ProcessUpdatePartitionMapDropApplicationMessage);
        PROCESS_MESSAGE(UPDATE_PARTITION_MAP_INCREMENTAL,
                        ProcessUpdatePartitionMapIncrementalMessage);
        PROCESS_MESSAGE(UPDATE_ADDED_WORKER, ProcessUpdateAddedWorkerMessage);
        PROCESS_MESSAGE(SHUT_DOWN_WORKER, ProcessShutDownWorkerMessage);
        default:
          LOG(FATAL) << "Unexpected message type.";
      }  // switch category.
      break;
    case MessageCategoryGroup::WORKER_COMMAND:
      command_receiver_->ReceiveCommandFromController(buffer);
      break;
    default:
      LOG(FATAL) << "Invalid message header!";
  }  // switch category group.
}
#undef PROCESS_MESSAGE

void WorkerCommunicationManager::ProcessAssignWorkerIdMessage(
    message::AssignWorkerId* message) {
  controller_record.assigned_worker_id = message->assigned_worker_id;

  LOG(INFO) << "Assigned worker id="
            << get_value(controller_record.assigned_worker_id);

  message::RegisterServicePort update_message;
  update_message.from_worker_id = controller_record.assigned_worker_id;
  update_message.route_service = route_service_;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(update_message);
  AppendSendingQueue(buffer, false);

  // The controller channel is ready.
  controller_record.is_ready = true;

  // Nofity the command receiver.
  command_receiver_->AssignWorkerId(controller_record.assigned_worker_id);

  delete message;
}

void WorkerCommunicationManager::ProcessUpdatePartitionMapAndWorkerMessage(
    message::UpdatePartitionMapAndWorker* message) {
  LOG(INFO) << "Update1";
  delete message;
}

void WorkerCommunicationManager::ProcessUpdatePartitionMapAddApplicationMessage(
    message::UpdatePartitionMapAddApplication* message) {
  LOG(INFO) << "Update2";
  delete message;
}

void WorkerCommunicationManager::
    ProcessUpdatePartitionMapDropApplicationMessage(
        message::UpdatePartitionMapDropApplication* message) {
  LOG(INFO) << "Update3";
  delete message;
}

void WorkerCommunicationManager::ProcessUpdatePartitionMapIncrementalMessage(
    message::UpdatePartitionMapIncremental* message) {
  LOG(INFO) << "Update4";
  delete message;
}

void WorkerCommunicationManager::ProcessUpdateAddedWorkerMessage(
    message::UpdateAddedWorker* message) {
  LOG(INFO) << "Update5";
  delete message;
}

void WorkerCommunicationManager::ProcessShutDownWorkerMessage(
    message::ShutDownWorker* message) {
  LOG(INFO) << "Update6";
  delete message;
}

/*
 * Helper functions.
 */

void WorkerCommunicationManager::AppendSendingQueue(
    struct evbuffer* buffer, bool enforce_controller_ready) {
  CHECK(controller_record.is_ready || !enforce_controller_ready);
  controller_record.send_queue.push_back(buffer);
  CHECK_EQ(event_add(controller_record.write_event, nullptr), 0);
}

/*
 * Command sending facility.
 */
void WorkerCommunicationManager::SendCommandToController(
    struct evbuffer* buffer) {
  message::ControlHeader header;
  CHECK(header.ExtractHeader(buffer));
  CHECK(header.get_category_group() ==
        message::MessageCategoryGroup::CONTROLLER_COMMAND);
  CHECK_EQ(header.kLength + header.get_length(), evbuffer_get_length(buffer));
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::AppendSendingQueue, this, buffer, true));
}

}  // namespace canary
