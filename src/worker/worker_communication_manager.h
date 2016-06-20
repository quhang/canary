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
 * @file src/worker/worker_communication_manager.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerCommunicationManager.
 * WorkerCommunicationManager talks with each other, and with
 * ControllerCommunicationManager.
 * @see src/controller/controller_communication_manager
 */

#ifndef CANARY_SRC_WORKER_WORKER_COMMUNICATION_MANAGER_H_
#define CANARY_SRC_WORKER_WORKER_COMMUNICATION_MANAGER_H_

#include "worker/worker_communication_interface.h"
#include "worker/worker_data_router.h"
#include "message/message_include.h"

namespace canary {

/**
 * A worker communication manager runs on each worker, and implements interfaces
 * for exchanging data between workers and exchanging commands with the
 * controller.
 *
 * @see src/worker/worker_communication_interface.h
 */
class WorkerCommunicationManager
    : public WorkerSendCommandInterface, public WorkerSendDataInterface {
 private:
  typedef WorkerCommunicationManager SelfType;
  /**
   * Controller record.
   */
  struct ControllerRecord {
    //! Controller host port.
    std::string host;
    //! Controller service port.
    std::string service;
    //! Assigned worker id.
    WorkerId assigned_worker_id;
    //! Controller communication socket.
    int socket_fd = -1;
    //! Whether the channel is ready.
    bool is_ready = false;
    //! Persistent read event.
    struct event* read_event = nullptr;
    //! Write event.
    struct event* write_event = nullptr;
    //! Send buffer, which is grabbed from the sending queue.
    struct evbuffer* send_buffer = nullptr;
    //! Send queue, which owns all the buffers.
    std::list<struct evbuffer*> send_queue;
    //! Receive buffer.
    struct evbuffer* receive_buffer = nullptr;
  } controller_record;

 public:
  //! Constructor.
  WorkerCommunicationManager() {}
  //! Destructor.
  virtual ~WorkerCommunicationManager() {}

  //! Initializes the communication manager.
  // Prepare call.
  void Initialize(network::EventMainThread* event_main_thread,
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
    event_main_thread_->AddInjectedEvent(std::bind(
            &SelfType::CallbackInitiateEvent, this));
    // Sets the initialization flag.
    is_initialized_ = true;
  }

 public:
  /*
   * Public static methods used to dispatch event callbacks.
   * Controller channel related.
   */

  //! Dispatches the connection initiation event on the controller channel.
  // Sync call.
  static void DispatchInitiateEvent(int, short,  // NOLINT
                                    void* arg) {
    auto manager = reinterpret_cast<SelfType*>(arg);
    manager->CallbackInitiateEvent();
  }

  //! Dispatches the connection feedback event on the controller channel.
  // Sync call.
  static void DispatchConnectEvent(int, short,  // NOLINT
                                   void* arg) {
    auto manager = reinterpret_cast<SelfType*>(arg);
    manager->CallbackConnectEvent();
  }

  //! Dispatches the read event on the controller channel.
  // Sync call.
  static void DispatchReadEvent(int, short,  // NOLINT
                                void* arg) {
    auto manager = reinterpret_cast<SelfType*>(arg);
    manager->CallbackReadEvent();
  }

  //! Dispatches the write event on the controller channel.
  // Sync call.
  static void DispatchWriteEvent(int, short,  // NOLINT
                                void* arg) {
    auto manager = reinterpret_cast<SelfType*>(arg);
    manager->CallbackWriteEvent();
  }

 private:
  /*
   * Core logic to handle connect/read/write events.
   */
  //! Timeout event to initiate the connection to the controller.
  // Sync call.
  void CallbackInitiateEvent() {
    controller_record.socket_fd = network::allocate_and_connect_socket(
        controller_record.host, controller_record.service);
    CHECK_GE(controller_record.socket_fd, 0);
    CHECK_EQ(event_base_once(
            event_base_, controller_record.socket_fd, EV_WRITE,
            DispatchConnectEvent, this, nullptr), 0);
  }

  //! Receives the connection feedback event from the controller.
  // Sync call.
  void CallbackConnectEvent() {
    const int error_flag = network::get_socket_error_number(
        controller_record.socket_fd);
    if (error_flag != 0) {
      // The socket is in error state, so has to reopen it.
      close(controller_record.socket_fd);
      controller_record.socket_fd = -1;
      event_main_thread_->AddDelayInjectedEvent(std::bind(
              &SelfType::CallbackInitiateEvent, this));
    } else {
      InitializeControllerRecord();
    }
  }

  void InitializeControllerRecord() {
    const int socket_fd = controller_record.socket_fd;
    CHECK_NE(socket_fd, -1);
    controller_record.is_ready = false;
    controller_record.read_event = CHECK_NOTNULL(event_new(
            event_base_, socket_fd, EV_READ | EV_PERSIST, &DispatchReadEvent,
            this));
    event_add(controller_record.read_event, nullptr);
    controller_record.write_event = CHECK_NOTNULL(event_new(
            event_base_, socket_fd, EV_WRITE, &DispatchWriteEvent, this));
    controller_record.send_buffer = nullptr;
    controller_record.receive_buffer = evbuffer_new();
  }


  //! Receives data or error from the controller.
  // Sync call.
  void CallbackReadEvent() {
    struct evbuffer* receive_buffer = controller_record.receive_buffer;
    const int socket_fd = controller_record.socket_fd;

    message::ControlHeader message_header;
    int status = 0;
    while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
      if (struct evbuffer* whole_message =
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
  void CallbackWriteEvent() {
    controller_record.send_buffer = network::send_as_much(
        controller_record.socket_fd,
        controller_record.send_buffer,
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

 private:
  void ProcessIncomingMessage(
      const message::ControlHeader& message_header, struct evbuffer* buffer) {
    using message::MessageCategoryGroup;
    using message::MessageCategory;
    using message::ControlHeader;
    // If assigned worker_id.
    // Differentiante between control messages and command messages.
    switch (message_header.get_category_group()) {
      case MessageCategoryGroup::DATA_PLANE_CONTROL:
        switch (message_header.get_category()) {
#define PROCESS_MESSAGE(TYPE, METHOD)  \
    case MessageCategory::TYPE: { auto message =  \
      message::ControlHeader::UnpackMessage<MessageCategory::TYPE>(buffer);  \
      METHOD(message); break; }
          PROCESS_MESSAGE(ASSIGN_WORKER_ID, ProcessAssignWorkerIdMessage);
          PROCESS_MESSAGE(UPDATE_PARTITION_MAP_AND_WORKER,
                          ProcessUpdatePartitionMapAndWorkerMessage);
          PROCESS_MESSAGE(UPDATE_PARTITION_MAP_ADD_APPLICATION,
                          ProcessUpdatePartitionMapAddApplicationMessage);
          PROCESS_MESSAGE(UPDATE_PARTITION_MAP_DROP_APPLICATION,
                          ProcessUpdatePartitionMapDropApplicationMessage);
          PROCESS_MESSAGE(UPDATE_PARTITION_MAP_INCREMENTAL,
                          ProcessUpdatePartitionMapIncrementalMessage);
          PROCESS_MESSAGE(UPDATE_ADDED_WORKER,
                          ProcessUpdateAddedWorkerMessage);
          PROCESS_MESSAGE(SHUT_DOWN_WORKER,
                          ProcessShutDownWorkerMessage);
          default:
            LOG(FATAL) << "Unexpected message type.";
        }  // switch category.
#undef PROCESS_MESSAGE
        break;
      case MessageCategoryGroup::WORKER_COMMAND:
        command_receiver_->ReceiveCommandFromController(buffer);
        break;
      default:
        LOG(FATAL) << "Invalid message header!";
    }  // switch category group.
  }

  void ProcessAssignWorkerIdMessage(message::AssignWorkerId* message) {
    controller_record.assigned_worker_id = message->assigned_worker_id;
    delete message;

    message::RegisterServicePort update_message;
    update_message.from_worker_id = controller_record.assigned_worker_id;
    update_message.route_service = route_service_;
    struct evbuffer* buffer
        = message::ControlHeader::PackMessage(update_message);
    AppendSendingQueue(buffer, false);

    controller_record.is_ready = true;
  }
  void ProcessUpdatePartitionMapAndWorkerMessage(
      message::UpdatePartitionMapAndWorker* message) {
    delete message;
  }
  void ProcessUpdatePartitionMapAddApplicationMessage(
      message::UpdatePartitionMapAddApplication* message) {
    delete message;
  }
  void ProcessUpdatePartitionMapDropApplicationMessage(
      message::UpdatePartitionMapDropApplication* message) {
    delete message;
  }
  void ProcessUpdatePartitionMapIncrementalMessage(
      message::UpdatePartitionMapIncremental* message) {
    delete message;
  }
  void ProcessUpdateAddedWorkerMessage(
      message::UpdateAddedWorker* message) {
    delete message;
  }
  void ProcessShutDownWorkerMessage(
      message::ShutDownWorker* message) {
    delete message;
  }

  void AppendSendingQueue(struct evbuffer* buffer,
                          bool enforce_controller_ready) {
    CHECK(controller_record.is_ready || !enforce_controller_ready);
    controller_record.send_queue.push_back(buffer);
    event_add(controller_record.write_event, nullptr);
  }

 public:
  void SendCommandToController(struct evbuffer* buffer) override {
    message::ControlHeader header;
    CHECK(header.ExtractHeader(buffer));
    CHECK(header.get_category_group() ==
          message::MessageCategoryGroup::CONTROLLER_COMMAND);
    CHECK_EQ(header.get_length(), evbuffer_get_length(buffer) + header.kLength);
    event_main_thread_->AddInjectedEvent(std::bind(
            &SelfType::AppendSendingQueue, this, buffer, true));
  }

  void SendDataToPartition(ApplicationId application_id,
                           StageId stage_id, PartitionId partition_id,
                           struct evbuffer*) override {
  }

  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override {
  }

  void ReduceAndSendDataToPartition(ApplicationId application_id,
                                    StageId stage_id,
                                    struct evbuffer* buffer,
                                    CombinerFunction combiner_function) override {
  }

  void BroadcastDatatoPartition(ApplicationId application_id,
                                StageId stage_id,
                                struct evbuffer* buffer) override {
  }

 protected:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  WorkerReceiveCommandInterface* command_receiver_ = nullptr;
  bool is_initialized_ = false;

  WorkerDataRouter data_router_;
  std::string route_service_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_COMMUNICATION_MANAGER_H_
