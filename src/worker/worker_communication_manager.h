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

#include <list>
#include <string>

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
class WorkerCommunicationManager : public WorkerSendCommandInterface,
                                   public WorkerSendDataInterface {
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
                  WorkerReceiveDataInterface* data_receiver);

 public:
  /*
   * Public static methods used to dispatch event callbacks.
   * Controller channel related.
   */

  //! Dispatches the connection feedback event on the controller channel.
  // Sync call.
  static void DispatchConnectEvent(int, short, void* arg);  // NOLINT

  //! Dispatches the read event on the controller channel.
  // Sync call.
  static void DispatchReadEvent(int, short, void* arg);  // NOLINT

  //! Dispatches the write event on the controller channel.
  // Sync call.
  static void DispatchWriteEvent(int, short, void* arg);  // NOLINT

 private:
  /*
   * Core logic to handle connect/read/write events.
   */
  //! Timeout event to initiate the connection to the controller.
  // Sync call.
  void CallbackInitiateEvent();

  //! Receives the connection feedback event from the controller.
  // Sync call.
  void CallbackConnectEvent();

  //! Initializes the controller record.
  void InitializeControllerRecord();

  //! Receives data or error from the controller.
  // Sync call.
  void CallbackReadEvent();

  //! Sends data to the controller.
  // Sync call.
  void CallbackWriteEvent();

 private:
  /*
   * Process incoming messages.
   */
  //! Process an incoming message.
  // Sync call.
  void ProcessIncomingMessage(const message::ControlHeader& message_header,
                              struct evbuffer* buffer);

  // All the following messages are data plane control messages.
  void ProcessAssignWorkerIdMessage(message::AssignWorkerId* message);
  void ProcessUpdatePartitionMapAndWorkerMessage(
      message::UpdatePartitionMapAndWorker* message);
  void ProcessUpdatePartitionMapAddApplicationMessage(
      message::UpdatePartitionMapAddApplication* message);
  void ProcessUpdatePartitionMapDropApplicationMessage(
      message::UpdatePartitionMapDropApplication* message);
  void ProcessUpdatePartitionMapIncrementalMessage(
      message::UpdatePartitionMapIncremental* message);
  void ProcessUpdateAddedWorkerMessage(message::UpdateAddedWorker* message);
  void ProcessShutDownWorkerMessage(message::ShutDownWorker* message);

 private:
  /*
   * Helper functions.
   */
  //! Append a message to the sending queue.
  // Sync call.
  void AppendSendingQueue(struct evbuffer* buffer,
                          bool enforce_controller_ready);

 public:
  /*
   * Command sending facility.
   */

  //! Sends a command to the controller.
  // Async call.
  void SendCommandToController(struct evbuffer* buffer) override;

 public:
  /*
   * Data routing facilities are implemented by the data router.
   * Async calls.
   * @see src/worker/worker_data_router.h
   */

  void SendDataToPartition(ApplicationId application_id, StageId stage_id,
                           PartitionId partition_id,
                           struct evbuffer* buffer) override {
    data_router_.SendDataToPartition(application_id, stage_id, partition_id,
                                     buffer);
  }

  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override {
    data_router_.SendDataToWorker(worker_id, buffer);
  }

  void ReduceAndSendDataToPartition(
      ApplicationId application_id, StageId stage_id, struct evbuffer* buffer,
      CombinerFunction combiner_function) override {
    data_router_.ReduceAndSendDataToPartition(application_id, stage_id, buffer,
                                              combiner_function);
  }

  void BroadcastDatatoPartition(ApplicationId application_id, StageId stage_id,
                                struct evbuffer* buffer) override {
    data_router_.BroadcastDatatoPartition(application_id, stage_id, buffer);
  }

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  WorkerReceiveCommandInterface* command_receiver_ = nullptr;
  bool is_initialized_ = false;

  WorkerDataRouter data_router_;
  std::string route_service_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_COMMUNICATION_MANAGER_H_
