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
 * @file src/controller/controller_communication_manager.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerCommunicationManager. ControllerCommunicationManager
 * talks with WorkerCommunicationManager.
 * @see src/worker/worker_communication_manager
 */

#ifndef CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_MANAGER_H_
#define CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_MANAGER_H_

#include <event2/listener.h>

#include <list>
#include <map>
#include <string>

#include "shared/internal.h"
#include "shared/partition_map.h"
#include "message/message_include.h"

namespace canary {

/**
 * The command sending interface at the controller side.
 */
class ControllerSendCommandInterface {
 public:
  //! Sends a command to a worker as long as the worker is notified up. The
  // buffer ownership is transferred.
  virtual void SendCommandToWorker(WorkerId worker_id,
                                   struct evbuffer* buffer) = 0;

  /*
   * The partition map will be synchronized to workers automatically.
   */
  //! Updates the partition map by adding an application.
  virtual void AddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map) = 0;

  //! Updates the partition map by dropping an application.
  virtual void DropApplication(ApplicationId application_id) = 0;

  //! Updates the partition map incrementally.
  virtual void UpdatePartitionMap(PartitionMapUpdate* partition_map_update) = 0;

  //! Shuts down a worker.
  virtual void ShutDownWorker(WorkerId worker_id) = 0;
};

/**
 * The command receiving interface at the controller side, as callback
 * functions.
 */
class ControllerReceiveCommandInterface {
 public:
  //! Called when receiving a command. The message header is kept, and the
  // buffer ownership is transferred.
  virtual void ReceiveCommand(struct evbuffer* buffer) = 0;

  //! Called when a worker is down, even if it is shut down by the controller.
  virtual void NotifyWorkerIsDown(WorkerId worker_id) = 0;

  //! Called when a worker is up.
  virtual void NotifyWorkerIsUp(WorkerId worker_id) = 0;
};

/**
 * A controller communication manager runs on the controller, and is responsible
 * for exchanging commands with workers, managing worker membership and
 * partition map synchronization.
 */
class ControllerCommunicationManager : public ControllerSendCommandInterface {
 private:
  /**
   * The data structure associated with a worker.
   */
  struct WorkerRecord {
    //! The id of the connected worker.
    WorkerId worker_id = WorkerId::INVALID;
    //! Worker hostname.
    std::string host;
    //! Worker port names.
    std::string service, route_service, transmit_service;
    //! Worker-controller communication socket.
    int socket_fd = -1;
    //! Whether the worker has replied with route/transmit ports.
    bool is_ready = false;
    //! Persistent read event.
    struct event* read_event = nullptr;
    //! Write event.
    struct event* write_event = nullptr;
    //! Send buffer, which is grabbed from the sending queue.
    struct evbuffer* send_buffer = nullptr;
    //! Send queue, which owns all the buffers.
    std::list<struct evbuffer*> sending_queue;
    //! Receive buffer.
    struct evbuffer* receive_buffer = nullptr;
  };

  /**
   * Used to pass argument during callback.
   */
  struct WorkerRecordEventArg {
    ControllerCommunicationManager* manager;
    WorkerRecord* worker_record;
  };

  //! Default backlog argument. Concurrent pending TCP connecting events.
  static const int kBacklog = -1;

 public:
  //! Constructor.
  ControllerCommunicationManager() {}
  //! Destructor.
  virtual ~ControllerCommunicationManager() {}

  //! Initializes the communication manager.
  // Prepare call.
  void Initialize(network::EventMainThread* event_main_thread,
                  ControllerReceiveCommandInterface* command_receiver);

 public:
  /*
   * Public static methods used to dispatch event callbacks.
   */

  //! Dispatches the accept event of the listening controller port.
  // Sync call.
  static void DispatchAcceptEvent(struct evconnlistener* listener,
                                  int socket_fd, struct sockaddr* address,
                                  int socklen, void* arg);

  //! Dispatches the accept error event of the listening controller port.
  // Sync call.
  static void DispatchAcceptErrorEvent(struct evconnlistener*, void*);

  //! Dispatches the read event on a worker channel.
  // Sync call.
  static void DispatchReadEvent(int socket_fd, short, void* arg);  // NOLINT

  //! Dispatches the write event on a worker channel.
  // Sync call.
  static void DispatchWriteEvent(int socket_fd, short, void* arg);  // NOLINT

 private:
  /*
   * Core logic to handle accept/read/write events.
   */

  //! Receives the accept event of the listening controller port.
  // Sync call.
  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen);

  //! Receives data or error on a socket to a worker.
  // Sync call.
  void CallbackReadEvent(WorkerRecord* worker_record);

  //! Sends data to a worker.
  // Sync call.
  void CallbackWriteEvent(WorkerRecord* worker_record);

 private:
  /*
   * Lifetime of a worker record.
   */

  //! Initializes a worker record.
  // Sync call.
  void InitializeWorkerRecord(WorkerId worker_id, int socket_fd,
                              const std::string& host,
                              const std::string& service);

  //! Activates a worker record.
  // Sync call.
  void ProcessRegisterServicePortMessage(message::RegisterServicePort* message);

  //! Cleans up a worker record if it is disconnected..
  // Sync call.
  void ProcessNotifyWorkerDisconnect(message::NotifyWorkerDisconnect* message);

  //! Cleans up a worker record.
  // Sync call.
  void CleanUpWorkerRecord(WorkerRecord* worker_record);

 public:
  /*
   * Public async interfaces.
   */

  //! Sends a command to a worker.
  // Async call.
  void SendCommandToWorker(WorkerId worker_id,
                           struct evbuffer* buffer) override;

  //! Updates the partition map by adding an application.
  // Async call.
  void AddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map) override;

  //! Updates the partition map by dropping an application.
  // Async call.
  void DropApplication(ApplicationId application_id) override;

  //! Updates the partition map incrementally.
  // Async call.
  void UpdatePartitionMap(PartitionMapUpdate* partition_map_update) override;

  //! Shuts down a worker.
  // Async call.
  void ShutDownWorker(WorkerId worker_id) override;

 private:
  /**
   * Implements async interfaces as synchronous callback.
   */

  //! Updates the partition map by adding an application.
  // Sync call.
  void InternalAddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map);

  //! Updates the partition map by dropping an application.
  // Sync call.
  void InternalDropApplication(ApplicationId application_id);

  //! Updates the partition map incrementally.
  // Sync call.
  void InternalUpdatePartitionMap(PartitionMapUpdate* partition_map_update);

  //! Shuts down a worker.
  // Sync call.
  void InternalShutDownWorker(WorkerId worker_id);

 private:
  /**
   * Internal helper functions.
   */

  //! Processes an incoming message.
  // Sync call.
  void ProcessIncomingMessage(const message::DataPlaneHeader& message_header,
                              struct evbuffer* buffer);

  //! Appends the worker sending queue.
  // Sync call.
  void AppendWorkerSendingQueue(WorkerId worker_id, struct evbuffer* buffer);

  //! Appends all ready sending queues.
  // Sync call.
  void AppendAllReadySendingQueue(struct evbuffer* buffer);

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  ControllerReceiveCommandInterface* command_receiver_ = nullptr;

  bool is_initialized_ = false;

  // Worker id keeps increasing and is not reused.
  WorkerId worker_id_allocator_ = WorkerId::FIRST;
  std::map<WorkerId, WorkerRecord*> worker_id_to_status_;

  PartitionMap internal_partition_map_;
  PartitionMapVersion internal_partition_map_version_ =
      PartitionMapVersion::FIRST;

  int listening_socket_ = -1;
  struct evconnlistener* listening_event_ = nullptr;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_MANAGER_H_
