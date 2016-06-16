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

DECLARE_STRING(controller_port);

namespace canary {

/**
 * The command sending interface at the controller side.
 */
class ControllerSendCommandInterface {
 public:
  //! Sends a command to a worker.
  virtual void SendCommandToWorker(WorkerId worker_id, Command command) = 0;
};

/**
 * The command receiving interface at the controller side, as callback
 * functions.
 */
class ControllerReceiveCommandInterface {
 public:
  //! Called when receiving a command from a worker.
  virtual void ReceiveCommandFromWorker(WorkerId worker_id,
                                        Command command) = 0;
  //! Called when a worker is down.
  virtual void NotifyWorkerIsDown(WorkerId worker_id) = 0;
  //! Called when a worker is up.
  virtual void NotifyWorkerIsUp(WorkerId worker_id) = 0;
};

/**
 * A controller communication manager runs on the controller, and implements
 * interfaces managing WorkerCommunicationManager and exchanging commands with
 * the workers.
 */
class ControllerCommunicationManager : ControllerSendCommandInterface {
 protected:
  /**
   * The data structure associated with a worker.
   */
  struct WorkerRecord {
    WorkerId worker_id = WorkerId::INVALID;
    std::string host, service, data_service, migrate_service;
    int socket_id = -1;
    bool is_ready = false;
    struct event* read_event = nullptr;
    struct event* write_event = nullptr;
    WorkerId next_worker_id_to_update = WorkerId::FIRST;
    PartitionMapVersion next_partition_map_version_to_update = WorkerId::FIRST;
  };

  /**
   * CallbackHelper.
   */
  struct CallbackHelper {
    CallbackHelper(ControllerCommunicationManager* input_manager,
                   WorkerRecord* input_worker_record)
        : manager(input_manager), worker_record(input_worker_record) {}
    ControllerCommunicationManager* manager;
    WorkerRecord* worker_record;
  };

 public:
  ControllerCommunicationManager() {}
  virtual ~ControllerCommunicationManager() {}

 public:
  //! Dispatch helper for accept a connection.
  static void DispatchCallbackAccept(
      struct evconnlistener* listener, int socket_fd, struct sockaddr* address,
      int socklen, void* arg) {
    auto manager = reinterpret_cast<ControllerCommunicationManager*>(arg);
    manager->CallbackAccept(listener, socket_fd, address, socklen);
  }

  //! Dispatch helper for reading event.
  static void DispatchCallbackRead(int socket_fd, short what, void* arg) {
    auto callback_arg = reinterpret_cast<CallbackHelper*>(arg);
    auto worker_record = callback_arg->worker_record;
    CHECK_EQ(socket_fd, worker_record->socket_fd);
    callback_arg->manager->CallbackRead(worker_record);
    delete callback_arg;
  }

  //! Dispatch helper for writing event.
  static void DispatchCallbackWrite(int socket_fd, short what, void* arg) {
    auto callback_arg = reinterpret_cast<CallbackHelper*>(arg);
    auto worker_record = callback_arg->worker_record;
    CHECK_EQ(socket_fd, worker_record->socket_fd);
    callback_arg->manager->CallbackWrite(worker_record);
    delete callback_arg;
  }

 protected:
  void CallbackAccept(struct evconnlistener* listener,
                      int socket_fd, struct sockaddr* address, int socklen) {
    CHECK_EQ(listener, listening_event_);
    network::translate_sockaddr_to_string(address, socklen, &host, &service);
    VLOG(2) << "Worker (" << host << ":" << service << ") "
        << "reaches the controller port.";
    WorkerId worker_id = worker_id_allocator_++;
    VLOG(2) << "Allocated WorkerId=" << get_value(worker_id);
    CHECK(worker_id_to_status_.find(worker_id) == worker_id_to_status_.end());
    auto worker_record = worker_id_to_status_[worker_id];

    worker_record->write_event = CHECK_NOTNULL(
        event_new(event_base_, socket_fd, EV_WRITE,
                  DispatchCallbackRead,
                  new CallbackHelper(this, worker_record)));
    worker_record->read_event = CHECK_NOTNULL(
        event_new(event_base_, socket_fd, EV_READ | EV_PERSISTENT,
                  DispatchCallbackWrite,
                  new CallbackHelper(this, worker_record)));
  }

  void CallbackRead(WorkerRecord* worker_record) {
    // Read the message.
    // If the message is RegisterServicePort, then trigger all write events to
    // update worker id.
    // If the message is NotifyWorkerDisconnect, then shutdown the worker if
    // necessary.
    // If the connection is down, clear up the worker.
  }

  void CallbackWrite(WorkerRecord* worker_record) {
  }

 protected:
  void ReceiveMessage(struct evbuffer* message) {
  }
  template<typename MessageType>
  void SendMessage(WorkerRecord* worker_record, MessageType* message) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(*message);
    }
    DataPlaneHeader header;
    header.set_length(evbuffer_get_length(buffer));
    header.set_category_group(get_message_category_group<MessageType>());
    header.set_category(get_message_category<MessageType>());
    evbuffer_prepend(buffer, header.content, header.kLength);
    // Length, Category, Type
    // Triggers the write event.
    CHECK_EQ(event_add(worker_record->write_event, nullptr), 0);
  }


 public:
  void Run() {
    event_base_ = CHECK_NOTNULL(event_base_new());
    listening_socket_ =
        network::allocate_and_bind_listen_socket(FLAGS_controller_port);
    CHECK_NE(listening_socket_, -1);
    listening_event_ = CHECK_NOTNULL(evconnlistenner_new(
            event_base_,
            &ControllerCommunicationManager::DispatchCallbackAccept, this,
            LEV_OPT_CLOSE_ON_FREE, -1, listening_socket_));
    event_base_dispatch(launch_base_);
    LOG(FATAL) << "Controller exited!";
  }

  //! Initializes the partition map for an application. Broadcast the map to all
  // workers.
  void InitializeApplicationPartitionMap(ApplicationId application_id,
                                         const PartitionMap& partition_map);
  //! Updates the partition map for an applicaiton. Broadcast the update to all
  // workers.
  void UpdateApplicationPartitionMap(
      ApplicationId application_id,
      const PartitionMapUpdate& partition_map_update);
  //! Shuts down a worker.
  void ShutDownWorker(WorkerId worker_id);
  //! Registers a command receiver.
  void RegisterControllerCommandReceiver(
      ControllerReceiveCommandInterface* command_receiver);

 private:
  WorkerId worker_id_allocator_ = WorkerId::FIRST;
  std::map<WorkerId, WorkerRecord*> worker_id_to_status_;

  struct event_base* event_base_ = nullptr;
  int listening_socket_ = -1;
  struct event* listening_event_ = nullptr;

};

// The message will be:
// MessageLength, MessageCategory, MessageType, data...
// The controller can:
// Assign worker_id.
// Send partition map update.
// Add a new worker to everyone.
// Shutdown a specific worker.

// The worker can:
// Notify a disconnected worker.
// Reply to assigned worker_id.

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_MANAGER_H_
