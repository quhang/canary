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

DECLARE_string(controller_port);

namespace canary {

/**
 * The command sending interface at the controller side.
 */
class ControllerSendCommandInterface {
 public:
  //! Sends a command to a worker. The buffer ownership is transferred.
  virtual void SendCommandToWorker(WorkerId worker_id,
                                   struct evbuffer* buffer) = 0;
};

/**
 * The command receiving interface at the controller side, as callback
 * functions.
 */
class ControllerReceiveCommandInterface {
 public:
  //! Called when receiving a command. The buffer header is kept. The buffer
  // ownership is transferred.
  virtual void ReceiveCommand(struct evbuffer* buffer) = 0;
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
                  ControllerReceiveCommandInterface* command_receiver) {
    event_main_thread_ = CHECK_NOTNULL(event_main_thread);
    command_receiver_ = CHECK_NOTNULL(command_receiver);
    event_base_ = event_main_thread_->get_event_base();
    listening_socket_ =
        network::allocate_and_bind_listen_socket(FLAGS_controller_port);
    CHECK_NE(listening_socket_, -1);
    listening_event_ = CHECK_NOTNULL(
        evconnlistener_new(event_base_, &DispatchAcceptEvent, this,
                           LEV_OPT_CLOSE_ON_FREE, kBacklog, listening_socket_));
    evconnlistener_set_error_cb(listening_event_, DispatchAcceptErrorEvent);
    is_initialized_ = true;
  }

 public:
  /*
   * Public static methods used to dispatch event callbacks.
   */

  //! Dispatches the accept event of the listening controller port.
  // Sync call.
  static void DispatchAcceptEvent(struct evconnlistener* listener,
                                  int socket_fd, struct sockaddr* address,
                                  int socklen, void* arg) {
    auto manager = reinterpret_cast<ControllerCommunicationManager*>(arg);
    manager->CallbackAcceptEvent(listener, socket_fd, address, socklen);
  }

  //! Dispatches the accept error event of the listening controller port.
  // Sync call.
  static void DispatchAcceptErrorEvent(struct evconnlistener*, void*) {
    LOG(FATAL) << "Failure on the listener: "
               << network::get_error_message(network::get_last_error_number());
  }

  //! Dispatches the read event on a worker channel.
  // Sync call.
  static void DispatchReadEvent(int socket_fd, short, void* arg) {  // NOLINT
    auto callback_arg = reinterpret_cast<WorkerRecordEventArg*>(arg);
    auto worker_record = callback_arg->worker_record;
    CHECK_EQ(socket_fd, worker_record->socket_fd);
    callback_arg->manager->CallbackReadEvent(worker_record);
    delete callback_arg;
  }

  //! Dispatches the write event on a worker channel.
  // Sync call.
  static void DispatchWriteEvent(int socket_fd, short, void* arg) {  // NOLINT
    auto callback_arg = reinterpret_cast<WorkerRecordEventArg*>(arg);
    auto worker_record = callback_arg->worker_record;
    CHECK_EQ(socket_fd, worker_record->socket_fd);
    callback_arg->manager->CallbackWriteEvent(worker_record);
    delete callback_arg;
  }

 private:
  /*
   * Core logic to handle accept/read/write events.
   */

  //! Receives the accept event of the listening controller port.
  // Sync call.
  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen) {
    CHECK_EQ(listener, listening_event_);
    std::string host, service;
    network::translate_sockaddr_to_string(address, socklen, &host, &service);
    VLOG(2) << host << ":" << service << " reaches the controller port.";
    const WorkerId worker_id = worker_id_allocator_++;
    VLOG(2) << "Allocated WorkerId=" << get_value(worker_id);
    InitializeWorkerRecord(worker_id, socket_fd, host, service);

    // Sends initialization message.
    message::AssignWorkerId message;
    message.assigned_worker_id = worker_id;
    struct evbuffer* buffer = message::DataPlaneHeader::PackMessage(message);
    AppendWorkerSendingQueue(worker_id, buffer);
  }

  //! Receives data or error on a socket to a worker.
  // Sync call.
  void CallbackReadEvent(WorkerRecord* worker_record) {
    message::DataPlaneHeader message_header;
    struct evbuffer* receive_buffer = worker_record->receive_buffer;
    while (evbuffer_read(receive_buffer, worker_record->socket_fd, -1) != -1) {
      // If the full header is received.
      if (message_header.ExtractHeader(receive_buffer)) {
        const size_t full_length =
            message_header.kLength + message_header.get_length();
        // If the full message is received.
        if (full_length <= evbuffer_get_length(receive_buffer)) {
          struct evbuffer* result = evbuffer_new();
          const int bytes =
              evbuffer_remove_buffer(receive_buffer, result, full_length);
          CHECK_GE(bytes, 0);
          CHECK_EQ(static_cast<size_t>(bytes), full_length);
          switch (message_header.get_category_group()) {
            case message::MessageCategoryGroup::DATA_PLANE_CONTROL:
              switch (message_header.get_category()) {
                case message::MessageCategory::REGISTER_SERVICE_PORT: {
                  auto message = message::DataPlaneHeader::UnpackMessage<
                      message::MessageCategory::REGISTER_SERVICE_PORT>(result);
                  ProcessRegisterServicePortMessage(message);
                  break;
                }
                case message::MessageCategory::NOTIFY_WORKER_DISCONNECT: {
                  auto message = message::DataPlaneHeader::UnpackMessage<
                      message::MessageCategory::NOTIFY_WORKER_DISCONNECT>(
                      result);
                  ProcessNotifyWorkerDisconnect(message);
                  break;
                }
                default:
                  LOG(FATAL) << "Unexpected type.";
              }  // switch category.
              break;
            case message::MessageCategoryGroup::WORKER_COMMAND:
              command_receiver_->ReceiveCommand(result);
              break;
            default:
              LOG(FATAL) << "Invalid message header!";
          }  // switch category group.
        }
      }
    }
    if (!network::is_blocked()) {
      CleanUpWorkerRecord(worker_record);
    }
  }

  //! Sends data to a worker.
  // Sync call.
  void CallbackWriteEvent(WorkerRecord* worker_record) {
    while (true) {
      if (worker_record->send_buffer == nullptr) {
        if (!worker_record->sending_queue.empty()) {
          worker_record->send_buffer = worker_record->sending_queue.front();
          worker_record->sending_queue.pop_front();
        } else {
          return;
        }
      }
      const int status =
          evbuffer_write(worker_record->send_buffer, worker_record->socket_fd);
      if (status == -1) {
        if (network::is_blocked()) {
          CHECK_EQ(event_add(worker_record->write_event, nullptr), 0);
          return;
        } else {
          CleanUpWorkerRecord(worker_record);
          return;
        }
      } else {
        CHECK_EQ(evbuffer_drain(worker_record->send_buffer, status), 0);
        // A message is sent.
        if (evbuffer_get_length(worker_record->send_buffer) == 0) {
          evbuffer_free(worker_record->send_buffer);
          worker_record->send_buffer = nullptr;
        }
      }
    }
  }

 private:
  /*
   * Lifetime of a worker record.
   */

  //! Initializes a worker record.
  // Sync call.
  WorkerRecord* InitializeWorkerRecord(WorkerId worker_id, int socket_fd,
                                       const std::string& host,
                                       const std::string& service) {
    CHECK(worker_id_to_status_.find(worker_id) == worker_id_to_status_.end());
    auto worker_record = worker_id_to_status_[worker_id];
    worker_record->worker_id = worker_id;
    worker_record->host = host;
    worker_record->service = service;
    worker_record->socket_fd = socket_fd;
    worker_record->is_ready = false;
    worker_record->read_event = CHECK_NOTNULL(event_new(
        event_base_, socket_fd, EV_READ | EV_PERSIST, &DispatchWriteEvent,
        new WorkerRecordEventArg{this, worker_record}));
    event_add(worker_record->read_event, nullptr);
    worker_record->write_event = CHECK_NOTNULL(
        event_new(event_base_, socket_fd, EV_WRITE, &DispatchReadEvent,
                  new WorkerRecordEventArg{this, worker_record}));
    worker_record->send_buffer = nullptr;
    worker_record->receive_buffer = evbuffer_new();
    return worker_record;
  }

  //! Activates a worker record.
  // Sync call.
  void ProcessRegisterServicePortMessage(
      message::RegisterServicePort* message) {
    auto worker_record = worker_id_to_status_.at(message->from_worker_id);
    worker_record->route_service = message->route_service;
    worker_record->transmit_service = message->transmit_service;
    {
      message::UpdateAddedWorker update_message;
      update_message.added_worker_id = worker_record->worker_id;
      update_message.route_service = worker_record->route_service;
      update_message.transmit_service = worker_record->transmit_service;
      struct evbuffer* buffer =
          message::DataPlaneHeader::PackMessage(update_message);
      AppendAllReadySendingQueue(buffer);
    }

    {
      message::UpdatePartitionMapAndWorker update_message;
      update_message.version_id = internal_partition_map_version_;
      update_message.partition_map = &internal_partition_map_;
      for (auto& pair : worker_id_to_status_) {
        update_message.worker_ports[pair.first] = std::make_pair(
            pair.second->route_service, pair.second->transmit_service);
      }
      struct evbuffer* buffer =
          message::DataPlaneHeader::PackMessage(update_message);
      AppendWorkerSendingQueue(worker_record->worker_id, buffer);
    }

    worker_record->is_ready = true;

    // Notifies that a worker is up.
    command_receiver_->NotifyWorkerIsUp(message->from_worker_id);
  }

  //! Cleans up a worker record if it is disconnected..
  // Sync call.
  void ProcessNotifyWorkerDisconnect(message::NotifyWorkerDisconnect* message) {
    const WorkerId shutdown_worker_id = message->disconnected_worker_id;
    auto iter = worker_id_to_status_.find(shutdown_worker_id);
    if (iter != worker_id_to_status_.end()) {
      CleanUpWorkerRecord(iter->second);
    }
  }

  //! Cleans up a worker record.
  // Sync call.
  void CleanUpWorkerRecord(WorkerRecord* worker_record) {
    network::close_socket(worker_record->socket_fd);
    if (worker_record->read_event) {
      event_free(worker_record->read_event);
    }
    if (worker_record->write_event) {
      event_free(worker_record->write_event);
    }
    if (worker_record->send_buffer) {
      evbuffer_free(worker_record->send_buffer);
    }
    if (worker_record->receive_buffer) {
      evbuffer_free(worker_record->receive_buffer);
    }
    for (auto buffer : worker_record->sending_queue) {
      if (buffer) {
        evbuffer_free(buffer);
      }
    }
    auto worker_id = worker_record->worker_id;
    worker_id_to_status_.erase(worker_id);
    // Notifies that a worker is down.
    command_receiver_->NotifyWorkerIsDown(worker_id);
  }

 public:
  /*
   * Public async interfaces.
   */
  //! Sends a command to a worker.
  // Async call.
  void SendCommandToWorker(WorkerId worker_id,
                           struct evbuffer* buffer) override {
    message::DataPlaneHeader header;
    CHECK(header.ExtractHeader(buffer));
    CHECK(header.get_category_group() ==
          message::MessageCategoryGroup::WORKER_COMMAND);
    CHECK_EQ(header.get_length(), evbuffer_get_length(buffer) + header.kLength);
    event_main_thread_->AddInjectedEvent(std::bind(
        &::canary::ControllerCommunicationManager::AppendWorkerSendingQueue,
        this, worker_id, buffer));
  }

  //! Updates the partition map by adding an application.
  // Async call.
  void RequestUpdateApplicationPartitionMapAddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map) {
    event_main_thread_->AddInjectedEvent(
        std::bind(&::canary::ControllerCommunicationManager::
                      UpdateApplicationPartitionMapAddApplication,
                  this, application_id, per_application_partition_map));
  }

  //! Updates the partition map by dropping an application.
  // Async call.
  void RequestUpdateApplicationPartitionMapDropApplication(
      ApplicationId application_id) {
    event_main_thread_->AddInjectedEvent(
        std::bind(&::canary::ControllerCommunicationManager::
                      UpdateApplicationPartitionMapDropApplication,
                  this, application_id));
  }

  //! Updates the partition map incrementally.
  // Async call.
  void RequestUpdateApplicationPartitionMapIncremental(
      PartitionMapUpdate* partition_map_update) {
    event_main_thread_->AddInjectedEvent(
        std::bind(&::canary::ControllerCommunicationManager::
                      UpdateApplicationPartitionMapIncremental,
                  this, partition_map_update));
  }

  //! Shuts down a worker.
  // Async call.
  void RequestShutDownWorker(WorkerId worker_id) {
    event_main_thread_->AddInjectedEvent(
        std::bind(&::canary::ControllerCommunicationManager::ShutDownWorker,
                  this, worker_id));
  }

 private:
  /**
   * Internal helper functions.
   */

  //! Appends the worker sending queue.
  // Sync call.
  void AppendWorkerSendingQueue(WorkerId worker_id, struct evbuffer* buffer) {
    auto iter = worker_id_to_status_.find(worker_id);
    if (iter == worker_id_to_status_.end()) {
      evbuffer_free(buffer);
      return;
    } else {
      WorkerRecord* worker_record = iter->second;
      worker_record->sending_queue.push_back(buffer);
      event_add(worker_record->write_event, nullptr);
    }
  }

  //! Updates the partition map by adding an application.
  // Sync call.
  void UpdateApplicationPartitionMapAddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map) {
    *internal_partition_map_.AddPerApplicationPartitionMap(application_id) =
        std::move(*per_application_partition_map);
    ++internal_partition_map_version_;

    message::UpdatePartitionMapAddApplication message;
    message.version_id = internal_partition_map_version_;
    message.add_application_id = application_id;
    message.per_application_partition_map = per_application_partition_map;
    struct evbuffer* buffer = message::DataPlaneHeader::PackMessage(message);
    AppendAllReadySendingQueue(buffer);

    delete per_application_partition_map;
  }

  //! Updates the partition map by dropping an application.
  // Sync call.
  void UpdateApplicationPartitionMapDropApplication(
      ApplicationId application_id) {
    CHECK(internal_partition_map_.DeletePerApplicationPartitionMap(
        application_id));
    ++internal_partition_map_version_;

    message::UpdatePartitionMapDropApplication message;
    message.version_id = internal_partition_map_version_;
    message.drop_application_id = application_id;
    struct evbuffer* buffer = message::DataPlaneHeader::PackMessage(message);
    AppendAllReadySendingQueue(buffer);
  }

  //! Updates the partition map incrementally.
  // Sync call.
  void UpdateApplicationPartitionMapIncremental(
      PartitionMapUpdate* partition_map_update) {
    internal_partition_map_.MergeUpdate(*partition_map_update);

    ++internal_partition_map_version_;

    message::UpdatePartitionMapIncremental message;
    message.version_id = internal_partition_map_version_;
    message.partition_map_update = partition_map_update;
    struct evbuffer* buffer = message::DataPlaneHeader::PackMessage(message);
    AppendAllReadySendingQueue(buffer);

    delete partition_map_update;
  }

  //! Appends all ready sending queues.
  // Sync call.
  void AppendAllReadySendingQueue(struct evbuffer* buffer) {
    for (auto& pair : worker_id_to_status_) {
      WorkerRecord* worker_record = pair.second;
      if (worker_record->is_ready) {
        struct evbuffer* send_buffer = evbuffer_new();
        CHECK_EQ(evbuffer_add_buffer_reference(send_buffer, buffer), 0);
        worker_record->sending_queue.push_back(send_buffer);
        event_add(worker_record->write_event, nullptr);
      }
    }
    evbuffer_free(buffer);
  }

  //! Shuts down a worker.
  // Sync call.
  void ShutDownWorker(WorkerId worker_id) {
    message::ShutDownWorker message;
    struct evbuffer* buffer = message::DataPlaneHeader::PackMessage(message);
    AppendWorkerSendingQueue(worker_id, buffer);
  }

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  ControllerReceiveCommandInterface* command_receiver_ = nullptr;

  bool is_initialized_ = false;

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
