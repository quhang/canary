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
 * @file src/controller/controller_communication_manager.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerCommunicationManager.
 */

#include "controller/controller_communication_manager.h"

namespace canary {

void ControllerCommunicationManager::Initialize(
    network::EventMainThread* event_main_thread,
    ControllerReceiveCommandInterface* command_receiver) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  command_receiver_ = CHECK_NOTNULL(command_receiver);
  event_base_ = event_main_thread_->get_event_base();
  // Registers the listening port.
  listening_socket_ =
      network::allocate_and_bind_listen_socket(FLAGS_controller_service);
  CHECK_NE(listening_socket_, -1);
  // Starts the listening service.
  listening_event_ = CHECK_NOTNULL(
      evconnlistener_new(event_base_, &DispatchAcceptEvent, this,
                         LEV_OPT_CLOSE_ON_FREE, kBacklog, listening_socket_));
  evconnlistener_set_error_cb(listening_event_, DispatchAcceptErrorEvent);
  // Sets the initialization flag.
  is_initialized_ = true;
  LOG(INFO) << "Controller communication manager is initialized.";
}

/*
 * Public static methods used to dispatch event callbacks.
 */

void ControllerCommunicationManager::DispatchAcceptEvent(
    struct evconnlistener* listener, int socket_fd, struct sockaddr* address,
    int socklen, void* arg) {
  auto manager = reinterpret_cast<SelfType*>(arg);
  manager->CallbackAcceptEvent(listener, socket_fd, address, socklen);
}

void ControllerCommunicationManager::DispatchAcceptErrorEvent(
    struct evconnlistener*, void*) {
  LOG(FATAL) << "Failure on the listener: "
             << network::get_error_message(network::get_last_error_number());
}

void ControllerCommunicationManager::DispatchReadEvent(int socket_fd,
                                                       short,  // NOLINT
                                                       void* arg) {
  auto worker_record = reinterpret_cast<WorkerRecord*>(arg);
  CHECK_EQ(socket_fd, worker_record->socket_fd);
  worker_record->manager->CallbackReadEvent(worker_record);
}

void ControllerCommunicationManager::DispatchWriteEvent(int socket_fd,
                                                        short,  // NOLINT
                                                        void* arg) {
  auto worker_record = reinterpret_cast<WorkerRecord*>(arg);
  CHECK_EQ(socket_fd, worker_record->socket_fd);
  worker_record->manager->CallbackWriteEvent(worker_record);
}

/*
 * Core logic to handle accept/read/write events.
 */

void ControllerCommunicationManager::CallbackAcceptEvent(
    struct evconnlistener* listener, int socket_fd, struct sockaddr* address,
    int socklen) {
  CHECK_EQ(listener, listening_event_);
  std::string host, service;
  network::translate_sockaddr_to_string(address, socklen, &host, &service);
  VLOG(1) << host << ":" << service << " reaches the controller port.";
  const WorkerId worker_id = worker_id_allocator_++;
  VLOG(1) << "Allocated WorkerId=" << get_value(worker_id);
  InitializeWorkerRecord(worker_id, socket_fd, host, service);
  // Handshake protocol:
  // controller -> worker: AssignWorkerId.
  // worker -> controller: RegisterServicePortMessage.
  message::AssignWorkerId message;
  message.assigned_worker_id = worker_id;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(message);
  AppendWorkerSendingQueue(worker_id, buffer);
}

void ControllerCommunicationManager::CallbackReadEvent(
    WorkerRecord* worker_record) {
  struct evbuffer* receive_buffer = worker_record->receive_buffer;
  const int socket_fd = worker_record->socket_fd;

  message::ControlHeader message_header;
  int status = 0;
  while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
    while (struct evbuffer* whole_message =
               message_header.SegmentMessage(receive_buffer)) {
      ProcessIncomingMessage(message_header, whole_message);
    }
  }
  if (status == 0 || (status == 1 && !network::is_blocked())) {
    CleanUpWorkerRecord(worker_record);
  }
}

void ControllerCommunicationManager::CallbackWriteEvent(
    WorkerRecord* worker_record) {
  worker_record->send_buffer = network::send_as_much(
      worker_record->socket_fd, worker_record->send_buffer,
      &worker_record->send_queue);
  if (worker_record->send_buffer != nullptr) {
    // Channel is blocked or has error.
    if (network::is_blocked()) {
      CHECK_EQ(event_add(worker_record->write_event, nullptr), 0);
    } else {
      CleanUpWorkerRecord(worker_record);
    }
  }
}

/*
 * Lifetime of a worker record.
 */

void ControllerCommunicationManager::InitializeWorkerRecord(
    WorkerId worker_id, int socket_fd, const std::string& host,
    const std::string& service) {
  CHECK(worker_id_to_status_.find(worker_id) == worker_id_to_status_.end());
  auto& worker_record = worker_id_to_status_[worker_id];
  worker_record.worker_id = worker_id;
  worker_record.host = host;
  worker_record.service = service;
  worker_record.socket_fd = socket_fd;
  worker_record.is_ready = false;
  worker_record.read_event =
      CHECK_NOTNULL(event_new(event_base_, socket_fd, EV_READ | EV_PERSIST,
                              &DispatchReadEvent, &worker_record));
  CHECK_EQ(event_add(worker_record.read_event, nullptr), 0);
  worker_record.write_event = CHECK_NOTNULL(event_new(
      event_base_, socket_fd, EV_WRITE, &DispatchWriteEvent, &worker_record));
  worker_record.send_buffer = nullptr;
  worker_record.receive_buffer = evbuffer_new();
  worker_record.manager = this;
}

void ControllerCommunicationManager::ProcessRegisterServicePortMessage(
    message::RegisterServicePort* message) {
  auto worker_record = &worker_id_to_status_.at(message->from_worker_id);
  worker_record->route_service = message->route_service;
  worker_record->transmit_service = message->transmit_service;

  // Tells other workers of the added worker.
  {
    message::UpdateAddedWorker update_message;
    update_message.added_worker_id = worker_record->worker_id;
    update_message.route_service = worker_record->route_service;
    update_message.transmit_service = worker_record->transmit_service;
    struct evbuffer* buffer =
        message::ControlHeader::PackMessage(update_message);
    AppendAllReadySendingQueue(buffer);
  }

  // Tells the worker of existing workers and the partition map.
  {
    message::UpdatePartitionMapAndWorker update_message;
    update_message.version_id = internal_partition_map_version_;
    update_message.partition_map = &internal_partition_map_;
    for (auto& pair : worker_id_to_status_) {
      update_message.worker_ports[pair.first] = std::make_pair(
          pair.second.route_service, pair.second.transmit_service);
    }
    struct evbuffer* buffer =
        message::ControlHeader::PackMessage(update_message);
    AppendWorkerSendingQueue(worker_record->worker_id, buffer);
  }

  // The worker is now ready.
  worker_record->is_ready = true;

  // Notifies that a worker is up.
  command_receiver_->NotifyWorkerIsUp(worker_record->worker_id);

  delete message;
}

void ControllerCommunicationManager::ProcessNotifyWorkerDisconnect(
    message::NotifyWorkerDisconnect* message) {
  const WorkerId shutdown_worker_id = message->disconnected_worker_id;
  auto iter = worker_id_to_status_.find(shutdown_worker_id);
  if (iter != worker_id_to_status_.end()) {
    CleanUpWorkerRecord(&iter->second);
  }
  delete message;
}

void ControllerCommunicationManager::CleanUpWorkerRecord(
    WorkerRecord* worker_record) {
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
  for (auto buffer : worker_record->send_queue) {
    if (buffer) {
      evbuffer_free(buffer);
    }
  }
  const auto worker_id = worker_record->worker_id;
  worker_id_to_status_.erase(worker_id);

  // Notifies that a worker is down.
  command_receiver_->NotifyWorkerIsDown(worker_id);
}

/*
 * Public async interfaces.
 */

void ControllerCommunicationManager::SendCommandToWorker(
    WorkerId worker_id, struct evbuffer* buffer) {
  message::ControlHeader header;
  CHECK(header.ExtractHeader(buffer));
  CHECK(header.get_category_group() ==
        message::MessageCategoryGroup::WORKER_COMMAND);
  CHECK_EQ(header.kLength + header.get_length(), evbuffer_get_length(buffer));
  event_main_thread_->AddInjectedEvent(std::bind(
      &SelfType::AppendWorkerSendingQueueIfReady, this, worker_id, buffer));
}

void ControllerCommunicationManager::AddApplication(
    ApplicationId application_id,
    PerApplicationPartitionMap* per_application_partition_map) {
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::InternalAddApplication, this, application_id,
                per_application_partition_map));
}

void ControllerCommunicationManager::DropApplication(
    ApplicationId application_id) {
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::InternalDropApplication, this, application_id));
}

void ControllerCommunicationManager::UpdatePartitionMap(
    PartitionMapUpdate* partition_map_update) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &SelfType::InternalUpdatePartitionMap, this, partition_map_update));
}

void ControllerCommunicationManager::ShutDownWorker(WorkerId worker_id) {
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::InternalShutDownWorker, this, worker_id));
}

/**
 * Implements async interfaces as synchronous callback.
 */

void ControllerCommunicationManager::InternalAddApplication(
    ApplicationId application_id,
    PerApplicationPartitionMap* per_application_partition_map) {
  *internal_partition_map_.AddPerApplicationPartitionMap(application_id) =
      std::move(*per_application_partition_map);
  ++internal_partition_map_version_;

  message::UpdatePartitionMapAddApplication message;
  message.version_id = internal_partition_map_version_;
  message.add_application_id = application_id;
  message.per_application_partition_map = per_application_partition_map;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(message);
  AppendAllReadySendingQueue(buffer);

  delete per_application_partition_map;
}

void ControllerCommunicationManager::InternalDropApplication(
    ApplicationId application_id) {
  CHECK(
      internal_partition_map_.DeletePerApplicationPartitionMap(application_id));
  ++internal_partition_map_version_;

  message::UpdatePartitionMapDropApplication message;
  message.version_id = internal_partition_map_version_;
  message.drop_application_id = application_id;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(message);
  AppendAllReadySendingQueue(buffer);
}

void ControllerCommunicationManager::InternalUpdatePartitionMap(
    PartitionMapUpdate* partition_map_update) {
  internal_partition_map_.MergeUpdate(*partition_map_update);

  ++internal_partition_map_version_;

  message::UpdatePartitionMapIncremental message;
  message.version_id = internal_partition_map_version_;
  message.partition_map_update = partition_map_update;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(message);
  AppendAllReadySendingQueue(buffer);

  delete partition_map_update;
}

void ControllerCommunicationManager::InternalShutDownWorker(
    WorkerId worker_id) {
  message::ShutDownWorker message;
  struct evbuffer* buffer = message::ControlHeader::PackMessage(message);
  AppendWorkerSendingQueueIfReady(worker_id, buffer);
}

/**
 * Internal helper functions.
 */

void ControllerCommunicationManager::ProcessIncomingMessage(
    const message::ControlHeader& message_header, struct evbuffer* buffer) {
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  using message::ControlHeader;
  switch (message_header.get_category_group()) {
    case MessageCategoryGroup::DATA_PLANE_CONTROL:
      switch (message_header.get_category()) {
        case MessageCategory::REGISTER_SERVICE_PORT: {
          auto message = ControlHeader::UnpackMessage<
              MessageCategory::REGISTER_SERVICE_PORT>(buffer);
          // Ownership transferred.
          ProcessRegisterServicePortMessage(message);
          break;
        }
        case message::MessageCategory::NOTIFY_WORKER_DISCONNECT: {
          auto message = ControlHeader::UnpackMessage<
              MessageCategory::NOTIFY_WORKER_DISCONNECT>(buffer);
          // Ownership transferred.
          ProcessNotifyWorkerDisconnect(message);
          break;
        }
        default:
          LOG(FATAL) << "Unexpected message type.";
      }  // switch category.
      break;
    case MessageCategoryGroup::CONTROLLER_COMMAND:
      command_receiver_->ReceiveCommand(buffer);
      break;
    default:
      LOG(FATAL) << "Invalid message header!";
  }  // switch category group.
}

void ControllerCommunicationManager::AppendWorkerSendingQueue(
    WorkerId worker_id, struct evbuffer* buffer) {
  AppendWorkerSendingQueueWithFlag(worker_id, buffer, false);
}

void ControllerCommunicationManager::AppendWorkerSendingQueueIfReady(
    WorkerId worker_id, struct evbuffer* buffer) {
  AppendWorkerSendingQueueWithFlag(worker_id, buffer, true);
}

void ControllerCommunicationManager::AppendWorkerSendingQueueWithFlag(
    WorkerId worker_id, struct evbuffer* buffer, bool enforce) {
  auto iter = worker_id_to_status_.find(worker_id);
  if (iter == worker_id_to_status_.end()) {
    evbuffer_free(buffer);
    return;
  } else {
    WorkerRecord* worker_record = &iter->second;
    CHECK(worker_record->is_ready || !enforce);
    worker_record->send_queue.push_back(buffer);
    event_add(worker_record->write_event, nullptr);
  }
}

void ControllerCommunicationManager::AppendAllReadySendingQueue(
    struct evbuffer* buffer) {
  for (auto& pair : worker_id_to_status_) {
    WorkerRecord* worker_record = &pair.second;
    if (worker_record->is_ready) {
      struct evbuffer* send_buffer = evbuffer_new();
      // Copy by reference.
      CHECK_EQ(evbuffer_add_buffer_reference(send_buffer, buffer), 0);
      worker_record->send_queue.push_back(send_buffer);
      event_add(worker_record->write_event, nullptr);
    }
  }
  evbuffer_free(buffer);
}

}  // namespace canary
