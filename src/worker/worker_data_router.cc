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
 * @file src/worker/worker_data_router.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerDataRouter.
 */

#include "worker/worker_data_router.h"

#include <algorithm>
#include <vector>

namespace canary {

void WorkerDataRouter::Initialize(WorkerId worker_id,
                                  network::EventMainThread* event_main_thread,
                                  WorkerReceiveDataInterface* data_receiver,
                                  const std::string& route_service) {
  self_worker_id_ = worker_id;
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  data_receiver_ = CHECK_NOTNULL(data_receiver);
  event_base_ = CHECK_NOTNULL(event_main_thread_->get_event_base());
  route_service_ = route_service;
  // Registers the listening port.
  listening_socket_ = network::allocate_and_bind_listen_socket(route_service_);
  CHECK_NE(listening_socket_, -1);
  // Starts the listening service.
  listening_event_ =
      evconnlistener_new(event_base_, &DispatchAcceptEvent, this,
                         LEV_OPT_CLOSE_ON_FREE, kBacklog, listening_socket_);
  CHECK_NOTNULL(listening_event_);
  evconnlistener_set_error_cb(listening_event_, DispatchAcceptErrorEvent);
  // Sets the initialization flag.
  is_initialized_ = true;
  VLOG(1) << "Worker data router is initialized. (id="
          << get_value(self_worker_id_) << ")";
}

void WorkerDataRouter::Finalize() {
  evconnlistener_free(listening_event_);
  for (auto& pair : worker_id_to_status_) {
    auto peer_record = &pair.second;
    if (peer_record->read_event) {
      event_free(peer_record->read_event);
      peer_record->read_event = nullptr;
    }
    if (peer_record->write_event) {
      event_free(peer_record->write_event);
      peer_record->write_event = nullptr;
    }
    // Caution: socket closing happens at last.
    if (peer_record->socket_fd >= 0) {
      network::close_socket(peer_record->socket_fd);
      peer_record->socket_fd = -1;
    }
  }
  is_shutdown_ = true;
}

/*
 * Public static methods used for libevent to dispatch events.
 */

void WorkerDataRouter::DispatchAcceptEvent(struct evconnlistener* listener,
                                           int socket_fd,
                                           struct sockaddr* address,
                                           int socklen, void* arg) {
  auto router = reinterpret_cast<SelfType*>(arg);
  router->CallbackAcceptEvent(listener, socket_fd, address, socklen);
}

void WorkerDataRouter::DispatchAcceptErrorEvent(struct evconnlistener*, void*) {
  LOG(WARNING) << "Accepting connection failed ("
               << network::get_error_message(network::get_last_error_number())
               << ")";
}

void WorkerDataRouter::DispatchConnectEvent(int socket_fd, short,  // NOLINT
                                            void* arg) {
  auto peer_record = reinterpret_cast<PeerRecord*>(arg);
  CHECK_EQ(socket_fd, peer_record->socket_fd);
  peer_record->router->CallbackConnectEvent(peer_record);
}

void WorkerDataRouter::DispatchPassiveConnectEvent(int socket_fd,
                                                   short,  // NOLINT
                                                   void* arg) {
  auto peer_record = reinterpret_cast<PeerRecord*>(arg);
  CHECK_EQ(socket_fd, peer_record->socket_fd);
  peer_record->router->CallbackPassiveConnectEvent(peer_record);
}

void WorkerDataRouter::DispatchReadEvent(int socket_fd, short,  // NOLINT
                                         void* arg) {
  auto peer_record = reinterpret_cast<PeerRecord*>(arg);
  CHECK_EQ(socket_fd, peer_record->socket_fd);
  peer_record->router->CallbackReadEvent(peer_record);
}

void WorkerDataRouter::DispatchWriteEvent(int socket_fd, short,  // NOLINT
                                          void* arg) {
  auto peer_record = reinterpret_cast<PeerRecord*>(arg);
  CHECK_EQ(socket_fd, peer_record->socket_fd);
  peer_record->router->CallbackWriteEvent(peer_record);
}

/*
 * Core logic to handle connect/read/write events.
 */

void WorkerDataRouter::CallbackInitiateEvent(PeerRecord* peer_record) {
  CHECK(!peer_record->is_ready);
  CHECK(!peer_record->is_passive);
  peer_record->socket_fd = network::allocate_and_connect_socket(
      peer_record->host, peer_record->route_service);
  CHECK_GE(peer_record->socket_fd, 0);
  // Triggered when the channel is ready to write.
  CHECK_EQ(event_base_once(event_base_, peer_record->socket_fd, EV_WRITE,
                           DispatchConnectEvent, peer_record, nullptr),
           0);
}

void WorkerDataRouter::CallbackConnectEvent(PeerRecord* peer_record) {
  CHECK(!peer_record->is_ready);
  CHECK(!peer_record->is_passive);
  const int error_flag =
      network::get_socket_error_number(peer_record->socket_fd);
  if (error_flag != 0) {
    // The socket is in error state, so has to reopen it.
    network::close_socket(peer_record->socket_fd);
    peer_record->socket_fd = -1;
    event_main_thread_->AddDelayInjectedEvent(
        std::bind(&SelfType::CallbackInitiateEvent, this, peer_record));
  } else {
    ActivateActivePeerRecord(peer_record);
  }
}

void WorkerDataRouter::CallbackAcceptEvent(struct evconnlistener* listener,
                                           int socket_fd,
                                           struct sockaddr* address,
                                           int socklen) {
  CHECK_EQ(listener, listening_event_);
  std::string host, service;
  network::translate_sockaddr_to_string(address, socklen, &host, &service);
  // Handshake protocol:
  // slave -> master: worker_id.
  auto peer_record = InitializePassivePeerRecord(socket_fd, host, service);
  CHECK_EQ(event_base_once(event_base_, socket_fd, EV_READ,
                           DispatchPassiveConnectEvent, peer_record, nullptr),
           0);
}

void WorkerDataRouter::CallbackPassiveConnectEvent(PeerRecord* peer_record) {
  CHECK(!peer_record->is_ready);
  CHECK(peer_record->is_passive);
  struct evbuffer* receive_buffer = peer_record->receive_buffer;
  const int socket_fd = peer_record->socket_fd;

  // Expects handshake message.
  int status = 0;
  while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
    if (struct evbuffer* whole_message =
            message::SegmentDataMessage(receive_buffer)) {
      auto header = CHECK_NOTNULL(message::ExamineDataHeader(whole_message));
      CHECK(header->category ==
            message::MessageCategory::DIRECT_DATA_HANDSHAKE);
      message::DirectDataHandshake message;
      message::RemoveDataHeader(whole_message);
      message::DeserializeMessage(whole_message, &message);
      ActivatePassivePeerRecord(message.from_worker_id, peer_record);
      return;
    }
  }
  if (status == 0 || (status == -1 && !network::is_blocked())) {
    evbuffer_free(peer_record->receive_buffer);
    network::close_socket(socket_fd);
    delete peer_record;
    LOG(WARNING) << "Handshake fails between workers!";
  } else {
    // Waits for more data.
    CHECK_EQ(event_base_once(event_base_, socket_fd, EV_READ,
                             DispatchPassiveConnectEvent, peer_record, nullptr),
             0);
  }
}

void WorkerDataRouter::CallbackReadEvent(PeerRecord* peer_record) {
  struct evbuffer* receive_buffer = peer_record->receive_buffer;
  const int socket_fd = peer_record->socket_fd;
  int status = 0;
  while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
    while (struct evbuffer* whole_message =
               message::SegmentDataMessage(receive_buffer)) {
      ProcessIncomingMessage(whole_message);
    }
  }
  if (status == 0 || (status == -1 && !network::is_blocked())) {
    CleanUpPeerRecord(peer_record);
  }
}

void WorkerDataRouter::CallbackWriteEvent(PeerRecord* peer_record) {
  peer_record->send_buffer =
      network::send_as_much(peer_record->socket_fd, peer_record->send_buffer,
                            &peer_record->send_queue);
  // Keeps sending if the sending succeeds.
  if (peer_record->send_buffer == nullptr) {
    // Send low priority queues later.
    peer_record->send_buffer =
        network::send_as_much(peer_record->socket_fd, peer_record->send_buffer,
                              &peer_record->low_priority_send_queue);
  }
  // Checks for error. Notice this branch cannot merge with the above.
  if (peer_record->send_buffer != nullptr) {
    // Channel is blocked or has error.
    if (network::is_blocked()) {
      CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
    } else {
      CleanUpPeerRecord(peer_record);
    }
  }
}

/*
 * Lifetime of a peer record.
 */

WorkerDataRouter::PeerRecord* WorkerDataRouter::InitializeActivePeerRecord(
    WorkerId worker_id, const std::string& worker_host,
    const std::string& worker_service) {
  CHECK(worker_id != self_worker_id_);
  VLOG(3) << get_value(self_worker_id_) << " actively connects to "
          << get_value(worker_id);
  CHECK(worker_id_to_status_.find(worker_id) == worker_id_to_status_.end());
  auto& peer_record = worker_id_to_status_[worker_id];
  peer_record.worker_id = worker_id;
  peer_record.host = worker_host;
  peer_record.route_service = worker_service;
  peer_record.router = this;
  peer_record.is_ready = false;
  peer_record.is_passive = false;
  return &peer_record;
}

//! Activates the peer record.
void WorkerDataRouter::ActivateActivePeerRecord(PeerRecord* peer_record) {
  const auto socket_fd = peer_record->socket_fd;
  CHECK_NE(socket_fd, -1);
  peer_record->is_ready = true;
  CHECK(!peer_record->is_passive);
  peer_record->read_event =
      CHECK_NOTNULL(event_new(event_base_, socket_fd, EV_READ | EV_PERSIST,
                              &DispatchReadEvent, peer_record));
  CHECK_EQ(event_add(peer_record->read_event, nullptr), 0);
  peer_record->write_event = CHECK_NOTNULL(event_new(
      event_base_, socket_fd, EV_WRITE, &DispatchWriteEvent, peer_record));
  CHECK(peer_record->send_buffer == nullptr);
  peer_record->receive_buffer = evbuffer_new();
  // Writes handshake message.
  {
    message::DirectDataHandshake message;
    message.from_worker_id = self_worker_id_;
    struct evbuffer* buffer = message::SerializeMessage(message);
    const auto length = evbuffer_get_length(buffer);
    auto header = message::AddDataHeader(buffer);
    header->length = length;
    header->FillInMessageType(message);
    peer_record->send_queue.push_back(buffer);
    CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
  }
  // Refreshes the pending messages to be routed.
  TriggerRefresh();
}

//! Initializes a passive peer record, after a connection comes in.
WorkerDataRouter::PeerRecord* WorkerDataRouter::InitializePassivePeerRecord(
    int socket_fd, const std::string& host, const std::string& service) {
  auto peer_record = new PeerRecord();
  CHECK(peer_record->worker_id == WorkerId::INVALID);
  peer_record->host = host;
  peer_record->route_service = service;
  peer_record->socket_fd = socket_fd;
  peer_record->is_ready = false;
  peer_record->is_passive = true;
  peer_record->receive_buffer = evbuffer_new();
  peer_record->router = this;
  return peer_record;
}

//! Activates a peer record, after receiving its worker id.
void WorkerDataRouter::ActivatePassivePeerRecord(WorkerId from_worker_id,
                                                 PeerRecord* old_peer_record) {
  CHECK(from_worker_id != self_worker_id_);
  CHECK(worker_id_to_status_.find(from_worker_id) ==
        worker_id_to_status_.end());
  VLOG(3) << get_value(self_worker_id_) << " is passively connected by "
          << get_value(from_worker_id);
  auto& peer_record = worker_id_to_status_[from_worker_id];
  peer_record = *old_peer_record;
  delete old_peer_record;
  peer_record.worker_id = from_worker_id;
  CHECK(peer_record.is_passive);
  peer_record.is_ready = true;
  const auto socket_fd = peer_record.socket_fd;
  peer_record.read_event =
      CHECK_NOTNULL(event_new(event_base_, socket_fd, EV_READ | EV_PERSIST,
                              &DispatchReadEvent, &peer_record));
  CHECK_EQ(event_add(peer_record.read_event, nullptr), 0);
  peer_record.write_event = CHECK_NOTNULL(event_new(
      event_base_, socket_fd, EV_WRITE, &DispatchWriteEvent, &peer_record));
  CHECK(peer_record.send_buffer == nullptr);
  CHECK_NOTNULL(peer_record.receive_buffer);
  CHECK_EQ(peer_record.router, this);
  // Refreshes the pending messages to be routed.
  TriggerRefresh();
}

//! Cleans up a peer record.
void WorkerDataRouter::CleanUpPeerRecord(PeerRecord* peer_record) {
  if (peer_record->read_event) {
    event_free(peer_record->read_event);
    peer_record->read_event = nullptr;
  }
  if (peer_record->write_event) {
    event_free(peer_record->write_event);
    peer_record->write_event = nullptr;
  }
  // Caution: socket closing must happen after event clearing, otherwise,
  // libevent might try to poll on invalid socket fd.
  if (peer_record->socket_fd >= 0) {
    network::close_socket(peer_record->socket_fd);
    peer_record->socket_fd = -1;
  }
  if (peer_record->send_buffer) {
    evbuffer_free(peer_record->send_buffer);
    peer_record->send_buffer = nullptr;
  }
  if (peer_record->receive_buffer) {
    evbuffer_free(peer_record->receive_buffer);
    peer_record->receive_buffer = nullptr;
  }
  for (auto buffer : peer_record->send_queue) {
    // Put sending messages into the pending queue.
    // TODO(quhang): resend un-acknowledged message as well.
    route_pending_queue_.push_back(buffer);
  }
  peer_record->send_queue.clear();
  const auto worker_id = peer_record->worker_id;
  worker_id_to_status_.erase(worker_id);
  VLOG(3) << get_value(self_worker_id_) << " loses connection with "
          << get_value(worker_id);
  // Triggers resending.
  TriggerRefresh();
}

/*
 * Public sending data interface. Must be called after initialization.
 */

void WorkerDataRouter::SendDataToPartition(ApplicationId application_id,
                                           VariableGroupId variable_group_id,
                                           PartitionId partition_id,
                                           StageId stage_id,
                                           struct evbuffer* buffer) {
  AddUnicastHeader(application_id, variable_group_id, partition_id, stage_id,
                   buffer);
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::SendUnicastData, this, buffer));
}

void WorkerDataRouter::SendDataToWorker(WorkerId worker_id,
                                        struct evbuffer* buffer) {
  CHECK(message::CheckIsIntegrateDataMessage(buffer));
  CHECK(message::ExamineDataHeader(buffer)->category_group ==
        message::MessageCategoryGroup::APPLICATION_DATA_DIRECT);
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::SendDirectData, this, worker_id, buffer));
}

void WorkerDataRouter::BroadcastDataToPartition(
    ApplicationId application_id, VariableGroupId variable_group_id,
    StageId stage_id, struct evbuffer* buffer) {
  event_main_thread_->AddInjectedEvent(
      std::bind(&SelfType::AddHeaderAndSendMulticastData, this, application_id,
                variable_group_id, stage_id, buffer));
}

/*
 * Synchronous call interfaces invoked by WorkerCommunicationManager.
 */

void WorkerDataRouter::UpdatePartitionMapAndWorker(
    message::UpdatePartitionMapAndWorker* message) {
  VLOG(3) << "UpdatePartitionMapAndWorker";
  CHECK(is_initialized_);
  internal_partition_map_version_ = message->version_id;
  internal_partition_map_ = std::move(*message->partition_map);
  delete message->partition_map;
  for (auto& pair : message->worker_addresses) {
    const WorkerId new_worker_id = pair.first;
    if (new_worker_id < self_worker_id_) {
      auto peer_record = InitializeActivePeerRecord(
          new_worker_id, pair.second.host, pair.second.service);
      CallbackInitiateEvent(peer_record);
    }
  }
  TriggerRefresh();
  delete message;
}

void WorkerDataRouter::UpdatePartitionMapAddApplication(
    message::UpdatePartitionMapAddApplication* message) {
  VLOG(3) << "UpdatePartitionMapAddApplication";
  CHECK(is_initialized_);
  internal_partition_map_version_ = message->version_id;
  *internal_partition_map_.AddPerApplicationPartitionMap(
      message->add_application_id) =
      std::move(*message->per_application_partition_map);
  delete message->per_application_partition_map;
  TriggerRefresh();
  delete message;
}

void WorkerDataRouter::UpdatePartitionMapDropApplication(
    message::UpdatePartitionMapDropApplication* message) {
  VLOG(3) << "UpdatePartitionMapDropApplication";
  CHECK(is_initialized_);
  internal_partition_map_version_ = message->version_id;
  CHECK(internal_partition_map_.DeletePerApplicationPartitionMap(
      message->drop_application_id));
  delete message;
}

void WorkerDataRouter::UpdatePartitionMapIncremental(
    message::UpdatePartitionMapIncremental* message) {
  VLOG(3) << "UpdatePartitionMapIncremental";
  CHECK(is_initialized_);
  internal_partition_map_version_ = message->version_id;
  internal_partition_map_.MergeUpdate(*message->partition_map_update);
  delete message->partition_map_update;
  TriggerRefresh();
  delete message;
}

void WorkerDataRouter::UpdateAddedWorker(message::UpdateAddedWorker* message) {
  VLOG(3) << "UpdateAddedWorker";
  CHECK(is_initialized_);
  if (message->added_worker_id < self_worker_id_) {
    auto peer_record = InitializeActivePeerRecord(
        message->added_worker_id, message->network_address.host,
        message->network_address.service);
    CallbackInitiateEvent(peer_record);
  }
  delete message;
}

void WorkerDataRouter::ShutDownWorker(message::ShutDownWorker* message) {
  CHECK(is_initialized_);
  delete message;
  Finalize();
}

/*
 * Routing facilities.
 */

// The buffer includes the header.
void WorkerDataRouter::SendUnicastData(struct evbuffer* buffer) {
  ProcessUnicastMessage(buffer);
}

void WorkerDataRouter::AddHeaderAndSendMulticastData(
    ApplicationId application_id, VariableGroupId variable_group_id,
    StageId stage_id, struct evbuffer* buffer) {
  std::map<WorkerId, PartitionIdVector> receiver_map;
  FillInMulticastReceiver(application_id, variable_group_id, &receiver_map);
  for (auto& pair : receiver_map) {
    const auto send_worker_id = pair.first;
    auto peer_record = GetPeerRecordIfReady(send_worker_id);
    if (peer_record != nullptr) {
      // Normal routine of multicast.
      struct evbuffer* send_buffer = evbuffer_new();
      {
        // Appends the receivers.
        CanaryOutputArchive archive(send_buffer);
        // PartitionIdVector.
        archive(pair.second);
      }
      CHECK_EQ(evbuffer_add_buffer_reference(send_buffer, buffer), 0);
      AddMulticastHeader(application_id, variable_group_id, stage_id,
                         internal_partition_map_version_, send_buffer);
      peer_record->send_queue.push_back(send_buffer);
      CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
    } else {
      // Multicast fails, and degrades to unicast.
      for (auto to_partition_id : pair.second) {
        struct evbuffer* send_buffer = evbuffer_new();
        CHECK_EQ(evbuffer_add_buffer_reference(send_buffer, buffer), 0);
        AddUnicastHeader(application_id, variable_group_id, to_partition_id,
                         stage_id, send_buffer);
        ProcessUnicastMessage(send_buffer);
      }
    }
  }
  evbuffer_free(buffer);
}

void WorkerDataRouter::FillInMulticastReceiver(
    ApplicationId application_id, VariableGroupId variable_group_id,
    std::map<WorkerId, PartitionIdVector>* receiver_map) {
  auto per_app_partition_map =
      internal_partition_map_.GetPerApplicationPartitionMap(application_id);
  CHECK(per_app_partition_map != nullptr)
      << "The partition map must be complete before doing multicast, "
      << "i.e. the number of receivers should be well defined.";
  const int num_partition =
      per_app_partition_map->QueryPartitioning(variable_group_id);
  for (int id = 0; id < num_partition; ++id) {
    const auto to_partition_id = static_cast<PartitionId>(id);
    const auto to_worker_id = per_app_partition_map->QueryWorkerId(
        variable_group_id, to_partition_id);
    CHECK(to_worker_id != WorkerId::INVALID);
    (*receiver_map)[to_worker_id].push_back(to_partition_id);
  }
}

void WorkerDataRouter::SendDirectData(WorkerId worker_id,
                                      struct evbuffer* buffer) {
  if (worker_id == self_worker_id_) {
    ProcessDirectMessage(buffer);
  } else {
    AppendLowPrioritySendingQueue(worker_id, buffer);
  }
}

void WorkerDataRouter::AppendLowPrioritySendingQueue(WorkerId worker_id,
                                                     struct evbuffer* buffer) {
  auto peer_record = GetPeerRecordIfReady(worker_id);
  if (peer_record != nullptr) {
    peer_record->low_priority_send_queue.push_back(buffer);
    CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
  } else {
    direct_pending_queue_[worker_id].push_back(buffer);
  }
}

/*
 * Receiving messages.
 */

void WorkerDataRouter::ProcessIncomingMessage(struct evbuffer* buffer) {
  auto header = CHECK_NOTNULL(message::ExamineDataHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  switch (header->category_group) {
    case MessageCategoryGroup::APPLICATION_DATA_ROUTE:
      switch (header->category) {
        case MessageCategory::ROUTE_DATA_UNICAST:
          ProcessUnicastMessage(buffer);
          break;
        case MessageCategory::ROUTE_DATA_MULTICAST:
          ProcessMulticastMessage(buffer);
          break;
        default:
          LOG(FATAL) << "Unknown message category!";
      }
      break;
    case MessageCategoryGroup::APPLICATION_DATA_DIRECT:
      ProcessDirectMessage(buffer);
      break;
    default:
      LOG(FATAL) << "Unknown message category group!";
  }
}

void WorkerDataRouter::ProcessUnicastMessage(struct evbuffer* buffer) {
  auto header = CHECK_NOTNULL(message::ExamineDataHeader(buffer));
  const auto application_id = header->to_application_id;
  const auto variable_group_id = header->to_variable_group_id;
  const auto partition_id = header->to_partition_id;
  const auto stage_id = header->to_stage_id;
  const auto dest_worker_id = internal_partition_map_.QueryWorkerId(
      application_id, variable_group_id, partition_id);
  if (header->partition_map_version >= internal_partition_map_version_ ||
      dest_worker_id == self_worker_id_) {
    // Delivers locally.
    message::RemoveDataHeader(buffer);
    const bool accepted = data_receiver_->ReceiveRoutedData(
        application_id, variable_group_id, partition_id, stage_id, buffer);
    if (!accepted) {
      // Rejected messages are put into pending queue.
      AddUnicastHeader(application_id, variable_group_id, partition_id,
                       stage_id, buffer);
      route_pending_queue_.push_back(buffer);
    }
  } else {
    header->partition_map_version = internal_partition_map_version_;
    auto peer_record = GetPeerRecordIfReady(dest_worker_id);
    if (peer_record != nullptr) {
      // Normal routing of unicast message.
      peer_record->send_queue.push_back(buffer);
      CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
    } else {
      route_pending_queue_.push_back(buffer);
    }
  }
}

void WorkerDataRouter::ProcessMulticastMessage(struct evbuffer* buffer) {
  auto header = message::StripDataHeader(buffer);
  PartitionIdVector partition_id_vector;
  {
    CanaryInputArchive archive(buffer);
    archive(partition_id_vector);
  }
  const auto application_id = header->to_application_id;
  const auto variable_group_id = header->to_variable_group_id;
  const auto stage_id = header->to_stage_id;
  for (auto partition_id : partition_id_vector) {
    // Makes a copy of the buffer.
    struct evbuffer* deliver_buffer = evbuffer_new();
    CHECK_EQ(evbuffer_add_buffer_reference(deliver_buffer, buffer), 0);
    const auto dest_worker_id = internal_partition_map_.QueryWorkerId(
        application_id, variable_group_id, partition_id);
    if (header->partition_map_version >= internal_partition_map_version_ ||
        dest_worker_id == self_worker_id_) {
      // Delivers locally.
      // Header is already removed. No need: message::RemoveDataHeader(buffer);
      const bool accepted = data_receiver_->ReceiveRoutedData(
          application_id, variable_group_id, partition_id, stage_id, buffer);
      if (!accepted) {
        // Rejected messages are put into pending queue.
        AddUnicastHeader(application_id, variable_group_id, partition_id,
                         stage_id, buffer);
        route_pending_queue_.push_back(buffer);
      }
    } else {
      AddUnicastHeader(application_id, variable_group_id, partition_id,
                       stage_id, deliver_buffer);
      ProcessUnicastMessage(deliver_buffer);
    }
  }
  evbuffer_free(buffer);
}

void WorkerDataRouter::ProcessDirectMessage(struct evbuffer* buffer) {
  data_receiver_->ReceiveDirectData(buffer);
}

/*
 * Helper functions.
 */

void WorkerDataRouter::AddUnicastHeader(ApplicationId application_id,
                                        VariableGroupId variable_group_id,
                                        PartitionId partition_id,
                                        StageId stage_id,
                                        struct evbuffer* buffer) {
  // Add header to the message.
  const auto length = evbuffer_get_length(buffer);
  auto header = message::AddDataHeader(buffer);
  header->length = length;
  header->category_group =
      message::MessageCategoryGroup::APPLICATION_DATA_ROUTE;
  header->category = message::MessageCategory::ROUTE_DATA_UNICAST;
  header->partition_map_version = PartitionMapVersion::INVALID;
  header->to_application_id = application_id;
  header->to_variable_group_id = variable_group_id;
  header->to_partition_id = partition_id;
  header->to_stage_id = stage_id;
}

void WorkerDataRouter::AddMulticastHeader(
    ApplicationId application_id, VariableGroupId variable_group_id,
    StageId stage_id, PartitionMapVersion partition_map_version,
    struct evbuffer* buffer) {
  // Add header to the message.
  const auto length = evbuffer_get_length(buffer);
  auto header = message::AddDataHeader(buffer);
  header->length = length;
  header->category_group =
      message::MessageCategoryGroup::APPLICATION_DATA_ROUTE;
  header->category = message::MessageCategory::ROUTE_DATA_MULTICAST;
  header->partition_map_version = partition_map_version;
  header->to_application_id = application_id;
  header->to_variable_group_id = variable_group_id;
  header->to_partition_id = PartitionId::INVALID;
  header->to_stage_id = stage_id;
}

WorkerDataRouter::PeerRecord* WorkerDataRouter::GetPeerRecordIfReady(
    WorkerId worker_id) {
  if (worker_id == WorkerId::INVALID) {
    return nullptr;
  }
  // self_worker_id_ is not in the map.
  auto iter = worker_id_to_status_.find(worker_id);
  if (iter == worker_id_to_status_.end()) {
    return nullptr;
  }
  if (!iter->second.is_ready) {
    return nullptr;
  }
  return std::addressof(iter->second);
}

void WorkerDataRouter::TriggerRefresh() {
  std::list<struct evbuffer*> buffer_queue;
  std::swap(buffer_queue, route_pending_queue_);
  for (auto buffer : buffer_queue) {
    SendUnicastData(buffer);
  }
  auto iter = direct_pending_queue_.begin();
  while (iter != direct_pending_queue_.end()) {
    auto peer_record = GetPeerRecordIfReady(iter->first);
    if (peer_record != nullptr) {
      for (auto buffer : iter->second) {
        peer_record->low_priority_send_queue.push_back(buffer);
      }
      iter = direct_pending_queue_.erase(iter);
      CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
    } else {
      ++iter;
    }
  }
}

}  // namespace canary
