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
 * @file src/worker/worker_data_router.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerDataRouter.
 */

#ifndef CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_
#define CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_

#include <event2/event.h>
#include <event2/listener.h>
#include <list>
#include <map>
#include <string>

#include "shared/internal.h"
#include "shared/partition_map.h"
#include "message/message_include.h"

#include "worker/worker_communication_interface.h"

namespace canary {

class WorkerDataRouter : public WorkerSendDataInterface {
 private:
  typedef WorkerDataRouter SelfType;
  /**
   * The data structure associated with a peer worker.
   */
  struct PeerRecord {
    //! The id of the peer worker.
    WorkerId worker_id = WorkerId::INVALID;
    //! Peer hostname.
    std::string host;
    //! Peer port names.
    std::string route_service;
    //! Peer-peer communication socket.
    int socket_fd = -1;
    //! Whether the peer is ready.
    bool is_ready = false;
    //! Whether the channel is passive.
    bool is_passive = false;
    //! Persistent read event.
    struct event* read_event = nullptr;
    //! Write event.
    struct event* write_event = nullptr;
    //! Send buffer, which is grabbed from the sending queue.
    struct evbuffer* send_buffer = nullptr;
    //! Send queue, which owns all the buffers.
    std::list<struct evbuffer*> send_queue;
    //! Send queue, which owns all the buffers.
    std::list<struct evbuffer*> low_priority_send_queue;
    //! Receive buffer.
    struct evbuffer* receive_buffer = nullptr;
    //! Pointer to the router.
    WorkerDataRouter* router = nullptr;
    //! Sequence number for sending.
    SequenceNumber next_sending_sequence_no = 0;
    //! Sequence number for acknowledging.
    SequenceNumber next_receiving_sequence_no = 0;
  };

  //! Default backlog argument. Concurrent pending TCP connecting events.
  static const int kBacklog = -1;

 public:
  //! Constructor.
  WorkerDataRouter() {}

  //! Destructor.
  virtual ~WorkerDataRouter() {}

  //! Initializes the router.
  void Initialize(WorkerId worker_id,
                  network::EventMainThread* event_main_thread,
                  WorkerReceiveDataInterface* data_receiver,
                  const std::string& route_service = FLAGS_worker_service) {
    self_worker_id_ = worker_id;
    event_main_thread_ = CHECK_NOTNULL(event_main_thread);
    data_receiver_ = CHECK_NOTNULL(data_receiver);
    event_base_ = event_main_thread_->get_event_base();
    route_service_ = route_service;
    // Registers the listening port.
    listening_socket_ =
        network::allocate_and_bind_listen_socket(route_service_);
    CHECK_NE(listening_socket_, -1);
    // Starts the listening service.
    listening_event_ = CHECK_NOTNULL(
        evconnlistener_new(event_base_, &DispatchAcceptEvent, this,
                           LEV_OPT_CLOSE_ON_FREE, kBacklog, listening_socket_));
    evconnlistener_set_error_cb(listening_event_, DispatchAcceptErrorEvent);
    // Sets the initialization flag.
    is_initialized_ = true;
    LOG(INFO) << "Worker data router is initialized.";
  }

  //! Shuts down the router.
  void Finalize() { LOG(FATAL) << "Not implemented."; }

 public:
  /*
   * Public dispatching calls.
   */

  //! Dispatches the accept event of the peer channel.
  // Sync call.
  static void DispatchAcceptEvent(struct evconnlistener* listener,
                                  int socket_fd, struct sockaddr* address,
                                  int socklen, void* arg) {
    auto router = reinterpret_cast<SelfType*>(arg);
    router->CallbackAcceptEvent(listener, socket_fd, address, socklen);
  }

  //! Dispatches the accept error event of the peer channel.
  // Sync call.
  static void DispatchAcceptErrorEvent(struct evconnlistener*, void*) {
    LOG(FATAL) << "Failure on the listener: "
               << network::get_error_message(network::get_last_error_number());
  }

  //! Dispatches the connection feedback event of the peer channel.
  // Sync call.
  static void DispatchConnectEvent(int socket_fd, short, void* arg) {
    auto peer_record = reinterpret_cast<PeerRecord*>(arg);
    CHECK_EQ(socket_fd, peer_record->socket_fd);
    peer_record->router->CallbackConnectEvent(peer_record);
  }

  //! Dispatches the read event when the handshake message is received.
  // Sync call.
  static void DispatchPassiveConnectEvent(int socket_fd, short,  // NOLINT
                                          void* arg) {
    auto peer_record = reinterpret_cast<PeerRecord*>(arg);
    CHECK_EQ(socket_fd, peer_record->socket_fd);
    peer_record->router->CallbackPassiveConnectEvent(peer_record);
  }

  //! Dispatches the read event on the peer channel.
  // Sync call.
  static void DispatchReadEvent(int socket_fd, short, void* arg) {  // NOLINT
    auto peer_record = reinterpret_cast<PeerRecord*>(arg);
    CHECK_EQ(socket_fd, peer_record->socket_fd);
    peer_record->router->CallbackReadEvent(peer_record);
  }

  //! Dispatches the write event on the peer channel.
  // Sync call.
  static void DispatchWriteEvent(int socket_fd, short, void* arg) {  // NOLINT
    auto peer_record = reinterpret_cast<PeerRecord*>(arg);
    CHECK_EQ(socket_fd, peer_record->socket_fd);
    peer_record->router->CallbackWriteEvent(peer_record);
  }

 public:
  /*
   * Public sending data interface.
   */

  //! Routes data to a partition. The header is not added.
  // Async call.
  void SendDataToPartition(ApplicationId application_id,
                           VariableGroupId variable_group_id,
                           PartitionId partition_id,
                           struct evbuffer* buffer) override {
    // Add header to the message.
    const auto length = evbuffer_get_length(buffer);
    auto header = message::AddDataHeader(buffer);
    header->length = length;
    header->category_group =
        message::MessageCategoryGroup::APPLICATION_DATA_ROUTE;
    header->category = message::MessageCategory::ROUTE_DATA_UNICAST;
    header->to_application_id = application_id;
    header->to_variable_group_id = variable_group_id;
    header->to_partition_id = partition_id;
    event_main_thread_->AddInjectedEvent(
        std::bind(&SelfType::RouteUnicastData, this, buffer));
  }

  //! Sends data to a worker. Used for data partition migration, or restoring
  // data partitions from storage. The header is added.
  // Async call.
  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override {
    CHECK(message::CheckIsIntegrateDataMessage(buffer));
    CHECK(message::ExamineDataHeader(buffer)->category_group ==
          message::MessageCategoryGroup::APPLICATION_DATA_DIRECT);
    event_main_thread_->AddInjectedEvent(std::bind(
        &SelfType::SendDataToWorkerInternal, this, worker_id, buffer));
  }

  //! Broadcasts data to all tasks in a stage.
  // Async call.
  void BroadcastDataToPartition(ApplicationId application_id,
                                VariableGroupId variable_group_id,
                                struct evbuffer* buffer) override {
    event_main_thread_->AddInjectedEvent(
        std::bind(&SelfType::BroadcastDataToPartition, this, application_id,
                  variable_group_id, buffer));
  }

 private:
  // Sync call.
  void SendDataToWorkerInternal(WorkerId worker_id, struct evbuffer* buffer) {
    LOG(FATAL) << "Not implemented.";
  }

  // Sync call.
  void BroadcastDataToPartitionInternal(ApplicationId application_id,
                                        VariableGroupId variable_group_id,
                                        struct evbuffer* buffer) {
    LOG(FATAL) << "Not implemented.";
  }

  //! Routes a unicast data message, with its header.
  // Sync call.
  void RouteUnicastData(struct evbuffer* buffer) {
    auto header = message::ExamineDataHeader(buffer);
    const auto dest_worker_id = internal_partition_map_.QueryWorkerId(
        header->to_application_id, header->to_variable_group_id,
        header->to_partition_id);
    header->partition_map_version = internal_partition_map_version_;
    AppendPeerSendingQueue(dest_worker_id, buffer);
  }

  //! Adds a data message to the peer sending queue.
  // Sync call.
  void AppendPeerSendingQueue(WorkerId worker_id, struct evbuffer* buffer) {
    auto peer_record = GetPeerRecordIfReady(worker_id);
    if (peer_record != nullptr) {
      peer_record->send_queue.push_back(buffer);
      CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
    } else {
      AddToPendingQueue(buffer);
    }
  }

  //! Adds a data message to the pending queue.
  // Sync call.
  void AddToPendingQueue(struct evbuffer* buffer) {
    auto header = message::ExamineDataHeader(buffer);
    CHECK(header->category == message::MessageCategory::ROUTE_DATA_UNICAST)
        << "Not implemented.";
    pending_queue_.push_back(buffer);
  }

  //! Trigger refreshing, i.e. reroutes apending messages.
  void TriggerRefresh() { ReexaminePendingQueue(); }

  //! Resends messages in the pending queue again.
  // Sync call.
  void ReexaminePendingQueue() {
    std::list<struct evbuffer*> buffer_queue;
    std::swap(buffer_queue, pending_queue_);
    for (auto buffer : buffer_queue) {
      RouteUnicastData(buffer);
    }
  }

  //! Returns the peer record if it is ready.
  // Sync call.
  PeerRecord* GetPeerRecordIfReady(WorkerId worker_id) {
    if (worker_id == WorkerId::INVALID) {
      return nullptr;
    }
    auto iter = worker_id_to_status_.find(worker_id);
    if (iter == worker_id_to_status_.end()) {
      return nullptr;
    }
    if (!iter->second.is_ready) {
      return nullptr;
    }
    return std::addressof(iter->second);
  }

 public:
  /*
   * Core logic to handle connect/read/write events.
   */

  //! Initiates the connection.
  // Sync call.
  void CallbackInitiateEvent(PeerRecord* peer_record) {
    peer_record->socket_fd = network::allocate_and_connect_socket(
        peer_record->host, peer_record->route_service);
    CHECK_GE(peer_record->socket_fd, 0);
    // Triggered when the channel is ready to write.
    CHECK_EQ(event_base_once(event_base_, peer_record->socket_fd, EV_WRITE,
                             DispatchConnectEvent, this, nullptr),
             0);
  }

  //! Responds to connection feedback.
  // Sync call.
  void CallbackConnectEvent(PeerRecord* peer_record) {
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

  //! Receives a connection.
  // Sync call.
  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen) {
    CHECK_EQ(listener, listening_event_);
    std::string host, service;
    network::translate_sockaddr_to_string(address, socklen, &host, &service);
    VLOG(1) << host << ":" << service << " reaches the worker routing port.";
    // Handshake protocol:
    // slave -> master: worker_id.
    auto peer_record = InitializePassivePeerRecord(socket_fd, host, service);
    CHECK_EQ(event_base_once(event_base_, socket_fd, EV_READ,
                             DispatchPassiveConnectEvent, peer_record, nullptr),
             0);
  }

  //! Responds to the handshake message.
  // Sync call.
  void CallbackPassiveConnectEvent(PeerRecord* peer_record) {
    CHECK(peer_record->is_passive);
    CHECK(!peer_record->is_ready);
    struct evbuffer* receive_buffer = peer_record->receive_buffer;
    const int socket_fd = peer_record->socket_fd;

    // Expects handshake message.
    int status = 0;
    while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
      if (struct evbuffer* whole_message =
              message::SegmentDataMessage(receive_buffer)) {
        auto header =
            CHECK_NOTNULL(message::ExamineControlHeader(whole_message));
        CHECK(header->category ==
              message::MessageCategory::DIRECT_DATA_HANDSHAKE);
        message::DirectDataHandshake message;
        message::RemoveControlHeader(whole_message);
        message::DeserializeMessage(whole_message, &message);
        ActivatePassivePeerRecord(message.from_worker_id, peer_record);
        return;
      }
    }
    if (status == 0 || (status == -1 && !network::is_blocked())) {
      evbuffer_free(peer_record->receive_buffer);
      delete peer_record;
      LOG(WARNING) << "Failed incoming connection!";
    } else {
      // Waits for more data.
      CHECK_EQ(
          event_base_once(event_base_, socket_fd, EV_READ,
                          DispatchPassiveConnectEvent, peer_record, nullptr),
          0);
    }
  }

  //! Receives data.
  // Sync call.
  void CallbackReadEvent(PeerRecord* peer_record) {
    struct evbuffer* receive_buffer = peer_record->receive_buffer;
    const int socket_fd = peer_record->socket_fd;

    int status = 0;
    while ((status = evbuffer_read(receive_buffer, socket_fd, -1)) > 0) {
      while (struct evbuffer* whole_message =
                 message::SegmentDataMessage(receive_buffer)) {
        ProcessIncomingMessage(whole_message);
      }
    }
    if (status == 0 || (status == 1 && !network::is_blocked())) {
      CleanUpPeerRecord(peer_record);
    }
  }

  //! Ready to write data.
  // Sync call.
  void CallbackWriteEvent(PeerRecord* peer_record) {
    peer_record->send_buffer =
        network::send_as_much(peer_record->socket_fd, peer_record->send_buffer,
                              &peer_record->send_queue);
    if (peer_record->send_buffer != nullptr) {
      CheckWriteChannel(peer_record);
    } else {
      // Send low priority queues later.
      peer_record->send_buffer = network::send_as_much(
          peer_record->socket_fd, peer_record->send_buffer,
          &peer_record->low_priority_send_queue);
      CleanUpPeerRecord(peer_record);
    }
  }

  //! Deals with write event errors.
  // Sync call.
  void CheckWriteChannel(PeerRecord* peer_record) {
    if (peer_record->send_buffer != nullptr) {
      // Channel is blocked or has error.
      if (network::is_blocked()) {
        CHECK_EQ(event_add(peer_record->write_event, nullptr), 0);
      } else {
        CleanUpPeerRecord(peer_record);
      }
    }
  }

  void ProcessIncomingMessage(struct evbuffer* buffer) {}

 public:
  /*
   * Sync call interfaces invoked by WorkerCommunicationManager.
   */

  //! Sets up partition map and worker peers.
  void UpdatePartitionMapAndWorker(
      message::UpdatePartitionMapAndWorker* message) {
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

  //! Updates partition map.
  void UpdatePartitionMapAddApplication(
      message::UpdatePartitionMapAddApplication* message) {
    internal_partition_map_version_ = message->version_id;
    *internal_partition_map_.AddPerApplicationPartitionMap(
        message->add_application_id) =
        std::move(*message->per_application_partition_map);
    delete message->per_application_partition_map;

    TriggerRefresh();
    delete message;
  }

  //! Updates partition map.
  void UpdatePartitionMapDropApplication(
      message::UpdatePartitionMapDropApplication* message) {
    internal_partition_map_version_ = message->version_id;
    CHECK(internal_partition_map_.DeletePerApplicationPartitionMap(
        message->drop_application_id));
    delete message;
  }

  //! Updates partition map.
  void UpdatePartitionMapIncremental(
      message::UpdatePartitionMapIncremental* message) {
    internal_partition_map_version_ = message->version_id;
    internal_partition_map_.MergeUpdate(*message->partition_map_update);
    delete message->partition_map_update;

    TriggerRefresh();
    delete message;
  }

  //! Adds a worker.
  void UpdateAddedWorker(message::UpdateAddedWorker* message) {
    if (message->added_worker_id < self_worker_id_) {
      auto peer_record = InitializeActivePeerRecord(
          message->added_worker_id, message->network_address.host,
          message->network_address.service);
      CallbackInitiateEvent(peer_record);
    }

    TriggerRefresh();
    delete message;
  }

  //! Shuts down the worker.
  void ShutDownWorker(message::ShutDownWorker* message) {
    delete message;
    Finalize();
  }

 public:
  /*
   * Lifetime of a peer record.
   */

  //! Initializes an active peer record, before connecting to that peer.
  PeerRecord* InitializeActivePeerRecord(WorkerId worker_id,
                                         const std::string& worker_host,
                                         const std::string& worker_service) {
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
  void ActivateActivePeerRecord(PeerRecord* peer_record) {
    CHECK_NE(peer_record->socket_fd, -1);
    peer_record->is_ready = true;
    CHECK(!peer_record->is_passive);
    peer_record->read_event = CHECK_NOTNULL(
        event_new(event_base_, peer_record->socket_fd, EV_READ | EV_PERSIST,
                  &DispatchReadEvent, peer_record));
    CHECK_EQ(event_add(peer_record->read_event, nullptr), 0);
    peer_record->write_event =
        CHECK_NOTNULL(event_new(event_base_, peer_record->socket_fd, EV_WRITE,
                                &DispatchWriteEvent, peer_record));
    CHECK(peer_record->send_buffer == nullptr);
    peer_record->receive_buffer = evbuffer_new();
    // Writes handshake message
    {
      message::DirectDataHandshake message;
      message.from_worker_id = peer_record->worker_id;
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
  PeerRecord* InitializePassivePeerRecord(int socket_fd,
                                          const std::string& host,
                                          const std::string& service) {
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
  void ActivatePassivePeerRecord(WorkerId from_worker_id,
                                 PeerRecord* old_peer_record) {
    CHECK(worker_id_to_status_.find(from_worker_id) ==
          worker_id_to_status_.end());
    auto& peer_record = worker_id_to_status_[from_worker_id];
    peer_record = *old_peer_record;
    delete old_peer_record;
    peer_record.worker_id = from_worker_id;
    CHECK(peer_record.is_passive);
    peer_record.is_ready = true;
    peer_record.read_event = CHECK_NOTNULL(
        event_new(event_base_, peer_record.socket_fd, EV_READ | EV_PERSIST,
                  &DispatchReadEvent, &peer_record));
    CHECK_EQ(event_add(peer_record.read_event, nullptr), 0);
    peer_record.write_event =
        CHECK_NOTNULL(event_new(event_base_, peer_record.socket_fd, EV_WRITE,
                                &DispatchWriteEvent, &peer_record));
    CHECK(peer_record.send_buffer == nullptr);
    CHECK_NOTNULL(peer_record.receive_buffer);
    CHECK_EQ(peer_record.router, this);
    // Refreshes the pending messages to be routed.
    TriggerRefresh();
  }

  //! Cleans up a peer record.
  void CleanUpPeerRecord(PeerRecord* peer_record) {
    network::close_socket(peer_record->socket_fd);
    if (peer_record->read_event) {
      event_free(peer_record->read_event);
    }
    if (peer_record->write_event) {
      event_free(peer_record->write_event);
    }
    if (peer_record->send_buffer) {
      evbuffer_free(peer_record->send_buffer);
    }
    if (peer_record->receive_buffer) {
      evbuffer_free(peer_record->receive_buffer);
    }
    for (auto buffer : peer_record->send_queue) {
      if (buffer) {
        evbuffer_free(buffer);
      }
    }
    // TODO: Put sending messages into the pending queue.
    const auto worker_id = peer_record->worker_id;
    worker_id_to_status_.erase(worker_id);
    // TODO: Notify the controller that the connection is down.
  }

 private:
  WorkerId self_worker_id_ = WorkerId::INVALID;
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  WorkerReceiveDataInterface* data_receiver_ = nullptr;
  std::string route_service_;

  bool is_initialized_ = false;

  std::map<WorkerId, PeerRecord> worker_id_to_status_;

  PartitionMap internal_partition_map_;
  PartitionMapVersion internal_partition_map_version_ =
      PartitionMapVersion::FIRST;

  int listening_socket_ = -1;
  struct evconnlistener* listening_event_ = nullptr;

  // Routing message that cannot be delivered now.
  std::list<struct evbuffer*> pending_queue_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_
