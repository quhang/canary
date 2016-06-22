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
    //! Whether the router is waiting for the peer to send activation message,
    // which includes its worker id.
    bool to_be_activated = false;
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
  };

  //! Default backlog argument. Concurrent pending TCP connecting events.
  static const int kBacklog = -1;

 public:
  WorkerDataRouter() {}
  virtual ~WorkerDataRouter() {}
  void Initialize(WorkerId worker_id,
                  network::EventMainThread* event_main_thread,
                  WorkerReceiveDataInterface* data_receiver,
                  const std::string& route_service) {
    // TODO
    return;
    self_worker_id_ = worker_id;
    event_main_thread_ = CHECK_NOTNULL(event_main_thread);
    // TODO
    // data_receiver_ = CHECK_NOTNULL(data_receiver);
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

 public:
  /*
   * Peer channel related.
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
  //! Routes data to a partition. The header is not added.
  void SendDataToPartition(ApplicationId application_id,
                           VariableGroupId variable_group_id,
                           PartitionId partition_id,
                           struct evbuffer* buffer) override {}
  //! Sends data to a worker. Used for data partition migration, or restoring
  // data partitions from storage. The header is not added.
  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override {}
  //! Broadcasts data to all tasks in a stage.
  void BroadcastDataToPartition(ApplicationId application_id,
                                VariableGroupId variable_group_id,
                                struct evbuffer* buffer) override {}

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
      InitializePeerRecord(peer_record);
    }
  }

  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen) {
    CHECK_EQ(listener, listening_event_);
    std::string host, service;
    network::translate_sockaddr_to_string(address, socklen, &host, &service);
    VLOG(1) << host << ":" << service << " reaches the worker routing port.";
    // Handshake protocol:
    // slave -> master: worker_id.
    auto peer_record = new PeerRecord();
    peer_record->worker_id = WorkerId::INVALID;
    peer_record->socket_fd = socket_fd;
    peer_record->is_ready = false;
    peer_record->to_be_activated = true;
    CHECK_EQ(event_base_once(event_base_, socket_fd, EV_READ, DispatchReadEvent,
                             peer_record, nullptr),
             0);
    peer_record->read_event = nullptr;
    peer_record->write_event = nullptr;
    peer_record->receive_buffer = evbuffer_new();
    peer_record->router = this;
  }

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

  void InitializePeerRecord(PeerRecord* peer_record) {}

  void CleanUpPeerRecord(PeerRecord* peer_record) {}

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
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_