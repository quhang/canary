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

/**
 * The data router at a worker routes data to other workers based on the global
 * partition map given by the worker communication manager.
 * @see src/worker/worke_communication_manager.cc.
 *
 * Once the partition map indicates that a data message should be routed to the
 * worker itself, the router will deliver the message to the user. It is
 * possible that the partition map is stale, and the message is delivered to the
 * user incorrectly. In this case, the user can send the message again. The
 * router will hold such message (whose destination seems to be itself) until
 * the partition map is further updated.
 *
 * The router can only be used after it is initialized, otherwise it will panic.
 * In case connected peer fails, the router will report to the communication
 * manager with a NotifyDisconnectWorker message.
 *
 * TODO(quhang): broadcast, sequence number.
 */
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
    //! Whether the peer is ready for exchanging packets.
    bool is_ready = false;
    //! Whether the channel is passive, i.e. the router did not initiate the
    // channel.
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
  // Prepare call.
  void Initialize(WorkerId worker_id,
                  network::EventMainThread* event_main_thread,
                  WorkerReceiveDataInterface* data_receiver,
                  const std::string& route_service = FLAGS_worker_service);

  //! Shuts down the router.
  void Finalize();

 public:
  /*
   * Public static methods used for libevent to dispatch events.
   */

  //! Dispatches the accept event for a passive channel.
  // Sync call.
  static void DispatchAcceptEvent(struct evconnlistener* listener,
                                  int socket_fd, struct sockaddr* address,
                                  int socklen, void* arg);

  //! Dispatches the accept error event of a passive channel.
  // Sync call.
  static void DispatchAcceptErrorEvent(struct evconnlistener*, void*);

  //! Dispatches the connection feedback event of an active channel.
  // Sync call.
  static void DispatchConnectEvent(int socket_fd, short, void* arg);  // NOLINT

  //! Dispatches the read event of initial handshake data from a passive
  // channel.
  // Sync call.
  static void DispatchPassiveConnectEvent(int socket_fd, short,  // NOLINT
                                          void* arg);

  //! Dispatches the read event.
  // Sync call.
  static void DispatchReadEvent(int socket_fd, short, void* arg);  // NOLINT

  //! Dispatches the write event.
  // Sync call.
  static void DispatchWriteEvent(int socket_fd, short, void* arg);  // NOLINT

 public:
  /*
   * Core logic to handle connect/read/write events.
   */

  //! Initiates an active connection.
  // Sync call.
  void CallbackInitiateEvent(PeerRecord* peer_record);

  //! Responds to connection feedback of an active connection.
  // Sync call.
  void CallbackConnectEvent(PeerRecord* peer_record);

  //! Accepts a passive connection.
  // Sync call.
  void CallbackAcceptEvent(struct evconnlistener* listener, int socket_fd,
                           struct sockaddr* address, int socklen);

  //! Receives initial handshake data from a passive connection.
  // Sync call.
  void CallbackPassiveConnectEvent(PeerRecord* peer_record);

  //! Receives data.
  // Sync call.
  void CallbackReadEvent(PeerRecord* peer_record);

  //! Ready to write data.
  // Sync call.
  void CallbackWriteEvent(PeerRecord* peer_record);

 private:
  /*
   * Lifetime of a peer record.
   */

  //! Initializes an active peer record, before connecting to that peer.
  PeerRecord* InitializeActivePeerRecord(WorkerId worker_id,
                                         const std::string& worker_host,
                                         const std::string& worker_service);

  //! Activates the peer record.
  void ActivateActivePeerRecord(PeerRecord* peer_record);

  //! Initializes a passive peer record, after a connection comes in.
  PeerRecord* InitializePassivePeerRecord(int socket_fd,
                                          const std::string& host,
                                          const std::string& service);

  //! Activates a peer record, after receiving its worker id.
  void ActivatePassivePeerRecord(WorkerId from_worker_id,
                                 PeerRecord* old_peer_record);

  //! Cleans up a peer record.
  void CleanUpPeerRecord(PeerRecord* peer_record);

 public:
  /*
   * Public sending data interface. Must be called after initialization.
   */

  //! Routes data to a partition. The header is not included in the buffer.
  // Async call.
  void SendDataToPartition(ApplicationId application_id,
                           VariableGroupId variable_group_id,
                           PartitionId partition_id,
                           struct evbuffer* buffer) override;

  //! Sends data to a worker. The header is included in the buffer.
  // Async call.
  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override;

  //! Broadcasts data to all partitions in a variable group. The header is not
  // included in the buffer.
  // Async call.
  void BroadcastDataToPartition(ApplicationId application_id,
                                VariableGroupId variable_group_id,
                                struct evbuffer* buffer) override;

 public:
  /*
   * Synchronous call interfaces invoked by WorkerCommunicationManager.
   */

  //! Sets up partition map and worker peers. Only connecting to smaller worker
  // id.
  void UpdatePartitionMapAndWorker(
      message::UpdatePartitionMapAndWorker* message);

  //! Updates partition map.
  void UpdatePartitionMapAddApplication(
      message::UpdatePartitionMapAddApplication* message);

  //! Updates partition map.
  void UpdatePartitionMapDropApplication(
      message::UpdatePartitionMapDropApplication* message);

  //! Updates partition map.
  void UpdatePartitionMapIncremental(
      message::UpdatePartitionMapIncremental* message);

  //! Adds a worker.
  void UpdateAddedWorker(message::UpdateAddedWorker* message);

  //! Shuts down the worker.
  void ShutDownWorker(message::ShutDownWorker* message);

 private:
  /*
   * Routing facilities.
   */

  //! Sends a unicast message. The header is included.
  // Sync call.
  void SendUnicastData(struct evbuffer* buffer);

  //! Adds a unicast/multicast message to the sending queue.
  // Sync call.
  void AppendSendingQueue(WorkerId worker_id, struct evbuffer* buffer);

  //! Sends a multicast message.
  void AddHeaderAndSendMulticastData(ApplicationId application_id,
                                     VariableGroupId variable_group_id,
                                     struct evbuffer* buffer);

  //! Sends a direct message. The header is included.
  void SendDirectData(WorkerId worker_id, struct evbuffer* buffer);

  //! Adds a direct message to the low priority sending queue.
  void AppendLowPrioritySendingQueue(WorkerId worker_id,
                                     struct evbuffer* buffer);

  /*
   * Receiving messages.
   */

  //! Processes incoming messages and delivers it.
  // Sync call.
  void ProcessIncomingMessage(struct evbuffer* buffer);

  //! Processes unicast message.
  // Sync call.
  void ProcessUnicastMessage(struct evbuffer* buffer);

  //! Processes multicast message.
  // Sync call.
  void ProcessMulticastMessage(struct evbuffer* buffer);

  //! Processes direct message.
  // Sync call.
  void ProcessDirectMessage(struct evbuffer* buffer);

  /*
   * Helper functions.
   */

  //! Adds unicast header.
  // Async call.
  void AddUnicastHeader(ApplicationId application_id,
                        VariableGroupId variable_group_id,
                        PartitionId partition_id, struct evbuffer* buffer);

  //! Returns the peer record if it is ready.
  // Sync call.
  PeerRecord* GetPeerRecordIfReady(WorkerId worker_id);

  //! Trigger refreshing, i.e. sending pending messages.
  // Sync call.
  void TriggerRefresh();

 private:
  //! Initialization flag.
  bool is_initialized_ = false;
  //! Shutdown flag.
  bool is_shutdown_ = false;
  //! Its own worker id.
  WorkerId self_worker_id_ = WorkerId::INVALID;

  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  WorkerReceiveDataInterface* data_receiver_ = nullptr;
  std::string route_service_;

  std::map<WorkerId, PeerRecord> worker_id_to_status_;

  PartitionMap internal_partition_map_;
  PartitionMapVersion internal_partition_map_version_ =
      PartitionMapVersion::FIRST;

  int listening_socket_ = -1;
  struct evconnlistener* listening_event_ = nullptr;

  //! Routing message that cannot be delivered now.
  std::list<struct evbuffer*> route_pending_queue_;
  std::map<WorkerId, std::list<struct evbuffer*>> direct_pending_queue_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_
