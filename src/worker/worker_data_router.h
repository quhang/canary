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

#include <list>
#include <string>

#include "shared/internal.h"
#include "shared/partition_map.h"

#include "worker/worker_communication_interface.h"

namespace canary {

class WorkerDataRouter : public WorkerSendDataInterface {
 private:
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

  //! Default backlog argument. Concurrent pending TCP connecting events.
  static const int kBacklog = -1;

 public:
  WorkerDataRouter() {}
  virtual ~WorkerDataRouter() {}
  void Initialize(network::EventMainThread* event_main_thread,
                  WorkerReceiveDataInterface* data_receiver,
                  const std::string& route_service) {}

 public:
  /*
   * Peer channel related.
   */

  //! Dispatches the accept event of the peer channel.
  // Sync call.
  static void DispatchPeerAcceptEvent(struct evconnlistener* listener,
                                      int socket_fd, struct sockaddr* address,
                                      int socklen, void* arg) {}

  //! Dispatches the accept error event of the peer channel.
  // Sync call.
  static void DispatchPeerAcceptErrorEvent(struct evconnlistener*, void*) {}

  //! Dispatches the connection feedback event of the peer channel.
  // Sync call.
  static void DispatchPeerConnectEvent(int socket_fd, void*) {}

  //! Dispatches the read event on the peer channel.
  // Sync call.
  static void DispatchPeerReadEvent(int socket_fd, short,  // NOLINT
                                    void* arg) {}

  //! Dispatches the write event on the peer channel.
  // Sync call.
  static void DispatchPeerWriteEvent(int socket_fd, short,  // NOLINT
                                     void* arg) {}

 public:
  //! Sends data to a partition. The data is an intermediate data chunk to be
  // routed to a "gather" task.
  void SendDataToPartition(ApplicationId application_id, StageId stage_id,
                           PartitionId partition_id,
                           struct evbuffer*) override {}
  //! Sends data to a worker. Used for data partition migration, or restoring
  // data partitions from storage.
  void SendDataToWorker(WorkerId worker_id, struct evbuffer* buffer) override {}
  //! Reduces data at the worker side, and then sends to the singular task in a
  // stage.
  void ReduceAndSendDataToPartition(
      ApplicationId application_id, StageId stage_id, struct evbuffer* buffer,
      CombinerFunction combiner_function) override {}
  //! Broadcasts data to all tasks in a stage.
  void BroadcastDatatoPartition(ApplicationId application_id, StageId stage_id,
                                struct evbuffer* buffer) override {}

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  struct event_base* event_base_ = nullptr;
  WorkerReceiveCommandInterface* command_receiver_ = nullptr;
  std::string route_service_;

  bool is_initialized_ = false;

  PartitionMap internal_partition_map_;
  PartitionMapVersion internal_partition_map_version_ =
      PartitionMapVersion::FIRST;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_DATA_ROUTER_H_
