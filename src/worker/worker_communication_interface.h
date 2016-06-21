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
 * @file src/worker/worker_communication_interface.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Communication interfaces at the worker side.
 */

#ifndef CANARY_SRC_WORKER_WORKER_COMMUNICATION_INTERFACE_H_
#define CANARY_SRC_WORKER_WORKER_COMMUNICATION_INTERFACE_H_

#include "shared/internal.h"

namespace canary {

/**
 * Sends data to other partitions or other workers.
 */
class WorkerSendDataInterface {
 public:
  //! A combiner function consumes two buffers and merges into one.
  typedef std::function<struct evbuffer*(struct evbuffer*, struct evbuffer*)>
      CombinerFunction;

  //! Sends data to a partition. The data is an intermediate data chunk to be
  // routed to a "gather" task.
  virtual void SendDataToPartition(ApplicationId application_id,
                                   StageId stage_id, PartitionId partition_id,
                                   struct evbuffer* buffer) = 0;
  //! Sends data to a worker. Used for data partition migration, or restoring
  // data partitions from storage.
  virtual void SendDataToWorker(WorkerId worker_id,
                                struct evbuffer* buffer) = 0;
  //! Reduces data at the worker side, and then sends to the singular task in a
  // stage.
  virtual void ReduceAndSendDataToPartition(
      ApplicationId application_id, StageId stage_id, struct evbuffer* buffer,
      CombinerFunction combiner_function) = 0;
  //! Broadcasts data to all tasks in a stage.
  virtual void BroadcastDatatoPartition(ApplicationId application_id,
                                        StageId stage_id,
                                        struct evbuffer* buffer) = 0;
};

/**
 * Receives data from another partition or another worker , as callback
 * functions.
 */
class WorkerReceiveDataInterface {
 public:
  //! Called when receiving data from a partition.
  virtual void ReceiveDataFromPartition(ApplicationId application_id,
                                        StageId stage_id,
                                        PartitionId partition_id,
                                        struct evbuffer* buffer) = 0;
  //! Called when receiving data from a worker.
  virtual void ReceiveDataFromWorker(WorkerId worker_id,
                                     struct evbuffer* buffer) = 0;
};

/**
 * Sends command to the controller.
 */
class WorkerSendCommandInterface {
 public:
  //! Sends a command to the controller.
  virtual void SendCommandToController(struct evbuffer* buffer) = 0;
};

/**
 * Receives command from the controller, as callback functions.
 */
class WorkerReceiveCommandInterface {
 public:
  //! Called when receiving a command from the controller.
  virtual void ReceiveCommandFromController(struct evbuffer* buffer) = 0;

  //! Called when the worker id is assigned.
  virtual void AssignWorkerId(WorkerId worker_id) = 0;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_COMMUNICATION_INTERFACE_H_
