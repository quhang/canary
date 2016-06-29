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
 * @file src/controller/controller_communication_interface.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerCommunicationInterface.
 */

#ifndef CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_INTERFACE_H_
#define CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_INTERFACE_H_

#include "shared/internal.h"
#include "shared/partition_map.h"

namespace canary {

/**
 * The command sending interface at the controller side.
 */
class ControllerSendCommandInterface {
 public:
  //! Sends a command to a worker. The command will not reach the worker if it
  // is not notified that the worker is up. The buffer ownership is transferred.
  virtual void SendCommandToWorker(WorkerId worker_id,
                                   struct evbuffer* buffer) = 0;

  /*
   * The partition map will be synchronized to workers automatically.
   */
  //! Updates the partition map by adding an application. The ownership of the
  // partition map is transferred.
  virtual void AddApplication(
      ApplicationId application_id,
      PerApplicationPartitionMap* per_application_partition_map) = 0;

  //! Updates the partition map by dropping an application.
  virtual void DropApplication(ApplicationId application_id) = 0;

  //! Updates the partition map incrementally. The ownership of the update data
  // structure is transferred.
  virtual void UpdatePartitionMap(PartitionMapUpdate* partition_map_update) = 0;

  //! Shuts down a worker, which is notified up.
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

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  virtual void NotifyWorkerIsUp(WorkerId worker_id) = 0;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_INTERFACE_H_
