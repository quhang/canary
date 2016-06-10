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
  virtual void ReceiveCommandFromWorker(
      WorkerId worker_id, Command command) = 0;
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
 public:
  //! Initializes the partition map for an application. Broadcast the map to all
  // workers.
  void InitializeApplicationPartitionMap(
      ApplicationId application_id, const PartitionMap& partition_map);
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
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_COMMUNICATION_MANAGER_H_
