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
 * @file src/controller/controller_scheduler.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerScheduler.
 */

#ifndef CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
#define CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_

#include "shared/internal.h"

#include "shared/partition_map.h"
#include "controller/controller_communication_interface.h"
#include "message/message_include.h"

namespace canary {

class ControllerScheduler : public ControllerReceiveCommandInterface {
 public:
  ControllerScheduler() {}
  virtual ~ControllerScheduler() {}

  void Initialize(network::EventMainThread* event_main_thread,
                  ControllerSendCommandInterface* send_command_interface) {
    event_main_thread_ = CHECK_NOTNULL(event_main_thread);
    send_command_interface_ = CHECK_NOTNULL(send_command_interface);
  }

  //! Called when receiving a command. The message header is kept, and the
  // buffer ownership is transferred.
  void ReceiveCommand(struct evbuffer* buffer) override {
    event_main_thread_->AddInjectedEvent(
        std::bind(&ControllerScheduler::InternalReceiveCommand, this, buffer));
  }

  //! Called when a worker is down, even if it is shut down by the controller.
  void NotifyWorkerIsDown(WorkerId worker_id) override {
    event_main_thread_->AddInjectedEvent(std::bind(
        &ControllerScheduler::InternalNotifyWorkerIsDown, this, worker_id));
  }

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void NotifyWorkerIsUp(WorkerId worker_id) override {
    event_main_thread_->AddInjectedEvent(std::bind(
        &ControllerScheduler::InternalNotifyWorkerIsUp, this, worker_id));
  }

 private:
  void InternalReceiveCommand(struct evbuffer* buffer) {
    LOG(INFO) << "Controller receives command.";
  }

  //! Called when a worker is down, even if it is shut down by the controller.
  void InternalNotifyWorkerIsDown(WorkerId worker_id) {}

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void InternalNotifyWorkerIsUp(WorkerId worker_id) {
    if (++num_workers_ == 2) {
      WorkerId first_worker = WorkerId::FIRST;
      WorkerId second_worker = get_next(first_worker);
      PartitionId first_partition = PartitionId::FIRST;
      PartitionId second_partition = get_next(first_partition);
      ApplicationId application_id = ApplicationId::FIRST;
      {
        message::WorkerLoadApplication load_application;
        load_application.application_id = application_id;
        send_command_interface_->SendCommandToWorker(
            first_worker,
            message::SerializeMessageWithControlHeader(load_application));
        send_command_interface_->SendCommandToWorker(
            second_worker,
            message::SerializeMessageWithControlHeader(load_application));
      }
      {
        message::WorkerLoadPartitions load_partitions;
        load_partitions.application_id = application_id;
        load_partitions.load_partitions.clear();
        load_partitions.load_partitions.emplace_back(VariableGroupId::FIRST,
                                                     first_partition);
        send_command_interface_->SendCommandToWorker(
            first_worker,
            message::SerializeMessageWithControlHeader(load_partitions));
        load_partitions.load_partitions.clear();
        load_partitions.load_partitions.emplace_back(VariableGroupId::FIRST,
                                                     second_partition);
        send_command_interface_->SendCommandToWorker(
            second_worker,
            message::SerializeMessageWithControlHeader(load_partitions));
      }
      auto per_application_partition_map = new PerApplicationPartitionMap();
      per_application_partition_map->SetNumVariableGroup(1);
      per_application_partition_map->SetPartitioning(VariableGroupId::FIRST, 2);
      per_application_partition_map->SetWorkerId(VariableGroupId::FIRST,
                                                 first_partition, first_worker);
      per_application_partition_map->SetWorkerId(
          VariableGroupId::FIRST, second_partition, second_worker);
      send_command_interface_->AddApplication(application_id,
                                              per_application_partition_map);
    }
  }

 private:
  network::EventMainThread* event_main_thread_ = nullptr;
  ControllerSendCommandInterface* send_command_interface_ = nullptr;
  int num_workers_ = 0;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
