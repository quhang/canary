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
 * @file src/controller/controller_scheduler.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerScheduler.
 */

#include "controller/controller_scheduler.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace canary {

void ControllerSchedulerBase::Initialize(
    network::EventMainThread* event_main_thread,
    ControllerSendCommandInterface* send_command_interface) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  send_command_interface_ = CHECK_NOTNULL(send_command_interface);
}

void ControllerSchedulerBase::ReceiveCommand(struct evbuffer* buffer) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalReceiveCommand, this, buffer));
}

void ControllerSchedulerBase::NotifyWorkerIsDown(WorkerId worker_id) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalNotifyWorkerIsDown, this, worker_id));
}

void ControllerSchedulerBase::NotifyWorkerIsUp(WorkerId worker_id) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalNotifyWorkerIsUp, this, worker_id));
}

#define PROCESS_MESSAGE(TYPE, METHOD)                                 \
  case MessageCategory::TYPE: {                                       \
    auto message =                                                    \
        new message::get_message_type<MessageCategory::TYPE>::Type(); \
    message::RemoveControlHeader(buffer);                             \
    message::DeserializeMessage(buffer, message);                     \
    METHOD(message);                                                  \
    break;                                                            \
  }
void ControllerScheduler::InternalReceiveCommand(struct evbuffer* buffer) {
  CHECK_NOTNULL(buffer);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  switch (header->category_group) {
    case MessageCategoryGroup::CONTROLLER_COMMAND:
      switch (header->category) {
        PROCESS_MESSAGE(CONTROLLER_RESPOND_MIGRATION_IN_PREPARED,
                        ProcessMigrationInPrepared);
        PROCESS_MESSAGE(CONTROLLER_RESPOND_MIGRATION_IN_DONE,
                        ProcessMigrationInDone);
        PROCESS_MESSAGE(CONTROLLER_RESPOND_STATUS_OF_PARTITION,
                        ProcessStatusOfPartition);
        PROCESS_MESSAGE(CONTROLLER_RESPOND_STATUS_OF_WORKER,
                        ProcessStatusOfWorker);
        default:
          LOG(FATAL) << "Unexpected message type!";
      }  // switch category.
      break;
    case MessageCategoryGroup::LAUNCH_COMMAND:
      switch (header->category) {
        PROCESS_MESSAGE(LAUNCH_APPLICATION, ProcessLaunchApplication);
        default:
          LOG(FATAL) << "Unexpected message type!";
      }  // switch category.
      break;
    default:
      LOG(FATAL) << "Invalid message header!";
  }  // switch category group.
}
#undef PROCESS_MESSAGE

void ControllerScheduler::FillInApplicationLaunchInfo(
    const message::LaunchApplication& launch_message,
    ApplicationRecord* application_record) {
  application_record->binary_location = launch_message.binary_location;
  application_record->application_parameter =
      launch_message.application_parameter;
  application_record->loaded_application = CanaryApplication::LoadApplication(
      launch_message.binary_location, launch_message.application_parameter,
      &application_record->loading_handle);
  application_record->variable_group_info_map =
      application_record->loaded_application->get_variable_group_info_map();
  application_record->total_partition = 0;
  application_record->complete_partition = 0;
  for (auto& pair : *application_record->variable_group_info_map) {
    application_record->total_partition += pair.second.parallelism;
  }
}

void ControllerScheduler::AssignPartitionToWorker(
    ApplicationRecord* application_record) {
  auto& per_app_partition_map = application_record->per_app_partition_map;
  const auto& variable_group_info_map =
      *application_record->variable_group_info_map;
  per_app_partition_map.SetNumVariableGroup(variable_group_info_map.size());
  for (const auto& pair : variable_group_info_map) {
    per_app_partition_map.SetPartitioning(pair.first, pair.second.parallelism);
    // Birdshot randomized placement.
    std::vector<WorkerId> worker_id_list(pair.second.parallelism);
    for (auto& worker_id : worker_id_list) {
      worker_id = NextAssignWorkerId();
    }
    std::random_shuffle(worker_id_list.begin(), worker_id_list.end());
    auto iter = worker_id_list.begin();
    for (int index = 0; index < pair.second.parallelism; ++index) {
      per_app_partition_map.SetWorkerId(
          pair.first, static_cast<PartitionId>(index), *(iter++));
    }
  }
}

WorkerId ControllerScheduler::NextAssignWorkerId() {
  CHECK(!worker_map_.empty());
  for (auto iter : worker_map_)
    if (iter.first > last_assigned_worker_id_) {
      last_assigned_worker_id_ = iter.first;
      return last_assigned_worker_id_;
    }
  last_assigned_worker_id_ = worker_map_.begin()->first;
  return last_assigned_worker_id_;
}

void ControllerScheduler::RequestLoadApplicationOnAllWorkers(
    ApplicationId application_id, const ApplicationRecord& application_record) {
  message::WorkerLoadApplication load_application_command;
  load_application_command.application_id = application_id;
  load_application_command.binary_location = application_record.binary_location;
  load_application_command.application_parameter =
      application_record.application_parameter;
  for (auto& pair : worker_map_) {
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(load_application_command));
    pair.second.loaded_applications.insert(application_id);
  }
}

void ControllerScheduler::UpdateWorkerOwnedPartitions(
    ApplicationId application_id,
    const PerApplicationPartitionMap& per_app_partition_map) {
  for (int index1 = 0; index1 < per_app_partition_map.QueryNumVariableGroup();
       ++index1) {
    const auto variable_group_id = static_cast<VariableGroupId>(index1);
    for (int index2 = 0;
         index2 < per_app_partition_map.QueryPartitioning(variable_group_id);
         ++index2) {
      const auto partition_id = static_cast<PartitionId>(index2);
      const auto worker_id =
          per_app_partition_map.QueryWorkerId(variable_group_id, partition_id);
      const FullPartitionId full_partition_id{application_id, variable_group_id,
                                              partition_id};
      worker_map_.at(worker_id).owned_partitions[application_id].insert(
          full_partition_id);
    }
  }
}

void ControllerScheduler::RequestLoadPartitions(ApplicationId application_id) {
  for (auto& pair : worker_map_) {
    const auto& owned_partitions = pair.second.owned_partitions;
    if (owned_partitions.find(application_id) == owned_partitions.end()) {
      continue;
    }
    if (owned_partitions.at(application_id).empty()) {
      continue;
    }
    message::WorkerLoadPartitions load_partitions_command;
    load_partitions_command.application_id = application_id;
    for (auto& full_partition_id : owned_partitions.at(application_id)) {
      load_partitions_command.load_partitions.emplace_back(
          full_partition_id.variable_group_id, full_partition_id.partition_id);
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(load_partitions_command));
  }
}

void ControllerScheduler::ProcessLaunchApplication(
    message::LaunchApplication* launch_message) {
  CHECK_NE(launch_message->fix_num_worker, 0);
  if (launch_message->fix_num_worker != -1 &&
      static_cast<int>(worker_map_.size()) < launch_message->fix_num_worker) {
    LOG(WARNING) << "Launching application failed: not enough workers!";
    delete launch_message;
    return;
  }
  if (launch_message->fix_num_worker == -1 && worker_map_.empty()) {
    LOG(WARNING) << "Launching application failed: no worker!";
    delete launch_message;
    return;
  }
  const ApplicationId assigned_application_id = (next_application_id_++);
  auto& application_record = application_map_[assigned_application_id];
  FillInApplicationLaunchInfo(*launch_message, &application_record);
  InitializeLoggingFile();
  std::string output_application_parameter;
  std::remove_copy_if(
      launch_message->application_parameter.begin(),
      launch_message->application_parameter.end(),
      std::back_inserter(output_application_parameter),
      [](auto c) { return c == ' ' || c == '\n' || c == '\t'; });
  fprintf(log_file_, "L %d %s %s\n", get_value(assigned_application_id),
          launch_message->binary_location.c_str(),
          output_application_parameter.c_str());
  FlushLoggingFile();
  AssignPartitionToWorker(&application_record);
  // Sends the partition map to workers.
  send_command_interface_->AddApplication(
      assigned_application_id,
      new PerApplicationPartitionMap(application_record.per_app_partition_map));
  RequestLoadApplicationOnAllWorkers(assigned_application_id,
                                     application_record);
  UpdateWorkerOwnedPartitions(assigned_application_id,
                              application_record.per_app_partition_map);
  RequestLoadPartitions(assigned_application_id);
  delete launch_message;
}

void ControllerScheduler::ProcessMigrationInPrepared(
    message::ControllerRespondMigrationInPrepared* respond_message) {
  delete respond_message;
}

void ControllerScheduler::ProcessMigrationInDone(
    message::ControllerRespondMigrationInDone* respond_message) {
  delete respond_message;
}

void ControllerScheduler::ProcessStatusOfPartition(
    message::ControllerRespondStatusOfPartition* respond_message) {
  InitializeLoggingFile();
  fprintf(log_file_, "P %d %d %d W %d\n",
          get_value(respond_message->application_id),
          get_value(respond_message->variable_group_id),
          get_value(respond_message->partition_id),
          get_value(respond_message->from_worker_id));
  const auto min_timestamp =
      respond_message->timestamp_statistics.begin()->second.second;
  for (auto& pair : respond_message->timestamp_statistics) {
    fprintf(log_file_, "T %d %d %f\n", get_value(pair.first),
            get_value(pair.second.first),
            (pair.second.second - min_timestamp) * 1.e3);
  }
  for (auto& pair : respond_message->cycle_statistics) {
    fprintf(log_file_, "C %d %d %f\n", get_value(pair.first),
            get_value(pair.second.first), pair.second.second * 1.e3);
  }
  if (respond_message->earliest_unfinished_stage_id == StageId::INVALID &&
      respond_message->last_finished_stage_id == StageId::COMPLETE) {
    auto& application_record =
        application_map_.at(respond_message->application_id);
    if (++application_record.complete_partition ==
        application_record.total_partition) {
      CleanUpApplication(respond_message->application_id, &application_record);
    }
  }
  delete respond_message;
}

void ControllerScheduler::ProcessStatusOfWorker(
    message::ControllerRespondStatusOfWorker* respond_message) {
  auto from_worker_id = respond_message->from_worker_id;
  if (worker_map_.find(from_worker_id) != worker_map_.end()) {
    worker_map_[from_worker_id].num_cores = respond_message->num_cores;
  }
  delete respond_message;
}

void ControllerScheduler::CleanUpApplication(
    ApplicationId application_id, ApplicationRecord* application_record) {
  LOG(INFO) << "Application " << get_value(application_id) << " is complete.";
  FlushLoggingFile();

  // Unloads partitions.
  for (auto& pair : worker_map_) {
    auto& owned_partitions = pair.second.owned_partitions;
    if (owned_partitions.find(application_id) == owned_partitions.end()) {
      continue;
    }
    if (owned_partitions.at(application_id).empty()) {
      owned_partitions.erase(application_id);
      continue;
    }
    message::WorkerUnloadPartitions unload_partitions_command;
    unload_partitions_command.application_id = application_id;
    for (auto& full_partition_id : owned_partitions.at(application_id)) {
      unload_partitions_command.unload_partitions.emplace_back(
          full_partition_id.variable_group_id, full_partition_id.partition_id);
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(unload_partitions_command));
    owned_partitions.erase(application_id);
  }
  // Unloads application.
  for (auto& pair : worker_map_) {
    auto& loaded_applications = pair.second.loaded_applications;
    if (loaded_applications.find(application_id) == loaded_applications.end()) {
      continue;
    }
    message::WorkerUnloadApplication unload_application_command;
    unload_application_command.application_id = application_id;
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(unload_application_command));
    loaded_applications.erase(application_id);
  }
  // Unload application binary.
  CanaryApplication::UnloadApplication(application_record->loading_handle,
                                       application_record->loaded_application);
  // Unload partition map.
  send_command_interface_->DropApplication(application_id);
  // Erases the application record.
  application_map_.erase(application_id);
}

void ControllerScheduler::InternalNotifyWorkerIsDown(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
  auto iter = worker_map_.find(worker_id);
  CHECK(iter != worker_map_.end());
  if (!iter->second.owned_partitions.empty()) {
    LOG(WARNING) << "Worker is down while there are active partitions ("
                 << get_value(worker_id) << ")!";
  }
  worker_map_.erase(iter);
}

void ControllerScheduler::InternalNotifyWorkerIsUp(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
  CHECK(worker_map_.find(worker_id) == worker_map_.end());
  worker_map_[worker_id].num_cores = -1;
}

void ControllerScheduler::InitializeLoggingFile() {
  if (!log_file_) {
    log_file_ = fopen(
        (FLAGS_controller_log_dir + FLAGS_controller_log_name).c_str(), "a");
    fprintf(log_file_, "B\n");
  }
}

void ControllerScheduler::FlushLoggingFile() {
  if (log_file_) {
    PCHECK(fflush(log_file_) == 0);
  }
}

}  // namespace canary
