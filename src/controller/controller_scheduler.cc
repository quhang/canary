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
    ControllerSendCommandInterface* send_command_interface,
    LaunchSendCommandInterface* launch_send_command_interface) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  send_command_interface_ = CHECK_NOTNULL(send_command_interface);
  launch_send_command_interface_ = CHECK_NOTNULL(launch_send_command_interface);
}

void ControllerSchedulerBase::ReceiveLaunchCommand(
    LaunchCommandId launch_command_id, struct evbuffer* buffer) {
  event_main_thread_->AddInjectedEvent(
      std::bind(&ControllerSchedulerBase::InternalReceiveLaunchCommand, this,
                launch_command_id, buffer));
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

#define PROCESS_LAUNCH_MESSAGE(TYPE, METHOD)                        \
  case MessageCategory::TYPE: {                                     \
    message::get_message_type<MessageCategory::TYPE>::Type message; \
    message::RemoveControlHeader(buffer);                           \
    message::DeserializeMessage(buffer, &message);                  \
    METHOD(launch_command_id, message);                             \
    break;                                                          \
  }
void ControllerScheduler::InternalReceiveLaunchCommand(
    LaunchCommandId launch_command_id, struct evbuffer* buffer) {
  CHECK_NOTNULL(buffer);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  CHECK(header->category_group == MessageCategoryGroup::LAUNCH_COMMAND);
  switch (header->category) {
    PROCESS_LAUNCH_MESSAGE(LAUNCH_APPLICATION, ProcessLaunchApplication);
    PROCESS_LAUNCH_MESSAGE(PAUSE_APPLICATION, ProcessPauseApplication);
    PROCESS_LAUNCH_MESSAGE(RESUME_APPLICATION, ProcessResumeApplication);
    PROCESS_LAUNCH_MESSAGE(CONTROL_APPLICATION_PRIORITY,
                           ProcessControlApplicationPriority);
    PROCESS_LAUNCH_MESSAGE(REQUEST_APPLICATION_STAT,
                           ProcessRequestApplicationStat);
    PROCESS_LAUNCH_MESSAGE(REQUEST_SHUTDOWN_WORKER,
                           ProcessRequestShutdownWorker);
    default:
      LOG(FATAL) << "Unexpected message type!";
  }  // switch category.
}
#undef PROCESS_LAUNCH_MESSAGE

#define PROCESS_MESSAGE(TYPE, METHOD)                               \
  case MessageCategory::TYPE: {                                     \
    message::get_message_type<MessageCategory::TYPE>::Type message; \
    message::RemoveControlHeader(buffer);                           \
    message::DeserializeMessage(buffer, &message);                  \
    METHOD(message);                                                \
    break;                                                          \
  }
void ControllerScheduler::InternalReceiveCommand(struct evbuffer* buffer) {
  CHECK_NOTNULL(buffer);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  CHECK(header->category_group == MessageCategoryGroup::CONTROLLER_COMMAND);
  switch (header->category) {
    // A worker is ready for receiving a migrated partition.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_MIGRATION_IN_PREPARED,
                    ProcessMigrationInPrepared);
    // A worker already received a migrated partition.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_MIGRATION_IN_DONE,
                    ProcessMigrationInDone);
    // A worker completes a partition.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_PARTITION_DONE, ProcessPartitionDone);
    // A worker responds the status of a partition.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_STATUS_OF_PARTITION,
                    ProcessStatusOfPartition);
    // A worker responds its status.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_STATUS_OF_WORKER, ProcessStatusOfWorker);
    // A worker responds that a barrier has been reached.
    PROCESS_MESSAGE(CONTROLLER_RESPOND_REACH_BARRIER, ProcessReachBarrier);
    default:
      LOG(FATAL) << "Unexpected message type!";
  }  // switch category.
}
#undef PROCESS_MESSAGE

void ControllerScheduler::InternalNotifyWorkerIsDown(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
  auto iter = worker_map_.find(worker_id);
  CHECK(iter != worker_map_.end());
  if (!iter->second.owned_partitions.empty()) {
    LOG(ERROR) << "Worker is down while there are active partitions ("
               << get_value(worker_id) << ")!";
  }
  worker_map_.erase(iter);
}

void ControllerScheduler::InternalNotifyWorkerIsUp(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
  CHECK(worker_map_.find(worker_id) == worker_map_.end());
  // TODO(quhang): worker status report machenism.
  worker_map_[worker_id].num_cores = -1;
}

/*
 * Processes launcher commands.
 */

bool ControllerScheduler::CheckLaunchApplicationMessage(
    const message::LaunchApplication& launch_message,
    message::LaunchApplicationResponse* response) {
  if (launch_message.priority_level < 0) {
    response->succeed = false;
    response->error_message = "Invalid priority level!";
    return response->succeed;
  }
  if (launch_message.fix_num_worker == 0) {
    response->succeed = false;
    response->error_message = "Invalid worker number!";
    return response->succeed;
  }
  if (launch_message.fix_num_worker != -1 &&
      static_cast<int>(worker_map_.size()) < launch_message.fix_num_worker) {
    response->succeed = false;
    response->error_message = "Need more workers for launching!";
    return response->succeed;
  }
  if (launch_message.fix_num_worker == -1 && worker_map_.empty()) {
    response->succeed = false;
    response->error_message = "No worker is launched!";
    return response->succeed;
  }
  response->succeed = true;
  return response->succeed;
}

void ControllerScheduler::ProcessLaunchApplication(
    LaunchCommandId launch_command_id,
    const message::LaunchApplication& launch_message) {
  message::LaunchApplicationResponse response;
  if (!CheckLaunchApplicationMessage(launch_message, &response)) {
    launch_send_command_interface_->SendLaunchResponseCommand(
        launch_command_id,
        message::SerializeMessageWithControlHeader(response));
    return;
  }
  // Assigns an application id.
  const ApplicationId assigned_application_id = (next_application_id_++);
  if (!InitializeApplicationRecord(assigned_application_id, launch_message,
                                   &response)) {
    launch_send_command_interface_->SendLaunchResponseCommand(
        launch_command_id,
        message::SerializeMessageWithControlHeader(response));
    return;
  }
  auto& application_record = application_map_.at(assigned_application_id);
  // TODO(quhang): adds interface.
  // Fills in the partition map.
  DecidePartitionMap(&application_record);
  CHECK(CheckPartitionMapIntegrity(assigned_application_id))
      << "The partition map is not filled in correctly!";
  // Sends the partition map to workers.
  send_command_interface_->AddApplication(
      assigned_application_id,
      new PerApplicationPartitionMap(application_record.per_app_partition_map));
  UpdateApplicationStateBasedOnPartitionMap(assigned_application_id);
  InitializePartitionsOfApplicationOnWorkers(assigned_application_id);
  // Sends response to the launcher.
  response.succeed = true;
  response.application_id = get_value(assigned_application_id);
  launch_send_command_interface_->SendLaunchResponseCommand(
      launch_command_id, message::SerializeMessageWithControlHeader(response));
  LOG(INFO) << "Launched application #" << get_value(assigned_application_id)
            << " (" << launch_message.binary_location.c_str() << ").";
}

void ControllerScheduler::ProcessPauseApplication(
    LaunchCommandId launch_command_id,
    const message::PauseApplication& pause_message) {
  // TODO(quhang): not implemented.
  // First pause, then wait for installing barrier.
}

bool ControllerScheduler::CheckResumeApplicationMessage(
    const message::ResumeApplication& resume_message,
    message::ResumeApplicationResponse* response) {
  if (!CheckValidApplicationId(resume_message, response)) {
    return response->succeed;
  }
  const auto application_id =
      static_cast<ApplicationId>(resume_message.application_id);
  auto& application_record = application_map_.at(application_id);
  if (application_record.application_state !=
      ApplicationRecord::ApplicationState::AT_BARRIER) {
    response->succeed = false;
    response->error_message =
        "The application cannot be resumed because it is not at a barrier!";
    return response->succeed;
  }
  response->succeed = true;
  return response->succeed;
}

void ControllerScheduler::ProcessResumeApplication(
    LaunchCommandId launch_command_id,
    const message::ResumeApplication& resume_message) {
  message::ResumeApplicationResponse response;
  if (!CheckResumeApplicationMessage(resume_message, &response)) {
    launch_send_command_interface_->SendLaunchResponseCommand(
        launch_command_id,
        message::SerializeMessageWithControlHeader(response));
    return;
  }
  const auto application_id =
      static_cast<ApplicationId>(resume_message.application_id);
  auto& application_record = application_map_.at(application_id);
  CHECK(application_record.application_state ==
        ApplicationRecord::ApplicationState::AT_BARRIER);
  application_record.application_state =
      ApplicationRecord::ApplicationState::RUNNING;
  application_record.next_barrier_stage = StageId::INVALID;
  message::WorkerReleaseBarrier release_barrier_command;
  SendCommandToPartitionInApplication(application_id, &release_barrier_command);
  response.succeed = true;
  response.application_id = resume_message.application_id;
  launch_send_command_interface_->SendLaunchResponseCommand(
      launch_command_id, message::SerializeMessageWithControlHeader(response));
}

bool ControllerScheduler::CheckControlApplicationPriorityMessage(
    const message::ControlApplicationPriority& control_message,
    message::ControlApplicationPriorityResponse* response) {
  if (!CheckValidApplicationId(control_message, response)) {
    return response->succeed;
  }
  if (control_message.priority_level < 0) {
    response->succeed = false;
    response->error_message = "Invalid priority level!";
    return response->succeed;
  }
  response->succeed = true;
  return response->succeed;
}

void ControllerScheduler::ProcessControlApplicationPriority(
    LaunchCommandId launch_command_id,
    const message::ControlApplicationPriority& control_message) {
  message::ControlApplicationPriorityResponse response;
  if (!CheckControlApplicationPriorityMessage(control_message, &response)) {
    launch_send_command_interface_->SendLaunchResponseCommand(
        launch_command_id,
        message::SerializeMessageWithControlHeader(response));
    return;
  }
  const auto application_id =
      static_cast<ApplicationId>(control_message.application_id);
  auto& application_record = application_map_.at(application_id);
  application_record.priority_level =
      PriorityLevel(control_message.priority_level);
  message::WorkerChangeApplicationPriority change_priority_command;
  change_priority_command.application_id = application_id;
  change_priority_command.priority_level = application_record.priority_level;
  for (auto& pair : worker_map_) {
    auto& loaded_applications = pair.second.loaded_applications;
    if (loaded_applications.find(application_id) == loaded_applications.end()) {
      continue;
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(change_priority_command));
  }
  response.succeed = true;
  response.application_id = get_value(application_id);
  launch_send_command_interface_->SendLaunchResponseCommand(
      launch_command_id, message::SerializeMessageWithControlHeader(response));
}

bool ControllerScheduler::CheckRequestApplicationStatMessage(
    const message::RequestApplicationStat& request_message,
    message::RequestApplicationStatResponse* response) {
  if (!CheckValidApplicationId(request_message, response)) {
    return response->succeed;
  }
  response->succeed = true;
  return response->succeed;
}

void ControllerScheduler::ProcessRequestApplicationStat(
    LaunchCommandId launch_command_id,
    const message::RequestApplicationStat& request_message) {
  message::RequestApplicationStatResponse response;
  if (!CheckRequestApplicationStatMessage(request_message, &response)) {
    launch_send_command_interface_->SendLaunchResponseCommand(
        launch_command_id,
        message::SerializeMessageWithControlHeader(response));
    return;
  }
  const auto application_id =
      static_cast<ApplicationId>(request_message.application_id);
  auto& application_record = application_map_.at(application_id);
  // Builds a new report id.
  ++application_record.report_id;
  application_record.report_partition_set.clear();
  application_record.report_command_list.push_back(launch_command_id);
  message::WorkerReportStatusOfPartitions report_partitions_command;
  SendCommandToPartitionInApplication(application_id,
                                      &report_partitions_command);
  // The stats will be sent later.
}

void ControllerScheduler::ProcessRequestShutdownWorker(
    LaunchCommandId launch_command_id,
    const message::RequestShutdownWorker& request_message) {
  // TODO(quhang): implement.
}

/*
 * Processes messages received from workers.
 */

void ControllerScheduler::ProcessMigrationInPrepared(
    const message::ControllerRespondMigrationInPrepared& respond_message) {
  // TODO(quhang): implement.
}

void ControllerScheduler::ProcessMigrationInDone(
    const message::ControllerRespondMigrationInDone& respond_message) {
  // TODO(quhang): implement.
}

void ControllerScheduler::ProcessPartitionDone(
    const message::ControllerRespondPartitionDone& respond_message) {
  const auto& running_stats = respond_message.running_stats;
  UpdateRunningStats(respond_message.from_worker_id,
                     respond_message.application_id,
                     respond_message.variable_group_id,
                     respond_message.partition_id, running_stats);
  CHECK(running_stats.earliest_unfinished_stage_id == StageId::INVALID &&
        running_stats.last_finished_stage_id == StageId::COMPLETE);
  auto& application_record =
      application_map_.at(respond_message.application_id);
  if (++application_record.complete_partition ==
      application_record.total_partition) {
    CleanUpApplication(respond_message.application_id);
  }
}

void ControllerScheduler::ProcessStatusOfPartition(
    const message::ControllerRespondStatusOfPartition& respond_message) {
  UpdateRunningStats(
      respond_message.from_worker_id, respond_message.application_id,
      respond_message.variable_group_id, respond_message.partition_id,
      respond_message.running_stats);
}

void ControllerScheduler::ProcessStatusOfWorker(
    const message::ControllerRespondStatusOfWorker& respond_message) {
  const auto from_worker_id = respond_message.from_worker_id;
  // Updates worker map.
  if (worker_map_.find(from_worker_id) != worker_map_.end()) {
    worker_map_[from_worker_id].num_cores = respond_message.num_cores;
  }
}

void ControllerScheduler::ProcessReachBarrier(
    const message::ControllerRespondReachBarrier& respond_message) {
  UpdateRunningStats(
      respond_message.from_worker_id, respond_message.application_id,
      respond_message.variable_group_id, respond_message.partition_id,
      respond_message.running_stats);
  auto& application_record =
      application_map_.at(respond_message.application_id);
  if (++application_record.blocked_partition ==
      application_record.total_partition) {
    // Update application state if every partition reaches the barrier.
    application_record.blocked_partition = 0;
    CHECK(application_record.application_state ==
          ApplicationRecord::ApplicationState::RUNNING);
    application_record.application_state =
        ApplicationRecord::ApplicationState::AT_BARRIER;
    LOG(INFO) << "Application #" << get_value(respond_message.application_id)
              << " reached barrier stage.";
  }
}

/*
 * Handling the state of an application.
 */

bool ControllerScheduler::InitializeApplicationRecord(
    ApplicationId application_id,
    const message::LaunchApplication& launch_message,
    message::LaunchApplicationResponse* response) {
  CHECK(application_map_.find(application_id) == application_map_.end());
  auto& application_record = application_map_[application_id];
  // Loads the application.
  application_record.loaded_application = CanaryApplication::LoadApplication(
      launch_message.binary_location, launch_message.application_parameter,
      &application_record.loading_handle);
  if (!application_record.loaded_application) {
    // Application loading failed.
    application_map_.erase(application_id);
    response->succeed = false;
    response->error_message = "Application binary is not found!";
    return response->succeed;
  }
  application_record.variable_group_info_map =
      application_record.loaded_application->get_variable_group_info_map();
  // Fills in launching information.
  application_record.binary_location = launch_message.binary_location;
  application_record.application_parameter =
      launch_message.application_parameter;
  if (launch_message.first_barrier_stage >= 0) {
    application_record.next_barrier_stage =
        StageId(launch_message.first_barrier_stage);
  } else {
    application_record.next_barrier_stage = StageId::INVALID;
  }
  CHECK_GE(launch_message.priority_level, 0);
  application_record.priority_level =
      PriorityLevel(launch_message.priority_level);
  // Initializes the application running states.
  application_record.application_state =
      ApplicationRecord::ApplicationState::RUNNING;
  application_record.total_partition = 0;
  application_record.complete_partition = 0;
  application_record.blocked_partition = 0;
  for (const auto& pair : *application_record.variable_group_info_map) {
    application_record.total_partition += pair.second.parallelism;
  }
  // Logs the application.
  LogApplication(application_id, launch_message.binary_location,
                 launch_message.application_parameter);
  response->succeed = true;
  return response->succeed;
}

void ControllerScheduler::UpdateRunningStats(
    WorkerId worker_id, ApplicationId application_id,
    VariableGroupId variable_group_id, PartitionId partition_id,
    const message::RunningStats& running_stats) {
  LogRunningStats(worker_id, application_id, variable_group_id, partition_id,
                  running_stats);
  auto& application_record = application_map_.at(application_id);
  const auto& cycle_stats = running_stats.cycle_stats;
  for (const auto& pair : cycle_stats) {
    application_record.total_used_cycles += pair.second.second;
  }
  application_record.report_partition_set.insert(
      FullPartitionId{application_id, variable_group_id, partition_id});
  if (static_cast<int>(application_record.report_partition_set.size()) ==
      application_record.total_partition) {
    message::RequestApplicationStatResponse response;
    response.application_id = get_value(application_id);
    response.succeed = true;
    response.cycles = application_record.total_used_cycles;
    for (auto launch_command_id : application_record.report_command_list) {
      launch_send_command_interface_->SendLaunchResponseCommand(
          launch_command_id,
          message::SerializeMessageWithControlHeader(response));
    }
    application_record.report_command_list.clear();
  }
}

void ControllerScheduler::CleanUpApplication(ApplicationId application_id) {
  LOG(INFO) << "Completed application #" << get_value(application_id) << ".";
  ApplicationRecord& application_record = application_map_.at(application_id);
  application_record.application_state =
      ApplicationRecord::ApplicationState::COMPLETE;
  FlushLoggingFile();
  message::WorkerUnloadPartitions unload_partitions_command;
  // Unloads partitions.
  SendCommandToPartitionInApplication(application_id,
                                      &unload_partitions_command);
  // Rewinds states in the other two maps.
  for (auto& pair : worker_map_) {
    auto owned_partitions = pair.second.owned_partitions;
    for (auto full_partition_id : owned_partitions[application_id]) {
      partition_record_map_.erase(full_partition_id);
    }
    owned_partitions.erase(application_id);
  }
  // Unloads the application.
  message::WorkerUnloadApplication unload_application_command;
  unload_application_command.application_id = application_id;
  for (auto& pair : worker_map_) {
    auto& loaded_applications = pair.second.loaded_applications;
    if (loaded_applications.find(application_id) == loaded_applications.end()) {
      continue;
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(unload_application_command));
    loaded_applications.erase(application_id);
  }
  // Unload application binary.
  CanaryApplication::UnloadApplication(application_record.loading_handle,
                                       application_record.loaded_application);
  // Unload partition map.
  send_command_interface_->DropApplication(application_id);
  // Erases the application record.
  application_map_.erase(application_id);
}

/*
 * Launching an application based on the partition map.
 */
bool ControllerScheduler::CheckPartitionMapIntegrity(
    ApplicationId application_id) {
  const auto& application_record = application_map_.at(application_id);
  const auto& per_app_partition_map = application_record.per_app_partition_map;
  const auto& variable_group_info_map =
      *application_record.variable_group_info_map;
  if (per_app_partition_map.QueryNumVariableGroup() !=
      static_cast<int>(variable_group_info_map.size())) {
    return false;
  }
  for (const auto& pair : variable_group_info_map) {
    if (per_app_partition_map.QueryPartitioning(pair.first) !=
        static_cast<int>(pair.second.parallelism)) {
      return false;
    }
    for (int index = 0; index < pair.second.parallelism; ++index) {
      auto worker_id = per_app_partition_map.QueryWorkerId(
          pair.first, static_cast<PartitionId>(index));
      if (worker_map_.find(worker_id) == worker_map_.end()) {
        return false;
      }
    }
  }
  return true;
}

void ControllerScheduler::UpdateApplicationStateBasedOnPartitionMap(
    ApplicationId application_id) {
  const PerApplicationPartitionMap& per_app_partition_map =
      application_map_.at(application_id).per_app_partition_map;
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
      // Inserts the partition record.
      partition_record_map_[full_partition_id].partition_state =
          PartitionRecord::PartitionState::INVALID;
    }
  }
}

void ControllerScheduler::InitializePartitionsOfApplicationOnWorkers(
    ApplicationId application_id) {
  const auto& application_record = application_map_.at(application_id);
  // Loads the application.
  message::WorkerLoadApplication load_application_command;
  load_application_command.application_id = application_id;
  load_application_command.binary_location = application_record.binary_location;
  load_application_command.application_parameter =
      application_record.application_parameter;
  load_application_command.first_barrier_stage =
      application_record.next_barrier_stage;
  load_application_command.priority_level = application_record.priority_level;
  for (auto& pair : worker_map_) {
    if (pair.second.owned_partitions.find(application_id) ==
        pair.second.owned_partitions.end()) {
      continue;
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(load_application_command));
    pair.second.loaded_applications.insert(application_id);
  }
  // Loads partitions.
  message::WorkerLoadPartitions load_partitions_command;
  SendCommandToPartitionInApplication(application_id, &load_partitions_command);
}

template <typename InputMessage, typename OutputMessage>
bool ControllerScheduler::CheckValidApplicationId(
    const InputMessage& input_message, OutputMessage* output_message) {
  if (input_message.application_id < 0) {
    output_message->succeed = false;
    output_message->error_message = "Invalid application id!";
    return output_message->succeed;
  }
  const auto application_id =
      static_cast<ApplicationId>(input_message.application_id);
  if (application_map_.find(application_id) == application_map_.end()) {
    output_message->succeed = false;
    output_message->error_message =
        "The application id does not specify a running application!";
    return output_message->succeed;
  }
  output_message->succeed = true;
  return output_message->succeed;
}

template <typename T>
void ControllerScheduler::SendCommandToPartitionInApplication(
    ApplicationId application_id, T* template_command) {
  CHECK_NOTNULL(template_command);
  template_command->application_id = application_id;
  for (const auto& pair : worker_map_) {
    const auto& owned_partitions = pair.second.owned_partitions;
    if (owned_partitions.find(application_id) == owned_partitions.end()) {
      continue;
    }
    if (owned_partitions.at(application_id).empty()) {
      continue;
    }
    template_command->partition_list.clear();
    for (const auto& full_partition_id : owned_partitions.at(application_id)) {
      template_command->partition_list.emplace_back(
          full_partition_id.variable_group_id, full_partition_id.partition_id);
    }
    send_command_interface_->SendCommandToWorker(
        pair.first,
        message::SerializeMessageWithControlHeader(*template_command));
  }
}

void ControllerScheduler::InitializeLoggingFile() {
  if (!log_file_) {
    log_file_ = fopen(
        (FLAGS_controller_log_dir + FLAGS_controller_log_name).c_str(), "a");
    fprintf(log_file_, "B\n");
    PCHECK(fflush(log_file_) == 0);
  }
}

void ControllerScheduler::LogApplication(
    ApplicationId application_id, const std::string& binary_location,
    const std::string& application_parameter) {
  InitializeLoggingFile();
  fprintf(log_file_, "L %d %s %s\n", get_value(application_id),
          binary_location.c_str(),
          TransformString(application_parameter).c_str());
  FlushLoggingFile();
}

std::string ControllerScheduler::TransformString(const std::string& input) {
  std::string result;
  std::remove_copy_if(
      input.begin(), input.end(), std::back_inserter(result),
      [](auto c) { return c == ' ' || c == '\n' || c == '\t'; });
  return std::move(result);
}

void ControllerScheduler::LogRunningStats(
    WorkerId worker_id, ApplicationId application_id,
    VariableGroupId variable_group_id, PartitionId partition_id,
    const message::RunningStats& running_stats) {
  InitializeLoggingFile();
  fprintf(log_file_, "P %d %d %d W %d\n", get_value(application_id),
          get_value(variable_group_id), get_value(partition_id),
          get_value(worker_id));
  const auto& timestamp_stats = running_stats.timestamp_stats;
  // TODO(quhang): min_timestamp is not correct.
  const auto min_timestamp = timestamp_stats.cbegin()->second.second;
  for (const auto& pair : timestamp_stats) {
    fprintf(log_file_, "T %d %d %f\n", get_value(pair.first),
            get_value(pair.second.first),
            (pair.second.second - min_timestamp) * 1.e3);
  }
  const auto& cycle_stats = running_stats.cycle_stats;
  for (const auto& pair : cycle_stats) {
    fprintf(log_file_, "C %d %d %f\n", get_value(pair.first),
            get_value(pair.second.first), pair.second.second * 1.e3);
  }
  FlushLoggingFile();
}

void ControllerScheduler::FlushLoggingFile() {
  if (log_file_) {
    PCHECK(fflush(log_file_) == 0);
  }
}

void ControllerScheduler::DecidePartitionMap(
    ApplicationRecord* application_record) {
  auto& per_app_partition_map = application_record->per_app_partition_map;
  const auto& variable_group_info_map =
      *application_record->variable_group_info_map;
  per_app_partition_map.SetNumVariableGroup(variable_group_info_map.size());
  for (const auto& pair : variable_group_info_map) {
    per_app_partition_map.SetPartitioning(pair.first, pair.second.parallelism);
    // Birdshot randomized placement.
    // Assigns partitions to workers randomly, and the number of assigned
    // partitions is based on the number of worker cores.
    std::vector<WorkerId> worker_id_list;
    GetWorkerAssignment(pair.second.parallelism, &worker_id_list);
    std::random_shuffle(worker_id_list.begin(), worker_id_list.end());
    auto iter = worker_id_list.begin();
    for (int index = 0; index < pair.second.parallelism; ++index) {
      per_app_partition_map.SetWorkerId(
          pair.first, static_cast<PartitionId>(index), *(iter++));
    }
  }
}

void ControllerScheduler::GetWorkerAssignment(
    int num_slot, std::vector<WorkerId>* assignment) {
  assignment->clear();
  assignment->resize(num_slot);
  for (auto& worker_id : *assignment) {
    auto iter = worker_map_.find(last_assigned_worker_id_);
    if (iter == worker_map_.end()) {
      // Assigns a partition to the next worker.
      worker_id = NextAssignWorkerId();
      last_assigned_partitions_ = 1;
      continue;
    }
    if (iter->second.num_cores == -1) {
      LOG(WARNING) << "The number of cores for a worker is unknown!";
      CHECK_EQ(last_assigned_partitions_, 1);
      worker_id = NextAssignWorkerId();
      last_assigned_partitions_ = 1;
    } else {
      if (last_assigned_partitions_ < iter->second.num_cores) {
        worker_id = last_assigned_worker_id_;
        ++last_assigned_partitions_;
      } else {
        worker_id = NextAssignWorkerId();
        last_assigned_partitions_ = 1;
      }
    }
  }
}

WorkerId ControllerScheduler::NextAssignWorkerId() {
  CHECK(!worker_map_.empty());
  auto iter = worker_map_.upper_bound(last_assigned_worker_id_);
  if (iter == worker_map_.end()) {
    last_assigned_worker_id_ = worker_map_.begin()->first;
  } else {
    last_assigned_worker_id_ = iter->first;
  }
  return last_assigned_worker_id_;
}

}  // namespace canary
