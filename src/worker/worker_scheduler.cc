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
 * @file src/worker/worker_scheduler.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerScheduler.
 */

#include "worker/worker_scheduler.h"

#include <utility>

namespace canary {

WorkerSchedulerBase::WorkerSchedulerBase() {
  PCHECK(pthread_mutex_init(&scheduling_lock_, nullptr) == 0);
  PCHECK(pthread_cond_init(&scheduling_cond_, nullptr) == 0);
}

WorkerSchedulerBase::~WorkerSchedulerBase() {
  pthread_mutex_destroy(&scheduling_lock_);
  pthread_cond_destroy(&scheduling_cond_);
}

void WorkerSchedulerBase::Initialize(
    WorkerSendCommandInterface* send_command_interface,
    WorkerSendDataInterface* send_data_interface) {
  send_command_interface_ = CHECK_NOTNULL(send_command_interface);
  send_data_interface_ = CHECK_NOTNULL(send_data_interface);
  // Caution: reads global variable.
  num_cores_ = FLAGS_worker_threads;
}

bool WorkerSchedulerBase::ReceiveRoutedData(ApplicationId application_id,
                                            VariableGroupId variable_group_id,
                                            PartitionId partition_id,
                                            StageId stage_id,
                                            struct evbuffer* buffer) {
  CHECK(is_ready_);
  auto iter = thread_map_.find(
      FullPartitionId{application_id, variable_group_id, partition_id});
  if (iter == thread_map_.end()) {
    // The data is rejected.
    return false;
  }
  WorkerLightThreadContext* thread_context = iter->second;
  thread_context->DeliverMessage(stage_id, buffer);
  return true;
}

void WorkerSchedulerBase::ReceiveDirectData(struct evbuffer* buffer) {
  // MIGRATION, step four: receives migrated data.
  CHECK(is_ready_);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  CHECK(header->category_group ==
        message::MessageCategoryGroup::APPLICATION_DATA_DIRECT);
  CHECK(header->category == message::MessageCategory::DIRECT_DATA_MIGRATE);
  message::DirectDataMigrate direct_data_migrate;
  message::RemoveControlHeader(buffer);
  message::DeserializeMessage(buffer, &direct_data_migrate);
  FullPartitionId full_partition_id{direct_data_migrate.application_id,
                                    direct_data_migrate.variable_group_id,
                                    direct_data_migrate.partition_id};
  auto iter = thread_map_.find(full_partition_id);
  CHECK(iter != thread_map_.end());
  iter->second->DeliverMessage(StageId::MIGRATE_IN,
                               direct_data_migrate.raw_buffer.buffer);
}

#define PROCESS_MESSAGE(TYPE, METHOD)                               \
  case MessageCategory::TYPE: {                                     \
    message::get_message_type<MessageCategory::TYPE>::Type message; \
    message::RemoveControlHeader(buffer);                           \
    message::DeserializeMessage(buffer, &message);                  \
    METHOD(message);                                                \
    break;                                                          \
  }
void WorkerSchedulerBase::ReceiveCommandFromController(
    struct evbuffer* buffer) {
  CHECK(is_ready_);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  using message::ControlHeader;
  CHECK(header->category_group == MessageCategoryGroup::WORKER_COMMAND);
  switch (header->category) {
    PROCESS_MESSAGE(WORKER_LOAD_APPLICATION, ProcessLoadApplication);
    PROCESS_MESSAGE(WORKER_UNLOAD_APPLICATION, ProcessUnloadApplication);
    PROCESS_MESSAGE(WORKER_LOAD_PARTITIONS, ProcessLoadPartitions);
    PROCESS_MESSAGE(WORKER_UNLOAD_PARTITIONS, ProcessUnloadPartitions);
    PROCESS_MESSAGE(WORKER_MIGRATE_IN_PARTITIONS, ProcessMigrateInPartitions);
    PROCESS_MESSAGE(WORKER_MIGRATE_OUT_PARTITIONS, ProcessMigrateOutPartitions);
    PROCESS_MESSAGE(WORKER_REPORT_STATUS_OF_PARTITIONS,
                    ProcessReportStatusOfPartitions);
    PROCESS_MESSAGE(WORKER_CHANGE_APPLICATION_PRIORITY,
                    ProcessChangeApplicationPriority);
    PROCESS_MESSAGE(WORKER_PAUSE_EXECUTION, ProcessPauseExecution);
    PROCESS_MESSAGE(WORKER_INSTALL_BARRIER, ProcessInstallBarrier);
    PROCESS_MESSAGE(WORKER_RELEASE_BARRIER, ProcessReleaseBarrier);
    default:
      LOG(FATAL) << "Unexpected message category!";
  }
}
#undef PROCESS_MESSAGE

void WorkerSchedulerBase::AssignWorkerId(WorkerId worker_id) {
  CHECK(!is_ready_);
  is_ready_ = true;
  self_worker_id_ = worker_id;
  // Responds with the worker status.
  message::ControllerRespondStatusOfWorker respond_status;
  respond_status.from_worker_id = self_worker_id_;
  respond_status.num_cores = num_cores_;
  send_command_interface_->SendCommandToController(
      message::SerializeMessageWithControlHeader(respond_status));
  // Starts execution.
  StartExecution();
}

/*
 * Processes commands received from the controller.
 */
void WorkerSchedulerBase::ProcessLoadApplication(
    const message::WorkerLoadApplication& worker_command) {
  const auto load_application_id = worker_command.application_id;
  CHECK(application_record_map_.find(load_application_id) ==
        application_record_map_.end())
      << "Cannot load an application twice!";
  auto& application_record = application_record_map_[load_application_id];
  application_record.application_id = load_application_id;
  application_record.binary_location = worker_command.binary_location;
  application_record.application_parameter =
      worker_command.application_parameter;
  application_record.first_barrier_stage = worker_command.first_barrier_stage;
  application_record.local_partitions = 0;
  // Loads application binary.
  LoadApplicationBinary(&application_record);
  // Sets the application priority level.
  SetApplicationPriorityLevel(load_application_id,
                              worker_command.priority_level);
}

void WorkerSchedulerBase::ProcessUnloadApplication(
    const message::WorkerUnloadApplication& worker_command) {
  const auto unload_application_id = worker_command.application_id;
  auto iter = application_record_map_.find(unload_application_id);
  CHECK(iter != application_record_map_.end()) << "No application to unload!";
  CHECK_EQ(iter->second.local_partitions, 0)
      << "Cannot unload an application when a partition is running!";
  // Unloads application binary.
  UnloadApplicationBinary(&iter->second);
  application_record_map_.erase(iter);
}

void WorkerSchedulerBase::ProcessLoadPartitions(
    const message::WorkerLoadPartitions& worker_command) {
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.partition_list) {
    VLOG(1) << "Load partition: " << get_value(application_id) << "/"
            << get_value(pair.first) << "/" << get_value(pair.second);
    FullPartitionId full_partition_id{application_id, pair.first, pair.second};
    // Constructs thread context.
    ConstructThreadContext(full_partition_id);
  }
  // Initializes and installs the first barrier.
  const auto first_barrier_stage =
      application_record_map_.at(application_id).first_barrier_stage;
  DeliverCommandToEachThread(
      worker_command, StageId::INIT, [first_barrier_stage]() {
        return internal_message::to_buffer(
            internal_message::InitCommand{first_barrier_stage});
      });
  // Refreshes the routing such that pending messages for this partition can
  // be delivered.
  send_data_interface_->RefreshRouting();
}

void WorkerSchedulerBase::ProcessUnloadPartitions(
    const message::WorkerUnloadPartitions& worker_command) {
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.partition_list) {
    FullPartitionId full_partition_id{application_id, pair.first, pair.second};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end());
    // Detaches the thread context, and kills it.
    auto detached_context = DetachThreadContext(full_partition_id);
    KillThreadContext(detached_context);
  }
}

void WorkerSchedulerBase::ProcessMigrateInPartitions(
    const message::WorkerMigrateInPartitions& worker_command) {
  // MIGRATION, step one: construct the context, and respond to the controller.
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.partition_list) {
    FullPartitionId full_partition_id{application_id, pair.first, pair.second};
    ConstructThreadContext(full_partition_id);
    message::ControllerRespondMigrationInDone response;
    response.from_worker_id = self_worker_id_;
    response.application_id = full_partition_id.application_id;
    response.variable_group_id = full_partition_id.variable_group_id;
    response.partition_id = full_partition_id.partition_id;
    send_command_interface_->SendCommandToController(
        message::SerializeMessageWithControlHeader(response));
  }
}

void WorkerSchedulerBase::WorkerSchedulerBase::ProcessMigrateOutPartitions(
    const message::WorkerMigrateOutPartitions& worker_command) {
  // MIGRATION, step two: tell the context to migrate out.
  const auto application_id = worker_command.application_id;
  for (const auto& partition_migrate_record :
       worker_command.migrate_out_partitions) {
    FullPartitionId full_partition_id{
        application_id, partition_migrate_record.variable_group_id,
        partition_migrate_record.partition_id};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end());
    iter->second->DeliverMessage(
        StageId::MIGRATE_OUT,
        internal_message::to_buffer(internal_message::MigrateOutCommand{
            partition_migrate_record.to_worker_id}));
    // Detach thread context.
    DetachThreadContext(full_partition_id);
  }
}

void WorkerSchedulerBase::ProcessReportStatusOfPartitions(
    const message::WorkerReportStatusOfPartitions& worker_command) {
  for (const auto& pair : worker_command.partition_list) {
    FullPartitionId full_partition_id{worker_command.application_id, pair.first,
                                      pair.second};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end()) << "No such thread to report status.";
    RequestReportOfThreadContext(iter->second);
  }
}

void WorkerSchedulerBase::ProcessPauseExecution(
    const message::WorkerPauseExecution& worker_command) {
  // TODO(quhang): not implemented.
}

void WorkerSchedulerBase::ProcessInstallBarrier(
    const message::WorkerInstallBarrier& worker_command) {
  // TODO(quhang): not implemented.
}

void WorkerSchedulerBase::ProcessReleaseBarrier(
    const message::WorkerReleaseBarrier& worker_command) {
  DeliverCommandToEachThread(worker_command, StageId::RELEASE_BARRIER,
                             []() { return nullptr; });
}

void WorkerSchedulerBase::ProcessChangeApplicationPriority(
    const message::WorkerChangeApplicationPriority& worker_command) {
  SetApplicationPriorityLevel(worker_command.application_id,
                              worker_command.priority_level);
}

template <typename T>
void WorkerSchedulerBase::DeliverCommandToEachThread(
    const T& command_from_controller, StageId command_stage_id,
    std::function<struct evbuffer*()> generator) {
  for (const auto& pair : command_from_controller.partition_list) {
    FullPartitionId full_partition_id{command_from_controller.application_id,
                                      pair.first, pair.second};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end())
        << "Failed to deliver a command to a partition";
    iter->second->DeliverMessage(command_stage_id, generator());
  }
}

/*
 * Schedules the execution of thread contexts.
 */
void WorkerSchedulerBase::RequestReportOfThreadContext(
    WorkerLightThreadContext* thread_context) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  CHECK(!thread_context->is_killed_);
  thread_context->need_report_ = true;
  if (!thread_context->is_running_) {
    request_report_set_.insert(thread_context);
  }
  PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
  pthread_mutex_unlock(&scheduling_lock_);
}

void WorkerSchedulerBase::ActivateThreadContext(
    WorkerLightThreadContext* thread_context) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  CHECK(!thread_context->is_killed_);
  thread_context->need_process_ = true;
  if (!thread_context->is_running_) {
    AddToPriorityQueue(thread_context);
  }
  PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
  pthread_mutex_unlock(&scheduling_lock_);
}

void WorkerSchedulerBase::KillThreadContext(
    WorkerLightThreadContext* thread_context) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  CHECK(!thread_context->is_killed_);
  thread_context->is_killed_ = true;
  if (!thread_context->is_running_) {
    AddToPriorityQueue(thread_context);
  }
  PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
  pthread_mutex_unlock(&scheduling_lock_);
}

void WorkerSchedulerBase::StartExecution() {
  // Reads global variable: the number of worker threads.
  thread_handle_list_.resize(num_cores_);
  sched_param param{0};
  for (auto& handle : thread_handle_list_) {
    PCHECK(pthread_create(&handle, nullptr,
                          &WorkerSchedulerBase::ExecutionRoutine, this) == 0);
    // Set worker thread priority lower than SCHED_OTHER.
    PCHECK(pthread_setschedparam(handle, SCHED_BATCH, &param) == 0);
  }
}

void* WorkerSchedulerBase::ExecutionRoutine(void* arg) {
  auto scheduler = reinterpret_cast<WorkerSchedulerBase*>(arg);
  scheduler->InternalExecutionRoutine();
  LOG(WARNING) << "Execution thread exits.";
  return nullptr;
}

void WorkerSchedulerBase::InternalExecutionRoutine() {
  ActionType action_type = ActionType::NONE;
  WorkerLightThreadContext* held_context = nullptr;
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  while (true) {
    if (held_context) {
      held_context->is_running_ = false;
      if (held_context->need_report_) {
        request_report_set_.insert(held_context);
      } else if (held_context->need_process_) {
        AddToPriorityQueue(held_context);
      } else if (held_context->is_killed_) {
        pthread_mutex_unlock(&scheduling_lock_);
        held_context->Finalize();
        UnloadPartition(held_context);
        PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
      }
    }
    std::tie(action_type, held_context) = GetNextAction();
    while (action_type == ActionType::NONE) {
      pthread_cond_wait(&scheduling_cond_, &scheduling_lock_);
      std::tie(action_type, held_context) = GetNextAction();
    }
    switch (action_type) {
      case ActionType::REPORT:
        held_context->is_running_ = true;
        held_context->need_report_ = false;
        pthread_mutex_unlock(&scheduling_lock_);
        held_context->Report();
        PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
        break;
      case ActionType::RUN:
        held_context->is_running_ = true;
        held_context->need_process_ = false;
        pthread_mutex_unlock(&scheduling_lock_);
        held_context->Run();
        PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
        break;
      default:
        LOG(FATAL) << "Internal error!.";
    }
  }
}

std::pair<WorkerSchedulerBase::ActionType, WorkerLightThreadContext*>
WorkerSchedulerBase::GetNextAction() {
  if (!request_report_set_.empty()) {
    auto iter = request_report_set_.begin();
    WorkerLightThreadContext* result = *iter;
    request_report_set_.erase(iter);
    return std::make_pair(ActionType::REPORT, result);
  }
  auto queue_iter = priority_queue_.begin();
  while (queue_iter != priority_queue_.end() && queue_iter->second.empty()) {
    queue_iter = priority_queue_.erase(queue_iter);
  }
  if (queue_iter != priority_queue_.end()) {
    CHECK(!queue_iter->second.empty());
    WorkerLightThreadContext* result = queue_iter->second.front();
    queue_iter->second.pop_front();
    result->is_in_priority_queue_ = false;
    return std::make_pair(ActionType::RUN, result);
  } else {
    return std::make_pair(ActionType::NONE, nullptr);
  }
}

void WorkerSchedulerBase::SetApplicationPriorityLevel(
    ApplicationId application_id, PriorityLevel priority_level) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  // Rearranges the priority queue if a priority level is changed.
  const bool rearrange_priority_queue =
      (application_priority_level_map_.find(application_id) !=
       application_priority_level_map_.end());
  application_priority_level_map_[application_id] = priority_level;
  if (rearrange_priority_queue) {
    RearrangePriorityQueue();
  }
  pthread_mutex_unlock(&scheduling_lock_);
}

void WorkerSchedulerBase::RearrangePriorityQueue() {
  decltype(priority_queue_) clone;
  clone.swap(priority_queue_);
  for (auto& pair : clone)
    for (auto& thread_context : pair.second) {
      const auto application_id = thread_context->get_application_id();
      const auto priority_level =
          application_priority_level_map_[application_id];
      priority_queue_[priority_level].push_back(thread_context);
    }
}

void WorkerSchedulerBase::AddToPriorityQueue(
    WorkerLightThreadContext* thread_context) {
  if (thread_context->is_in_priority_queue_) {
    return;
  }
  thread_context->is_in_priority_queue_ = true;
  const auto application_id = thread_context->get_application_id();
  const auto priority_level = application_priority_level_map_[application_id];
  priority_queue_[priority_level].push_back(thread_context);
}

void WorkerSchedulerBase::ConstructThreadContext(
    const FullPartitionId& full_partition_id) {
  const auto application_id = full_partition_id.application_id;
  WorkerLightThreadContext* thread_context = LoadPartition(full_partition_id);
  thread_context->worker_id_ = self_worker_id_;
  thread_context->application_id_ = application_id;
  thread_context->variable_group_id_ = full_partition_id.variable_group_id;
  thread_context->partition_id_ = full_partition_id.partition_id;
  thread_context->send_command_interface_ = send_command_interface_;
  thread_context->send_data_interface_ = send_data_interface_;
  // Sets the application.
  thread_context->canary_application_ =
      application_record_map_.at(application_id).loaded_application;
  thread_context->worker_scheduler_ = this;
  // Initializes the thread context.
  thread_context->Initialize();
  // Registers the thread.
  CHECK(thread_map_.find(full_partition_id) == thread_map_.end());
  thread_map_[full_partition_id] = thread_context;
  ++application_record_map_.at(application_id).local_partitions;
}

WorkerLightThreadContext* WorkerSchedulerBase::DetachThreadContext(
    const FullPartitionId& full_partition_id) {
  WorkerLightThreadContext* result = nullptr;
  auto iter = thread_map_.find(full_partition_id);
  CHECK(iter != thread_map_.end());
  result = iter->second;
  thread_map_.erase(iter);
  --application_record_map_.at(full_partition_id.application_id)
        .local_partitions;
  return result;
}

void WorkerScheduler::LoadApplicationBinary(
    ApplicationRecord* application_record) {
  CHECK_NOTNULL(application_record);
  VLOG(1) << "Load application banary: " << application_record->binary_location;
  application_record->loaded_application = CanaryApplication::LoadApplication(
      application_record->binary_location,
      application_record->application_parameter,
      &application_record->loading_handle);
  CHECK_NOTNULL(application_record->loaded_application);
}

void WorkerScheduler::UnloadApplicationBinary(
    ApplicationRecord* application_record) {
  VLOG(1) << "Unload application binary.";
  CanaryApplication::UnloadApplication(application_record->loading_handle,
                                       application_record->loaded_application);
}

WorkerLightThreadContext* WorkerScheduler::LoadPartition(FullPartitionId) {
  return new WorkerExecutionContext();
}

void WorkerScheduler::UnloadPartition(
    WorkerLightThreadContext* thread_context) {
  delete thread_context;
}

}  // namespace canary
