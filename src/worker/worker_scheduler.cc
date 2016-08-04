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
  CHECK(is_ready_);
  CHECK_NOTNULL(buffer);
  LOG(FATAL) << "Not implemented.";
  // TODO(quhang): fill in.
  // Migrated partitions.
  // File system traffic.
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
    PROCESS_MESSAGE(WORKER_RELEASE_BARRIER, ProcessReleaseBarrier);
    PROCESS_MESSAGE(WORKER_CHANGE_APPLICATION_PRIORITY,
                    ProcessChangeApplicationPriority);
    default:
      LOG(FATAL) << "Unexpected message category!";
  }
}
#undef PROCESS_MESSAGE

void WorkerSchedulerBase::AssignWorkerId(WorkerId worker_id) {
  is_ready_ = true;
  self_worker_id_ = worker_id;
  message::ControllerRespondStatusOfWorker respond_status;
  respond_status.from_worker_id = self_worker_id_;
  respond_status.num_cores = num_cores_;
  send_command_interface_->SendCommandToController(
      message::SerializeMessageWithControlHeader(respond_status));
  StartExecution();
}

void WorkerSchedulerBase::ProcessLoadApplication(
    const message::WorkerLoadApplication& worker_command) {
  const auto launch_application_id = worker_command.application_id;
  CHECK(application_record_map_.find(launch_application_id) ==
        application_record_map_.end());
  auto& application_record = application_record_map_[launch_application_id];
  application_record.application_id = launch_application_id;
  application_record.binary_location = worker_command.binary_location;
  application_record.application_parameter =
      worker_command.application_parameter;
  application_record.first_barrier_stage = worker_command.first_barrier_stage;
  application_record.local_partitions = 0;
  LoadApplicationBinary(&application_record);
  SetApplicationPriorityLevel(launch_application_id,
                              worker_command.priority_level);
}

void WorkerSchedulerBase::ProcessUnloadApplication(
    const message::WorkerUnloadApplication& worker_command) {
  const auto remove_application_id = worker_command.application_id;
  auto iter = application_record_map_.find(remove_application_id);
  CHECK(iter != application_record_map_.end());
  CHECK_EQ(iter->second.local_partitions, 0)
      << "Cannot unload an application when a partition is running.";
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
    ConstructThreadContext(full_partition_id);
  }
  const auto first_barrier_stage =
      application_record_map_.at(application_id).first_barrier_stage;
  DeliverCommandToEachThread(
      worker_command, StageId::INIT, [first_barrier_stage]() {
        return internal_message::to_buffer(
            internal_message::InitCommand{first_barrier_stage});
      });
  // Refreshes the routing such that pending messages for this partition can
  // be deliverd.
  send_data_interface_->RefreshRouting();
}

void WorkerSchedulerBase::ProcessUnloadPartitions(
    const message::WorkerUnloadPartitions& worker_command) {
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.partition_list) {
    FullPartitionId full_partition_id{application_id, pair.first, pair.second};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end());
    iter->second->Finalize();
    DestructThreadContext(full_partition_id);
  }
}

void WorkerSchedulerBase::ProcessMigrateInPartitions(
    const message::WorkerMigrateInPartitions& worker_command) {
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.partition_list) {
    FullPartitionId full_partition_id{application_id, pair.first, pair.second};
    ConstructThreadContext(full_partition_id);
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end());
    iter->second->DeliverMessage(StageId::MIGRATE_IN, nullptr);
  }
}
void WorkerSchedulerBase::WorkerSchedulerBase::ProcessMigrateOutPartitions(
    const message::WorkerMigrateOutPartitions& worker_command) {
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
    DestructThreadContext(full_partition_id);
  }
}

void WorkerSchedulerBase::ProcessReportStatusOfPartitions(
    const message::WorkerReportStatusOfPartitions& worker_command) {
  DeliverCommandToEachThread(worker_command, StageId::REQUEST_REPORT,
                             []() { return nullptr; });
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

void WorkerSchedulerBase::ActivateThreadContext(
    WorkerLightThreadContext* thread_context) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  const auto application_id = thread_context->get_application_id();
  const auto priority_level = application_priority_level_map_[application_id];
  activated_thread_priority_queue_[priority_level].push_back(thread_context);
  PCHECK(pthread_mutex_unlock(&scheduling_lock_) == 0);
  PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
}

WorkerLightThreadContext* WorkerSchedulerBase::GetActivatedThreadContext() {
  WorkerLightThreadContext* result = nullptr;
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  while (true) {
    auto iter = activated_thread_priority_queue_.begin();
    while (iter != activated_thread_priority_queue_.end()) {
      auto& queue = iter->second;
      if (queue.empty()) {
        // Moves to the next queue.
        iter = activated_thread_priority_queue_.erase(iter);
      } else {
        // Takes an activated thread context.
        result = queue.front();
        queue.pop_front();
        break;
      }
    }
    // If an activated thread context is taken.
    if (result) {
      break;
    }
    PCHECK(pthread_cond_wait(&scheduling_cond_, &scheduling_lock_) == 0);
  }
  pthread_mutex_unlock(&scheduling_lock_);
  return result;
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
  decltype(activated_thread_priority_queue_) clone;
  clone.swap(activated_thread_priority_queue_);
  for (auto& pair : clone)
    for (auto& thread_context : pair.second) {
      const auto application_id = thread_context->get_application_id();
      const auto priority_level =
          application_priority_level_map_[application_id];
      activated_thread_priority_queue_[priority_level].push_back(
          thread_context);
    }
}

void* WorkerSchedulerBase::ExecutionRoutine(void* arg) {
  auto scheduler = reinterpret_cast<WorkerSchedulerBase*>(arg);
  auto thread_context = scheduler->GetActivatedThreadContext();
  while (thread_context != nullptr) {
    if (thread_context->Enter()) {
      do {
        thread_context->Run();
      } while (!thread_context->Exit());
    }
    thread_context = scheduler->GetActivatedThreadContext();
  }
  LOG(WARNING) << "Execution thread exits.";
  return nullptr;
}

void WorkerSchedulerBase::ConstructThreadContext(
    const FullPartitionId& full_partition_id) {
  const auto application_id = full_partition_id.application_id;
  WorkerLightThreadContext* thread_context = LoadPartition(full_partition_id);
  thread_context->worker_id_ = self_worker_id_;
  thread_context->application_id_ = application_id;
  thread_context->variable_group_id_ = full_partition_id.variable_group_id;
  thread_context->partition_id_ = full_partition_id.partition_id;
  // Sets related callback/interface.
  thread_context->activate_callback_ = std::bind(
      &WorkerSchedulerBase::ActivateThreadContext, this, thread_context);
  thread_context->send_command_interface_ = send_command_interface_;
  thread_context->send_data_interface_ = send_data_interface_;
  // Sets the application.
  thread_context->canary_application_ =
      application_record_map_.at(application_id).loaded_application;
  // Initializes the thread context.
  thread_context->Initialize();
  // Registers the thread.
  CHECK(thread_map_.find(full_partition_id) == thread_map_.end());
  thread_map_[full_partition_id] = thread_context;
  ++application_record_map_.at(application_id).local_partitions;
}

void WorkerSchedulerBase::DestructThreadContext(
    const FullPartitionId& full_partition_id) {
  // Deregisters the thread.
  auto iter = thread_map_.find(full_partition_id);
  CHECK(iter != thread_map_.end());
  thread_map_.erase(iter);
  --application_record_map_.at(full_partition_id.application_id)
        .local_partitions;
  // Caution: the thread context is not deleted, it remains there as a tomb.
}

void WorkerScheduler::StartExecution() {
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

}  // namespace canary
