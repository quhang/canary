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
}

bool WorkerSchedulerBase::ReceiveRoutedData(ApplicationId application_id,
                                            VariableGroupId variable_group_id,
                                            PartitionId partition_id,
                                            StageId stage_id,
                                            struct evbuffer* buffer) {
  CHECK(is_initialized_);
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
  CHECK(is_initialized_);
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
  CHECK(is_initialized_);
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
    PROCESS_MESSAGE(WORKER_REPORT_STATUS_OF_WORKER,
                    ProcessReportStatusOfWorker);
    PROCESS_MESSAGE(WORKER_CONTROL_PARTITIONS, ProcessControlPartitions);
    default:
      LOG(FATAL) << "Unexpected message category!";
  }
}
#undef PROCESS_MESSAGE

void WorkerSchedulerBase::AssignWorkerId(WorkerId worker_id) {
  is_initialized_ = true;
  self_worker_id_ = worker_id;
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
  application_record.local_partitions = 0;
  LoadApplicationBinary(&application_record);
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
  for (const auto& pair : worker_command.load_partitions) {
    const auto full_partition_id =
        FullPartitionId{application_id, pair.first, pair.second};
    VLOG(1) << "Load " << get_value(application_id) << "/"
            << get_value(pair.first) << "/" << get_value(pair.second);
    CHECK(thread_map_.find(full_partition_id) == thread_map_.end());
    ++application_record_map_.at(application_id).local_partitions;
    WorkerLightThreadContext* thread_context = LoadPartition(full_partition_id);
    thread_context->set_application_id(application_id);
    thread_context->set_variable_group_id(pair.first);
    thread_context->set_partition_id(pair.second);
    thread_context->set_priority_level(PriorityLevel::FIRST);
    thread_context->set_activate_callback(std::bind(
        &WorkerSchedulerBase::ActivateThreadContext, this, thread_context));
    thread_context->set_send_command_interface(send_command_interface_);
    thread_context->set_send_data_interface(send_data_interface_);
    thread_context->Initialize();
    thread_map_[full_partition_id] = thread_context;
    thread_context->DeliverMessage(StageId::INIT, nullptr);
    send_data_interface_->RefreshRouting();
  }
}

void WorkerSchedulerBase::ProcessUnloadPartitions(
    const message::WorkerUnloadPartitions& worker_command) {
  const auto application_id = worker_command.application_id;
  for (const auto& pair : worker_command.unload_partitions) {
    const auto full_partition_id =
        FullPartitionId{application_id, pair.first, pair.second};
    auto iter = thread_map_.find(full_partition_id);
    CHECK(iter != thread_map_.end());
    --application_record_map_.at(application_id).local_partitions;
    UnloadPartition(iter->second);
    iter->second->Finalize();
    // The thread context is not deleted, it remains there as a tomb.
    thread_map_.erase(iter);
  }
}

void WorkerSchedulerBase::ProcessMigrateInPartitions(
    const message::WorkerMigrateInPartitions& worker_command) {
  LOG(FATAL) << "WORKER_MIGRATE_IN_PARTITIONS";
}
void WorkerSchedulerBase::WorkerSchedulerBase::ProcessMigrateOutPartitions(
    const message::WorkerMigrateOutPartitions& worker_command) {
  LOG(FATAL) << "WORKER_MIGRATE_OUT_PARTITIONS";
}
void WorkerSchedulerBase::ProcessReportStatusOfPartitions(
    const message::WorkerReportStatusOfPartitions& worker_command) {
  LOG(FATAL) << "WORKER_REPORT_STATUS_OF_PARTITIONS";
}
void WorkerSchedulerBase::ProcessReportStatusOfWorker(
    const message::WorkerReportStatusOfWorker& worker_command) {
  LOG(FATAL) << "WORKER_REPORT_STATUS_OF_WORKER";
}
void WorkerSchedulerBase::ProcessControlPartitions(
    const message::WorkerControlPartitions& worker_command) {
  LOG(FATAL) << "WORKER_CONTROL_PARTITIONS";
}

void WorkerSchedulerBase::ActivateThreadContext(
    WorkerLightThreadContext* thread_context) {
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  activated_thread_queue_.push_back(thread_context);
  PCHECK(pthread_mutex_unlock(&scheduling_lock_) == 0);
  PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
}

WorkerLightThreadContext* WorkerSchedulerBase::GetActivatedThreadContext() {
  WorkerLightThreadContext* result = nullptr;
  PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
  while (activated_thread_queue_.empty()) {
    PCHECK(pthread_cond_wait(&scheduling_cond_, &scheduling_lock_) == 0);
  }
  result = activated_thread_queue_.front();
  activated_thread_queue_.pop_front();
  pthread_mutex_unlock(&scheduling_lock_);
  return result;
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

void WorkerScheduler::StartExecution() {
  // Reads global variable: the number of worker threads.
  thread_handle_list_.resize(FLAGS_worker_threads);
  for (auto& handle : thread_handle_list_) {
    // TODO(quhang): set thread priority.
    PCHECK(pthread_create(&handle, nullptr,
                          &WorkerSchedulerBase::ExecutionRoutine, this) == 0);
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

  // Instantiates the application object.
  application_record->variable_info_map =
      application_record->loaded_application->get_variable_info_map();
  application_record->statement_info_map =
      application_record->loaded_application->get_statement_info_map();
}

void WorkerScheduler::UnloadApplicationBinary(
    ApplicationRecord* application_record) {
  VLOG(1) << "Unload application binary.";
  CanaryApplication::UnloadApplication(application_record->loading_handle,
                                       application_record->loaded_application);
}

WorkerLightThreadContext* WorkerScheduler::LoadPartition(FullPartitionId) {
  VLOG(1) << "Load partition.";
  return new WorkerExecutionContext();
}

void WorkerScheduler::UnloadPartition(
    WorkerLightThreadContext* thread_context) {
  CHECK_NOTNULL(thread_context);
  return;
}

}  // namespace canary
