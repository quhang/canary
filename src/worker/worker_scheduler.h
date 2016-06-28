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
 * @file src/worker/worker_scheduler.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerScheduler.
 */

#ifndef CANARY_SRC_WORKER_WORKER_SCHEDULER_H_
#define CANARY_SRC_WORKER_WORKER_SCHEDULER_H_

#include <list>
#include <map>
#include <string>
#include <vector>

#include "shared/internal.h"
#include "shared/partition_map.h"
#include "worker/worker_communication_interface.h"
#include "worker/worker_light_thread_context.h"
#include "message/message_include.h"

namespace canary {

class WorkerScheduler : public WorkerReceiveCommandInterface,
                        public WorkerReceiveDataInterface {
 private:
  //! The record of an application.
  struct ApplicationRecord {
    ApplicationId application_id = ApplicationId::INVALID;
    std::string binary_location;
    std::string application_parameter;
    int local_partitions = 0;
  };

 public:
  //! Constructor.
  WorkerScheduler() {
    PCHECK(pthread_mutex_init(&scheduling_lock_, nullptr) == 0);
    PCHECK(pthread_cond_init(&scheduling_cond_, nullptr) == 0);
  }
  //! Destroctor.
  virtual ~WorkerScheduler() {
    pthread_mutex_destroy(&scheduling_lock_);
    pthread_cond_destroy(&scheduling_cond_);
  }

  // Initializes the worker scheduler.
  void Initialize(WorkerSendCommandInterface* send_command_interface,
                  WorkerSendDataInterface* send_data_interface) {
    send_command_interface_ = CHECK_NOTNULL(send_command_interface);
    send_data_interface_ = CHECK_NOTNULL(send_data_interface);
  }

  //! Delivers the routed data to a thread context, and returns false if no
  // thread context is found.
  bool ReceiveRoutedData(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id, StageId stage_id,
                         struct evbuffer* buffer) override {
    auto iter = thread_map_.find(
        FullPartitionId{application_id, variable_group_id, partition_id});
    if (iter == thread_map_.end()) {
      return false;
    }
    WorkerLightThreadContext* thread_context = iter->second;
    thread_context->DeliverMessage(stage_id, buffer);
    return true;
  }

  //! Delivers direct data.
  void ReceiveDirectData(struct evbuffer* buffer) override {
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
  //! Processes worker commands.
  void ReceiveCommandFromController(struct evbuffer* buffer) override {
    auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
    using message::MessageCategoryGroup;
    using message::MessageCategory;
    using message::ControlHeader;
    CHECK(header->category_group == MessageCategoryGroup::WORKER_COMMAND);
    // If assigned worker_id.
    // Differentiante between control messages and command messages.
    switch (header->category) {
      PROCESS_MESSAGE(WORKER_LOAD_APPLICATION, ProcessLoadApplication);
      PROCESS_MESSAGE(WORKER_UNLOAD_APPLICATION, ProcessUnloadApplication);
      PROCESS_MESSAGE(WORKER_LOAD_PARTITIONS, ProcessLoadPartitions);
      PROCESS_MESSAGE(WORKER_UNLOAD_PARTITIONS, ProcessUnloadPartitions);
      PROCESS_MESSAGE(WORKER_MIGRATE_IN_PARTITIONS, ProcessMigrateInPartitions);
      PROCESS_MESSAGE(WORKER_MIGRATE_OUT_PARTITIONS,
                      ProcessMigrateOutPartitions);
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

  //! Assigns a worker id.
  void AssignWorkerId(WorkerId worker_id) override {
    is_initialized_ = true;
    self_worker_id_ = worker_id;
    thread_handle_list_.resize(FLAGS_worker_threads);
    for (auto& handle : thread_handle_list_) {
      PCHECK(pthread_create(
              &handle, nullptr, &WorkerScheduler::ExecutionRoutine, this) == 0);
    }
  }

 private:
  //! Loads an application.
  void ProcessLoadApplication(
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
    LOG(INFO) << "Load application.";
    // TODO(quhang): dynamically load the application binary.
  }

  //! Unloads an application.
  void ProcessUnloadApplication(
      const message::WorkerUnloadApplication& worker_command) {
    const auto remove_application_id = worker_command.application_id;
    auto iter = application_record_map_.find(remove_application_id);
    CHECK(iter != application_record_map_.end());
    CHECK_EQ(iter->second.local_partitions, 0);
    application_record_map_.erase(iter);
    LOG(INFO) << "Unload application.";
  }

  // Loads partitions.
  void ProcessLoadPartitions(
      const message::WorkerLoadPartitions& worker_command) {
    const auto application_id = worker_command.application_id;
    for (const auto& pair : worker_command.load_partitions) {
      const auto full_partition_id =
          FullPartitionId{application_id, pair.first, pair.second};
      CHECK(thread_map_.find(full_partition_id) == thread_map_.end());
      ++application_record_map_.at(application_id).local_partitions;
      auto thread_context = new WorkerExecutionContext();
      thread_context->set_application_id(application_id);
      thread_context->set_variable_group_id(pair.first);
      thread_context->set_partition_id(pair.second);
      thread_context->set_priority_level(PriorityLevel::FIRST);
      thread_context->set_activate_callback(std::bind(
          &WorkerScheduler::ActivateThreadQueue, this, thread_context));
      thread_context->set_send_command_interface(send_command_interface_);
      thread_context->set_send_data_interface(send_data_interface_);
      thread_context->Initialize();
      thread_map_[full_partition_id] = thread_context;
      thread_context->DeliverMessage(StageId::INIT, nullptr);
    }
  }

  // Unloads partitions.
  void ProcessUnloadPartitions(
      const message::WorkerUnloadPartitions& worker_command) {
    const auto application_id = worker_command.application_id;
    for (const auto& pair : worker_command.unload_partitions) {
      const auto full_partition_id =
          FullPartitionId{application_id, pair.first, pair.second};
      auto iter = thread_map_.find(full_partition_id);
      CHECK(iter != thread_map_.end());
      --application_record_map_.at(application_id).local_partitions;
      // TODO(quhang): checks the thread context is not active and cleans up the
      // partition.
      thread_map_.erase(iter);
    }
  }

  void ProcessMigrateInPartitions(
      const message::WorkerMigrateInPartitions& worker_command) {
    LOG(FATAL) << "WORKER_MIGRATE_IN_PARTITIONS";
  }
  void ProcessMigrateOutPartitions(
      const message::WorkerMigrateOutPartitions& worker_command) {
    LOG(FATAL) << "WORKER_MIGRATE_OUT_PARTITIONS";
  }
  void ProcessReportStatusOfPartitions(
      const message::WorkerReportStatusOfPartitions& worker_command) {
    LOG(FATAL) << "WORKER_REPORT_STATUS_OF_PARTITIONS";
  }
  void ProcessReportStatusOfWorker(
      const message::WorkerReportStatusOfWorker& worker_command) {
    LOG(FATAL) << "WORKER_REPORT_STATUS_OF_WORKER";
  }
  void ProcessControlPartitions(
      const message::WorkerControlPartitions& worker_command) {
    LOG(FATAL) << "WORKER_CONTROL_PARTITIONS";
  }

 private:
  //! Activates a thread context.
  void ActivateThreadQueue(WorkerLightThreadContext* thread_context) {
    PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
    activated_thread_queue_.push_back(thread_context);
    PCHECK(pthread_mutex_unlock(&scheduling_lock_) == 0);
    PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
  }

  //! Retrieves an activated thread context.
  WorkerLightThreadContext* GetActivatedThreadQueue() {
    WorkerLightThreadContext* result = nullptr;
    PCHECK(pthread_mutex_lock(&scheduling_lock_) == 0);
    while (!activated_thread_queue_.empty()) {
      PCHECK(pthread_cond_wait(&scheduling_cond_, &scheduling_lock_) == 0);
    }
    result = activated_thread_queue_.back();
    activated_thread_queue_.pop_back();
    pthread_mutex_unlock(&scheduling_lock_);
    return result;
  }

  //! Execution routine.
  static void* ExecutionRoutine(void* arg) {
    auto scheduler = reinterpret_cast<WorkerScheduler*>(arg);
    auto thread_context = scheduler->GetActivatedThreadQueue();
    while (thread_context != nullptr) {
      if (thread_context->Enter()) {
        do {
          thread_context->Run();
        } while (!thread_context->Exit());
      }
      thread_context = scheduler->GetActivatedThreadQueue();
    }
    LOG(WARNING) << "Execution thread exits.";
    return nullptr;
  }

 private:
  //! Thread scheduling sychronization lock and condition variable.
  pthread_mutex_t scheduling_lock_;
  pthread_cond_t scheduling_cond_;

  //! All thread contexts.
  std::map<FullPartitionId, WorkerLightThreadContext*> thread_map_;
  //! Activated thread contexts.
  std::list<WorkerLightThreadContext*> activated_thread_queue_;

  //! All application records.
  std::map<ApplicationId, ApplicationRecord> application_record_map_;

  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;

  bool is_initialized_ = false;
  WorkerId self_worker_id_ = WorkerId::INVALID;

  std::vector<pthread_t> thread_handle_list_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_SCHEDULER_H_
