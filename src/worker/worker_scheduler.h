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

#include "worker/worker_communication_interface.h"

namespace canary {

// WorkerMigrationThreadContext.
// WorkerFileServiceThreadContext.
// WorkerPartitionExecutionThreadContext.

class WorkerLightThreadContext {
 private:
  struct StageBuffer {
    std::list<struct evbuffer*> buffers;
    int expected_buffer = -1;
  };
  struct CommandBuffer {
    StageId stage_id;
    struct evbuffer* command;
  };

 public:
  WorkerLightThreadContext() {
    PCHECK(pthread_mutex_init(&internal_lock_, nullptr) == 0);
    running_ = false;
  }

  virtual ~WorkerLightThreadContext() {
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    CHECK(!running_);
    pthread_mutex_unlock(&internal_lock_);
    pthread_mutex_destroy(internal_lock_);
  }

  // Initializes the light thread.
  virtual void Initialize() = 0;
  // Finalizes the light thread.
  virtual void Finalize() = 0;

  //! Delivers a message.
  void DeliverMessage(StageId stage_id, struct evbuffer* buffer) {
    bool to_activate = false;
    if (stage_id >= StageId::FIRST) {
      // Normal message.
      PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
      auto& stage_buffer = stage_buffer_map[stage_id];
      stage_buffer.buffers.push_back(buffer);
      // If enough messages are received for the stage.
      if (stage_buffer.expected_buffer ==
          static_cast<int>(stage_buffer.buffers.size())) {
        ready_stages_.push_back(stage_id);
        if (!running_) {
          to_activate = true;
        }
      }
      pthread_mutex_unlock(&internal_lock_);
    } else {
      // A special command that needs special attention.
      PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
      command_list_.resize(command_list_.size() + 1);
      auto& command_buffer = command_list_.back();
      command_buffer.stage_id = stage_id;
      command_bufer.command = buffer;
      to_activate = !running_;
      pthread_mutex_unlock(&internal_lock_);
    }

    if (to_activate) {
      Activate();
    }
  }

  //! Tries to enter the execution context of the thread. Returns true if there
  // is data to process.
  bool Enter() {
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    const bool success =
        (!running_) && (!ready_stages_.empty() || !command_list_.empty());
    running_ = true;
    pthread_mutex_unlock(&internal_lock_);
    return success;
  }

  //! Tries to exit the execution context of the thread. Returns true if there
  // is no data to process.
  bool Exit() {
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    const bool success = (ready_stages_.empty() && command_list_.empty());
    if (success) {
      running_ = false;
    }
    pthread_mutex_unlock(&internal_lock_);
    return success;
  }

  //! Forces the execution context to exit.
  void ForceExit() {
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    const bool to_activate = (ready_stages_.empty() && command_list_.empty());
    running_ = false;
    pthread_mutex_unlock(&internal_lock_);

    if (to_activate) {
      Activate();
    }
  }

  //! Registers how many messages are expected for a message.
  void RegisterReceivingMessage(StageId stage_id, int num_message) {
    CHECK(stage_id >= StageId::FIRST);
    bool to_activate = false;

    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    auto& stage_buffer = received_stage_buffers_[stage_id];
    stage_buffer.expected_buffer = num_message;
    if (num_message == static_cast<int>(stage_buffer.buffers.size())) {
      ready_stages_.push_back(stage_id);
      if (!running_) {
        to_activate = true;
      }
    }
    pthread_mutex_unlock(&internal_lock_);

    if (to_activate) {
      Activate();
    }
  }

  bool RetrieveCommand(StageId* stage_id, struct evbuffer** command) {
    // TODO(quhang): gets a command.
  }

  bool RetrieveStageBuffer(StageId* stage_id,
                           std::list<struct evbuffer*>* buffer_list) {
    // TODO(quhang): gets a stage data.
  }

  virtual void Run() = 0;

 private:
  void Activate() {
    // Adds itself to the scheduler.
  }

  pthread_mutex_t internal_lock_;
  bool runnning_ = false;

  std::map<StageId, StageBuffer> stage_buffer_map_;
  std::list<StageId> ready_stages_;

  std::list<CommandBuffer> command_list_;
};

class WorkerScheduler : public WorkerReceiveCommandInterface,
                        public WorkerReceiveDataInterface {
 public:
  WorkerScheduler() {}
  virtual ~WorkerScheduler() {}

  void Initialize(WorkerSendCommandInterface* send_command_interface,
                  WorkerSendDataInterface* send_data_interface) {
    send_command_interface_ = CHECK_NOTNULL(send_command_interface);
    send_data_interface_ = CHECK_NOTNULL(send_data_interface);
  }

  bool ReceiveRoutedData(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id, StageId stage_id,
                         struct evbuffer* buffer) override {
    auto iter = thread_map_.find(
        FullPartitionId(application_id, variable_group_id, partition_id));
    if (iter == thread_map_.end()) {
      return false;
    }
    WorkerLightThreadContext* thread_context = *iter;
    thread_context->DeliverMessage(stage_id, buffer);
    return true;
  }

  void ReceiveDirectData(struct evbuffer* buffer) override {
    // If it is a migrated partition, delivers to the partition migration
    // manager.
    // If it is file system traffic, delivers to the file system manager thread.
  }

  void ReceiveCommandFromController(struct evbuffer* buffer) override {
    // LoadApplication.
    // LoadPartitions.
    // UnloadPartitions.
    // UnloadApplication.

    // MigrateInPartitions.
    // MigrateOutPartitions.

    // ReportPartitionStatus.
    // ReportWorkerStatus.
    auto partition_execution = new WorkerPartitionExecutionThreadContext();
    // Add the execution context.
  }

  void AssignWorkerId(WorkerId worker_id) override {
    is_initialized_ = true;
    self_worker_id_ = worker_id;
    // TODO(quhang): responds back to the controller.
    // TODO(quhang): starts execution routine.
  }

  void PushActivatedThreadQueue(WorkerLightThreadContext* thread_context) {
    PCHECK(pthread_mutex_lock(&scheduling_lock_), 0);
    activated_thread_queue_.push_back(thread_context);
    PCHECK(pthread_mutex_unlock(&scheduling_lock_), 0);
    PCHECK(pthread_cond_signal(&scheduling_cond_) == 0);
  }

  WorkerLightThreadContext* GetActivatedThreadQueue() {
    WorkerLightThreadContext* result = nullptr;
    PCHECK(pthread_mutex_lock(&scheduling_lock_), 0);
    while (!activated_thread_queue_.empty()) {
      PCHECK(pthread_cond_wait(&scheduling_cond_, &scheduling_lock_) == 0);
    }
    result = activated_thread_queue_.back();
    activated_thread_queue_.pop_back();
    pthread_mutex_unlock(&scheduling_lock_);
    return result;
  }

  static void ExecutionRoutine(WorkerScheduler* scheduler) {
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
  }

 private:
  pthread_mutex_t scheduling_lock_;
  pthread_cond_t scheduling_cond_;

  std::map<FullPartitionId, WorkerLightThreadContext*> thread_map_;
  phread_mutex_t internal_lock_;

  std::list<LightThreadContext*> activated_thread_queue_;

  bool is_initialized_ = false;
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;

  WorkerId self_worker_id_ = WorkerId::INVALID;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_SCHEDULER_H_
