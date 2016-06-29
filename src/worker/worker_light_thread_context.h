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
 * @file src/worker/worker_light_thread_context.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerLightThreadContext.
 */

#ifndef CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
#define CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_

#include <list>
#include <map>

#include "shared/internal.h"
#include "worker/worker_communication_interface.h"

namespace canary {

// WorkerMigrationThreadContext.
// WorkerFileServiceThreadContext.
// WorkerPartitionExecutionThreadContext.

/**
 * The execution context of a lightweight thread, which is responsible for
 * execution of a partition or other special tasks.
 */
class WorkerLightThreadContext {
 private:
  //! The buffer storing received data of a stage.
  struct StageBuffer {
    std::list<struct evbuffer*> buffer_list;
    int expected_buffer = -1;
  };
  //! The buffer storing a command.
  struct CommandBuffer {
    StageId stage_id;
    struct evbuffer* command;
  };

 public:
  //! Constructor.
  WorkerLightThreadContext() {
    PCHECK(pthread_mutex_init(&internal_lock_, nullptr) == 0);
  }

  //! Destructor.
  virtual ~WorkerLightThreadContext() {
    pthread_mutex_destroy(&internal_lock_);
  }

  //! Initializes the light thread.
  virtual void Initialize() = 0;

  //! Finalizes the light thread.
  virtual void Finalize() = 0;

  //! Runs the thread.
  virtual void Run() = 0;

 public:
  //! Gets/sets the metadata of the thread.
  ApplicationId get_application_id() const { return application_id_; }
  void set_application_id(ApplicationId application_id) {
    application_id_ = application_id;
  }
  VariableGroupId get_variable_group_id() const { return variable_group_id_; }
  void set_variable_group_id(VariableGroupId variable_group_id) {
    variable_group_id_ = variable_group_id;
  }
  PartitionId get_partition_id() const { return partition_id_; }
  void set_partition_id(PartitionId partition_id) {
    partition_id_ = partition_id;
  }
  PriorityLevel get_priority_level() const { return priority_level_; }
  void set_priority_level(PriorityLevel priority_level) {
    priority_level_ = priority_level;
  }
  void set_activate_callback(std::function<void()> activate_callback) {
    activate_callback_ = std::move(activate_callback);
  }
  WorkerSendCommandInterface* get_send_command_interface() {
    return send_command_interface_;
  }
  void set_send_command_interface(WorkerSendCommandInterface* interface) {
    send_command_interface_ = interface;
  }
  WorkerSendDataInterface* get_send_data_interface() {
    return send_data_interface_;
  }
  void set_send_data_interface(WorkerSendDataInterface* interface) {
    send_data_interface_ = interface;
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
    CHECK(running_);
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
    const bool to_activate = (!ready_stages_.empty() || command_list_.empty());
    running_ = false;
    pthread_mutex_unlock(&internal_lock_);

    if (to_activate && activate_callback_) {
      activate_callback_();
    }
  }

  //! Delivers a message.
  void DeliverMessage(StageId stage_id, struct evbuffer* buffer) {
    bool to_activate = false;
    if (stage_id >= StageId::FIRST) {
      // Normal data routed to a stage.
      PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
      auto& stage_buffer = stage_buffer_map_[stage_id];
      stage_buffer.buffer_list.push_back(buffer);
      // If enough messages are received for the stage.
      if (stage_buffer.expected_buffer ==
          static_cast<int>(stage_buffer.buffer_list.size())) {
        ready_stages_.push_back(stage_id);
        if (!running_) {
          to_activate = true;
        }
      }
      pthread_mutex_unlock(&internal_lock_);
    } else {
      // A command requiring attention.
      PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
      command_list_.resize(command_list_.size() + 1);
      auto& command_buffer = command_list_.back();
      command_buffer.stage_id = stage_id;
      command_buffer.command = buffer;
      to_activate = !running_;
      pthread_mutex_unlock(&internal_lock_);
    }

    if (to_activate && activate_callback_) {
      activate_callback_();
    }
  }

 protected:
  //! Registers how many messages are expected for a message.
  void RegisterReceivingMessage(StageId stage_id, int num_message) {
    CHECK(stage_id >= StageId::FIRST);
    bool to_activate = false;

    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    auto& stage_buffer = stage_buffer_map_[stage_id];
    stage_buffer.expected_buffer = num_message;
    if (num_message == static_cast<int>(stage_buffer.buffer_list.size())) {
      ready_stages_.push_back(stage_id);
      if (!running_) {
        to_activate = true;
      }
    }
    pthread_mutex_unlock(&internal_lock_);

    if (to_activate && activate_callback_) {
      activate_callback_();
    }
  }

  //! Retrieves a command.
  bool RetrieveCommand(StageId* stage_id, struct evbuffer** command) {
    bool result = false;
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    if (!command_list_.empty()) {
      auto& command_buffer = command_list_.front();
      *stage_id = command_buffer.stage_id;
      if (command != nullptr) {
        *command = command_buffer.command;
      }
      command_list_.pop_front();
      result = true;
    }
    pthread_mutex_unlock(&internal_lock_);
    return result;
  }

  //! Retrieves the buffer of a stage.
  bool RetrieveStageBuffer(StageId* stage_id,
                           std::list<struct evbuffer*>* buffer_list) {
    bool result = false;
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    if (!ready_stages_.empty()) {
      const auto ready_stage = ready_stages_.front();
      ready_stages_.pop_front();
      auto iter = stage_buffer_map_.find(ready_stage);
      CHECK(iter != stage_buffer_map_.end());
      *stage_id = ready_stage;
      buffer_list->swap(iter->second.buffer_list);
      stage_buffer_map_.erase(iter);
      result = true;
    }
    pthread_mutex_unlock(&internal_lock_);
    return result;
  }

 private:
  ApplicationId application_id_ = ApplicationId::INVALID;
  VariableGroupId variable_group_id_ = VariableGroupId::INVALID;
  PartitionId partition_id_ = PartitionId::INVALID;
  PriorityLevel priority_level_ = PriorityLevel::INVALID;
  std::function<void()> activate_callback_;
  WorkerSendCommandInterface* send_command_interface_ = nullptr;
  WorkerSendDataInterface* send_data_interface_ = nullptr;

  //! Synchronization lock.
  pthread_mutex_t internal_lock_;
  bool running_ = false;

  //! Received commands.
  std::list<CommandBuffer> command_list_;
  //! Received data.
  std::map<StageId, StageBuffer> stage_buffer_map_;
  std::list<StageId> ready_stages_;
};

class WorkerExecutionContext : public WorkerLightThreadContext {
 public:
  WorkerExecutionContext() {}
  virtual ~WorkerExecutionContext() {}

  //! Initializes the light thread.
  void Initialize() override { LOG(INFO) << "Initialized!"; }

  //! Finalizes the light thread.
  void Finalize() override { LOG(INFO) << "Finalized!"; }

  //! Runs the thread.
  void Run() override {
    struct evbuffer* command;
    StageId command_stage_id;
    if (RetrieveCommand(&command_stage_id, &command)) {
      ProcessCommand(command_stage_id, command);
      return;
    }
    std::list<struct evbuffer*> buffer_list;
    StageId stage_id;
    if (RetrieveStageBuffer(&stage_id, &buffer_list)) {
      ProcessData(stage_id, &buffer_list);
      return;
    }
  }

 private:
  void ProcessCommand(StageId command_stage_id, struct evbuffer* command) {
    CHECK(command_stage_id < StageId::INVALID);
    switch (command_stage_id) {
      case StageId::INIT:
        if (get_partition_id() == PartitionId::FIRST) {
          RegisterReceivingMessage(StageId::FIRST, 1);
        } else {
          struct evbuffer* buffer = evbuffer_new();
          CanaryOutputArchive archive(buffer);
          archive(StageId::FIRST);
          get_send_data_interface()->SendDataToPartition(
              get_application_id(), get_variable_group_id(), PartitionId::FIRST,
              StageId::FIRST, buffer);
        }
        LOG(INFO) << "INIT command.";
        break;
      default:
        LOG(FATAL) << "Unknown command stage id!";
    }
    evbuffer_free(command);
  }
  void ProcessData(StageId stage_id, std::list<struct evbuffer*>* buffer_list) {
    if (get_partition_id() == PartitionId::FIRST) {
      CHECK_EQ(buffer_list->size(), 1u);
      CanaryInputArchive archive(buffer_list->front());
      StageId in_message_stage_id;
      archive(in_message_stage_id);
      LOG(INFO) << "First partition receives: " <<
          get_value(in_message_stage_id);
      CHECK(in_message_stage_id == stage_id);

      struct evbuffer* send_buffer = evbuffer_new();
      CanaryOutputArchive out_archive(send_buffer);
      out_archive(get_next(stage_id));
      get_send_data_interface()->SendDataToPartition(
          get_application_id(), get_variable_group_id(),
          get_next(PartitionId::FIRST), get_next(stage_id), send_buffer);
      RegisterReceivingMessage(get_next(stage_id, 2), 1);
    } else {
      CHECK_EQ(buffer_list->size(), 1u);
      CanaryInputArchive archive(buffer_list->front());
      StageId in_message_stage_id;
      archive(in_message_stage_id);
      LOG(INFO) << "Second partition receives: " <<
          get_value(in_message_stage_id);
      CHECK(in_message_stage_id == stage_id);

      struct evbuffer* send_buffer = evbuffer_new();
      CanaryOutputArchive out_archive(send_buffer);
      out_archive(get_next(stage_id));
      get_send_data_interface()->SendDataToPartition(
          get_application_id(), get_variable_group_id(),
          PartitionId::FIRST, get_next(stage_id), send_buffer);
      RegisterReceivingMessage(get_next(stage_id, 2), 1);
    }

    for (auto buffer_to_delete : *buffer_list) {
      evbuffer_free(buffer_to_delete);
    }
  }
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_WORKER_LIGHT_THREAD_CONTEXT_H_
