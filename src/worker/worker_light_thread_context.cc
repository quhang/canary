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
 * @file src/worker/worker_light_thread_context.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerLightThreadContext.
 */

#include "worker/worker_light_thread_context.h"

#include "message/message_include.h"
#include "worker/worker_scheduler.h"

namespace canary {

WorkerLightThreadContext::WorkerLightThreadContext() {
  PCHECK(pthread_mutex_init(&internal_lock_, nullptr) == 0);
}

WorkerLightThreadContext::~WorkerLightThreadContext() {
  pthread_mutex_destroy(&internal_lock_);
}

void WorkerLightThreadContext::DeliverMessage(StageId stage_id,
                                              struct evbuffer* buffer) {
  if (stage_id >= StageId::FIRST) {
    // Normal data routed to a stage.
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    auto& stage_buffer = stage_buffer_map_[stage_id];
    stage_buffer.buffer_list.push_back(buffer);
    // If enough messages are received for the stage.
    if (stage_buffer.expected_buffer ==
        static_cast<int>(stage_buffer.buffer_list.size())) {
      ready_stages_.push_back(stage_id);
      pthread_mutex_unlock(&internal_lock_);
      // Activates.
      worker_scheduler_->ActivateThreadContext(this);
    } else {
      pthread_mutex_unlock(&internal_lock_);
    }
  } else {
    // A command requiring attention.
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    command_list_.resize(command_list_.size() + 1);
    auto& command_buffer = command_list_.back();
    command_buffer.stage_id = stage_id;
    command_buffer.command = buffer;
    pthread_mutex_unlock(&internal_lock_);
    // Activates.
    worker_scheduler_->ActivateThreadContext(this);
  }
}

void WorkerLightThreadContext::RegisterReceivingData(StageId stage_id,
                                                     int num_message) {
  CHECK(stage_id >= StageId::FIRST);
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  auto& stage_buffer = stage_buffer_map_[stage_id];
  stage_buffer.expected_buffer = num_message;
  if (num_message == static_cast<int>(stage_buffer.buffer_list.size())) {
    ready_stages_.push_back(stage_id);
    pthread_mutex_unlock(&internal_lock_);
    // Activates.
    worker_scheduler_->ActivateThreadContext(this);
  } else {
    pthread_mutex_unlock(&internal_lock_);
  }
}

bool WorkerLightThreadContext::RetrieveCommand(StageId* stage_id,
                                               struct evbuffer** command) {
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

bool WorkerLightThreadContext::RetrieveData(
    StageId* stage_id, std::list<struct evbuffer*>* buffer_list) {
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

void WorkerExecutionContext::Initialize() {
  stage_graph_.set_statement_info_map(
      get_canary_application()->get_statement_info_map());
}

void WorkerExecutionContext::Finalize() { DeallocatePartitionData(); }

void WorkerExecutionContext::RunCommands() {
  struct evbuffer* command;
  StageId command_stage_id;
  while (RetrieveCommand(&command_stage_id, &command)) {
    switch (command_stage_id) {
      case StageId::INIT:
        CHECK(partition_state_ == PartitionState::UNINITIALIZED);
        partition_state_ = PartitionState::RUNNING;
        ProcessInitCommand(command);
        break;
      case StageId::CONTROL_FLOW_DECISION:
        CHECK(partition_state_ != PartitionState::UNINITIALIZED);
        ProcessControlFlowDecision(command);
        break;
      case StageId::MIGRATE_IN:
        // TODO(quhang): not implemented.
        LOG(FATAL) << "Not implemented.";
        break;
      case StageId::MIGRATE_OUT:
        // TODO(quhang): not implemented.
        LOG(FATAL) << "Not implemented.";
        break;
      case StageId::PAUSE_EXECUTION:
        // TODO(quhang): not implemented.
        LOG(FATAL) << "Not implemented.";
        break;
      case StageId::INSTALL_BARRIER:
        // TODO(quhang): not implemented.
        LOG(FATAL) << "Not implemented.";
        break;
      case StageId::RELEASE_BARRIER:
        CHECK(partition_state_ == PartitionState::IN_BARRIER);
        partition_state_ = PartitionState::RUNNING;
        CHECK(command == nullptr);
        ProcessReleaseBarrier();
        break;
      default:
        LOG(FATAL) << "Unknown command stage id!";
    }
  }
}

bool WorkerExecutionContext::RunOneStage() {
  std::list<struct evbuffer*> buffer_list;
  StageId stage_id;
  if (RetrieveData(&stage_id, &buffer_list)) {
    RunGatherStage(stage_id, pending_gather_stages_.at(stage_id), &buffer_list);
    pending_gather_stages_.erase(stage_id);
    return true;
  }
  StatementId statement_id;
  std::tie(stage_id, statement_id) = stage_graph_.GetNextReadyStage();
  if (stage_id == StageId::COMPLETE) {
    message::ControllerRespondPartitionDone report;
    FillInStats(&report);
    get_send_command_interface()->SendCommandToController(
        message::SerializeMessageWithControlHeader(report));
    CHECK(partition_state_ != PartitionState::COMPLETE);
    partition_state_ = PartitionState::COMPLETE;
    // No more stages to run.
    return false;
  } else if (stage_id == StageId::REACH_BARRIER) {
    message::ControllerRespondReachBarrier report;
    FillInStats(&report);
    get_send_command_interface()->SendCommandToController(
        message::SerializeMessageWithControlHeader(report));
    CHECK(partition_state_ != PartitionState::IN_BARRIER);
    partition_state_ = PartitionState::IN_BARRIER;
    // No more stages to run.
    return false;
  } else if (stage_id == StageId::INVALID) {
    // No more stages to run.
    return false;
  } else {
    RunStage(stage_id, statement_id);
    return true;
  }
}

void WorkerExecutionContext::Run() {
  if (partition_state_ == PartitionState::COMPLETE) {
    return;
  }
  RunCommands();
  if (partition_state_ != PartitionState::RUNNING) {
    return;
  }
  while (RunOneStage()) {
    continue;
  }
}

void WorkerExecutionContext::Report() {
  if (partition_state_ == PartitionState::COMPLETE) {
    return;
  }
  message::ControllerRespondStatusOfPartition report;
  FillInStats(&report);
  get_send_command_interface()->SendCommandToController(
      message::SerializeMessageWithControlHeader(report));
}

void WorkerExecutionContext::BuildStats(message::RunningStats* running_stats) {
  running_stats->earliest_unfinished_stage_id =
      stage_graph_.get_earliest_unfinished_stage_id();
  running_stats->last_finished_stage_id =
      stage_graph_.get_last_finished_stage_id();
  stage_graph_.retrieve_timestamp_stats(&running_stats->timestamp_stats);
  stage_graph_.retrieve_cycle_stats(&running_stats->cycle_stats);
}

//! Fills in running stats into a command.
template <typename T>
void WorkerExecutionContext::FillInStats(T* report) {
  report->from_worker_id = get_worker_id();
  report->application_id = get_application_id();
  report->variable_group_id = get_variable_group_id();
  report->partition_id = get_partition_id();
  BuildStats(&report->running_stats);
}

void WorkerExecutionContext::ProcessInitCommand(struct evbuffer* command) {
  // Caution: initialization generates the initial task graph.
  stage_graph_.Initialize(get_variable_group_id(), get_partition_id());
  using internal_message::InitCommand;
  auto struct_message = internal_message::to_command<InitCommand>(command);
  if (struct_message.first_barrier_stage != StageId::INVALID) {
    stage_graph_.InsertBarrier(struct_message.first_barrier_stage);
  }
  AllocatePartitionData();
}

void WorkerExecutionContext::ProcessControlFlowDecision(
    struct evbuffer* command) {
  using internal_message::ControlDecisionCommand;
  auto struct_message =
      internal_message::to_command<ControlDecisionCommand>(command);
  stage_graph_.FeedControlFlowDecision(struct_message.stage_id,
                                       struct_message.decision);
}

void WorkerExecutionContext::ProcessReleaseBarrier() {
  stage_graph_.ReleaseBarrier();
}

void WorkerExecutionContext::RunGatherStage(
    StageId stage_id, StatementId statement_id,
    std::list<struct evbuffer*>* buffer_list) {
  VLOG(1) << "Run gather stage=" << get_value(stage_id)
          << " statement=" << get_value(statement_id)
          << " variable_group=" << get_value(get_variable_group_id())
          << " partition=" << get_value(get_partition_id());
  const auto statement_info =
      get_canary_application()->get_statement_info_map()->at(statement_id);
  CanaryTaskContext task_context;
  PrepareTaskContext(stage_id, statement_id, statement_info, &task_context);
  task_context.receive_buffer_.swap(*buffer_list);

  const auto start_time = time::Clock::now();
  const int needed_message = (statement_info.int_task_function)(&task_context);
  const auto end_time = time::Clock::now();
  CHECK_EQ(needed_message, 0);
  stage_graph_.CompleteStage(stage_id, time::timepoint_to_double(start_time),
                             time::duration_to_double(end_time - start_time));
}

void WorkerExecutionContext::RunStage(StageId stage_id,
                                      StatementId statement_id) {
  VLOG(1) << "Run stage=" << get_value(stage_id)
          << " statement=" << get_value(statement_id)
          << " variable_group=" << get_value(get_variable_group_id())
          << " partition=" << get_value(get_partition_id());
  const auto statement_info =
      get_canary_application()->get_statement_info_map()->at(statement_id);
  CanaryTaskContext task_context;
  PrepareTaskContext(stage_id, statement_id, statement_info, &task_context);
  switch (statement_info.statement_type) {
    case CanaryApplication::StatementType::TRANSFORM: {
      const auto start_time = time::Clock::now();
      (statement_info.void_task_function)(&task_context);
      const auto end_time = time::Clock::now();
      stage_graph_.CompleteStage(
          stage_id, time::timepoint_to_double(start_time),
          time::duration_to_double(end_time - start_time));
      break;
    }
    case CanaryApplication::StatementType::SCATTER: {
      const auto start_time = time::Clock::now();
      (statement_info.void_task_function)(&task_context);
      const auto end_time = time::Clock::now();
      stage_graph_.CompleteStage(
          stage_id, time::timepoint_to_double(start_time),
          time::duration_to_double(end_time - start_time));
      break;
    }
    case CanaryApplication::StatementType::GATHER: {
      const auto start_time = time::Clock::now();
      const int needed_message =
          (statement_info.int_task_function)(&task_context);
      const auto end_time = time::Clock::now();
      if (needed_message == 0) {
        VLOG(1) << "Gather stage needs no data, and falls throught.";
        stage_graph_.CompleteStage(
            stage_id, time::timepoint_to_double(start_time),
            time::duration_to_double(end_time - start_time));
      } else {
        VLOG(1) << "Gather stage needs message of " << needed_message;
        pending_gather_stages_[stage_id] = statement_id;
        RegisterReceivingData(stage_id, needed_message);
      }
      break;
    }
    case CanaryApplication::StatementType::WHILE: {
      const auto start_time = time::Clock::now();
      const bool decision = (statement_info.bool_task_function)(&task_context);
      const auto end_time = time::Clock::now();
      stage_graph_.CompleteStage(
          stage_id, time::timepoint_to_double(start_time),
          time::duration_to_double(end_time - start_time));
      // Broadcast control flow decision.
      for (const auto& pair :
           *get_canary_application()->get_variable_group_info_map()) {
        get_send_data_interface()->BroadcastDataToPartition(
            get_application_id(), pair.first, StageId::CONTROL_FLOW_DECISION,
            SerializeControlFlowDecision(stage_id, decision));
      }
      break;
    }
    case CanaryApplication::StatementType::LOOP:
    case CanaryApplication::StatementType::END_LOOP:
    case CanaryApplication::StatementType::END_WHILE:
      LOG(FATAL) << "Invalid statement type!";
      break;
    default:
      LOG(FATAL) << "Unknown statement type!";
      break;
  }
}

void WorkerExecutionContext::PrepareTaskContext(
    StageId stage_id, StatementId statement_id,
    const CanaryApplication::StatementInfo& statement_info,
    CanaryTaskContext* task_context) {
  task_context->send_data_interface_ = get_send_data_interface();
  for (const auto& pair : statement_info.variable_access_map) {
    if (pair.second == CanaryApplication::VariableAccess::READ) {
      task_context->read_partition_data_map_[pair.first] =
          local_partition_data_.at(pair.first).get();
    } else {
      CHECK(pair.second == CanaryApplication::VariableAccess::WRITE);
      task_context->write_partition_data_map_[pair.first] =
          local_partition_data_.at(pair.first).get();
    }
  }
  task_context->self_partition_id_ = get_value(get_partition_id());
  task_context->application_id_ = get_application_id();
  if (statement_info.statement_type ==
      CanaryApplication::StatementType::SCATTER) {
    task_context->scatter_partitioning_ = statement_info.parallelism;
    task_context->gather_partitioning_ =
        statement_info.paired_gather_parallelism;
    task_context->gather_variable_group_id_ = get_canary_application()
                                                  ->get_statement_info_map()
                                                  ->at(get_next(statement_id))
                                                  .variable_group_id;
    task_context->gather_stage_id_ = get_next(stage_id);
  } else if (statement_info.statement_type ==
             CanaryApplication::StatementType::GATHER) {
    task_context->scatter_partitioning_ =
        statement_info.paired_scatter_parallelism;
    task_context->gather_partitioning_ = statement_info.parallelism;
  }
}

struct evbuffer* WorkerExecutionContext::SerializeControlFlowDecision(
    StageId stage_id, bool decision) {
  internal_message::ControlDecisionCommand command{stage_id, decision};
  return internal_message::to_buffer(command);
}

void WorkerExecutionContext::AllocatePartitionData() {
  const auto variable_group_info_map =
      get_canary_application()->get_variable_group_info_map();
  const auto variable_info_map =
      get_canary_application()->get_variable_info_map();
  for (auto variable_id :
       variable_group_info_map->at(get_variable_group_id()).variable_id_set) {
    local_partition_data_[variable_id].reset(
        variable_info_map->at(variable_id).data_prototype->Clone());
    local_partition_data_[variable_id]->Initialize();
  }
}

void WorkerExecutionContext::DeallocatePartitionData() {
  for (auto& pair : local_partition_data_) {
    if (pair.second) {
      pair.second->Finalize();
      pair.second.reset();
    }
  }
}

}  // namespace canary
