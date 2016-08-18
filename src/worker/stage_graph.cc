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
 * @file src/worker/stage_graph.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class StageGraph.
 */

#include "worker/stage_graph.h"

#include <algorithm>

namespace canary {

void StageGraph::Initialize(VariableGroupId self_variable_group_id,
                            PartitionId self_partition_id) {
  CHECK_NOTNULL(statement_info_map_);
  self_variable_group_id_ = self_variable_group_id;
  self_partition_id_ = self_partition_id;
  CHECK(self_variable_group_id_ != VariableGroupId::INVALID);
  SpawnLocalStages();
}

void StageGraph::CompleteStage(StageId complete_stage_id, double cycles) {
  CHECK(!IsBlockedStage(complete_stage_id))
      << "Internal error when implementing a barrier!";
  VLOG(1) << "Complete stage " << get_value(complete_stage_id);
  auto iter = uncomplete_stage_map_.find(complete_stage_id);
  CHECK(iter != uncomplete_stage_map_.end());
  auto& stage_record = iter->second;
  const auto& statement_info =
      statement_info_map_->at(stage_record.statement_id);
  // Updates the last finished stage.
  if (last_finished_stage_id_ == StageId::INVALID) {
    last_finished_stage_id_ = complete_stage_id;
  } else {
    last_finished_stage_id_ =
        std::max(last_finished_stage_id_, complete_stage_id);
  }
  // Updates running stats.
  if (self_partition_id_ == PartitionId::FIRST && statement_info.track_needed) {
    timestamp_storage_[complete_stage_id] =
        std::make_pair(stage_record.statement_id,
                       time::timepoint_to_double(time::WallClock::now()));
  }
  UpdateCycleStats(complete_stage_id, stage_record.statement_id, cycles);
  // Updates variable access map, so that later spawned stages do not depend on
  // this complete stage.
  for (const auto& pair : statement_info.variable_access_map) {
    if (pair.second == CanaryApplication::VariableAccess::READ) {
      variable_access_map_[pair.first].read_stages.erase(complete_stage_id);
    } else {
      CHECK(pair.second == CanaryApplication::VariableAccess::WRITE);
      if (variable_access_map_[pair.first].write_stage == complete_stage_id) {
        variable_access_map_[pair.first].write_stage = StageId::INVALID;
      }
    }
  }
  // Triggers following stages.
  for (auto after_stage : stage_record.after_set) {
    if (--uncomplete_stage_map_.at(after_stage).before_set_size == 0) {
      VLOG(1) << "Triggers stage " << get_value(after_stage);
      if (IsBlockedStage(after_stage)) {
        barrier_ready_stage_queue_.insert(after_stage);
      } else {
        ready_stage_queue_.insert(after_stage);
      }
    }
  }
  // Removes the stage.
  uncomplete_stage_map_.erase(iter);
  SpawnLocalStages();
}

std::pair<StageId, StatementId> StageGraph::GetNextReadyStage() {
  SpawnLocalStages();
  if (!ready_stage_queue_.empty()) {
    auto iter = ready_stage_queue_.begin();
    const StageId result_stage_id = *iter;
    ready_stage_queue_.erase(iter);
    return std::make_pair(
        result_stage_id,
        uncomplete_stage_map_.at(result_stage_id).statement_id);
  } else {
    if (HaveReachedBarrierStage()) {
      return std::make_pair(StageId::REACH_BARRIER, StatementId::INVALID);
    } else if (no_more_statement_to_spawn_ && uncomplete_stage_map_.empty()) {
      return std::make_pair(StageId::COMPLETE, StatementId::INVALID);
    } else {
      return std::make_pair(StageId::INVALID, StatementId::INVALID);
    }
  }
}

void StageGraph::FeedControlFlowDecision(StageId stage_id,
                                         bool control_decision) {
  CHECK(received_control_flow_decisions_.find(stage_id) ==
        received_control_flow_decisions_.end());
  received_control_flow_decisions_[stage_id] = control_decision;
  SpawnLocalStages();
}

void StageGraph::retrieve_cycle_stats(
    std::map<StageId, std::pair<StatementId, double>>* cycle_storage_result) {
  cycle_storage_result->swap(cycle_storage_);
  StageId safe_boundary_stage;
  // The stats of a few stage might not be merged correctly, and they should not
  // be sent.
  if (uncomplete_stage_map_.empty()) {
    if (no_more_statement_to_spawn_) {
      // All stats are safe to send.
      return;
    } else {
      safe_boundary_stage = get_next(last_finished_stage_id_);
    }
  } else {
    safe_boundary_stage = uncomplete_stage_map_.begin()->first;
  }
  bool removed = false;
  // the first element s.t. iter->first >= safe_boundary_stage.
  auto iter = cycle_storage_result->lower_bound(safe_boundary_stage);
  while (iter != cycle_storage_result->end()) {
    removed = true;
    cycle_storage_.insert(*iter);
    iter = cycle_storage_result->erase(iter);
  }
  auto back_iter = cycle_storage_result->rbegin();
  if (back_iter != cycle_storage_result->rend() && !removed) {
    // The last stat is unsafe to commit.
    cycle_storage_.insert(*back_iter);
    cycle_storage_result->erase(back_iter->first);
  }
}

bool StageGraph::InsertBarrier(StageId stage_id) {
  CHECK(stage_id >= StageId::FIRST);
  CHECK(next_barrier_stage_id_ == StageId::INVALID)
      << "Barriers cannot be nested!";
  next_barrier_stage_id_ = stage_id;
  CHECK(last_finished_stage_id_ <= stage_id)
      << "Invalid barrier: the execution already passed the barrier stage!";
  CHECK(barrier_ready_stage_queue_.empty());
  auto iter = ready_stage_queue_.begin();
  while (iter != ready_stage_queue_.end()) {
    if (IsBlockedStage(*iter)) {
      barrier_ready_stage_queue_.insert(*iter);
      iter = ready_stage_queue_.erase(iter);
    } else {
      ++iter;
    }
  }
  return true;
}

void StageGraph::ReleaseBarrier() {
  CHECK(next_barrier_stage_id_ != StageId::INVALID) << "No barrier to release.";
  next_barrier_stage_id_ = StageId::INVALID;
  ready_stage_queue_.insert(barrier_ready_stage_queue_.begin(),
                            barrier_ready_stage_queue_.end());
  barrier_ready_stage_queue_.clear();
}

bool StageGraph::IsBlockedStage(StageId stage_id) {
  if (next_barrier_stage_id_ == StageId::INVALID) {
    return false;
  }
  if (stage_id == get_next(next_barrier_stage_id_)) {
    auto statement_id = uncomplete_stage_map_.at(stage_id).statement_id;
    if (statement_info_map_->at(statement_id).statement_type ==
        CanaryApplication::StatementType::GATHER) {
      // The next gathering stage is not blocked.
      return false;
    } else {
      return true;
    }
  }
  return stage_id > next_barrier_stage_id_;
}

bool StageGraph::HaveReachedBarrierStage() {
  if (next_barrier_stage_id_ == StageId::INVALID) {
    return false;
  }
  // If the barrier stage or its following stage has not been spawned, the
  // barrier is not reached.
  if (next_stage_to_spawn_ <= get_next(next_barrier_stage_id_)) {
    return false;
  }
  if (uncomplete_stage_map_.empty()) {
    return true;
  }
  StageId unfinished_stage_id = uncomplete_stage_map_.begin()->first;
  if (unfinished_stage_id == get_next(next_barrier_stage_id_)) {
    auto statement_id =
        uncomplete_stage_map_.at(unfinished_stage_id).statement_id;
    if (statement_info_map_->at(statement_id).statement_type ==
        CanaryApplication::StatementType::GATHER) {
      // The next gathering stage has to be finished so that the barrier is
      // reached.
      return false;
    } else {
      return true;
    }
  }
  return unfinished_stage_id > next_barrier_stage_id_;
}

void StageGraph::UpdateCycleStats(StageId stage_id, StatementId statement_id,
                                  double cycles) {
  auto iter = cycle_storage_.rbegin();
  // Finds the record whose stage id is exactly before the given stage id.
  while (iter != cycle_storage_.rend() && iter->first > stage_id) {
    --iter;
  }
  if (iter != cycle_storage_.rend() &&
      get_distance(iter->first, stage_id) ==
          get_distance(iter->second.first, statement_id)) {
    // Merge the cycles with the former stage.
    iter->second.second += cycles;
  } else {
    cycle_storage_[stage_id] = std::make_pair(statement_id, cycles);
    // The first element s.t. next_iter->first > stage_id.
    auto next_iter = cycle_storage_.upper_bound(stage_id);
    if (next_iter != cycle_storage_.end() &&
        get_distance(stage_id, next_iter->first) ==
            get_distance(statement_id, next_iter->second.first)) {
      // Merge the cycles with the next stage.
      iter->second.second += next_iter->second.second;
      cycle_storage_.erase(next_iter);
    }
  }
}

void StageGraph::SpawnLocalStages() {
  while (uncomplete_stage_map_.size() <= kMaxUncompleteStages &&
         ExamineNextStatement()) {
    continue;
  }
}

bool StageGraph::ExamineNextStatement() {
  // If there is no more statement to spawn, returns false.
  if (no_more_statement_to_spawn_) {
    return false;
  }
  // If there is no more statement to spawn, returns false.
  if (statement_info_map_->find(next_statement_to_spawn_) ==
      statement_info_map_->end()) {
    no_more_statement_to_spawn_ = true;
    return false;
  }
  if (is_blocked_by_control_flow_decision_) {
    auto iter = received_control_flow_decisions_.find(next_stage_to_spawn_);
    if (iter == received_control_flow_decisions_.end()) {
      // If the control flow is blocked, returns false.
      return false;
    }
    const auto& statement_info =
        statement_info_map_->at(next_statement_to_spawn_);
    CHECK(statement_info.statement_type ==
          CanaryApplication::StatementType::WHILE);
    if (iter->second) {
      // Spawns a next while loop.
      ++next_statement_to_spawn_;
    } else {
      // Exits the loop.
      next_statement_to_spawn_ = statement_info.branch_statement;
    }
    ++next_stage_to_spawn_;
    received_control_flow_decisions_.erase(iter);
    is_blocked_by_control_flow_decision_ = false;
    return true;
  }

  const auto& statement_info =
      statement_info_map_->at(next_statement_to_spawn_);
  switch (statement_info.statement_type) {
    case CanaryApplication::StatementType::TRANSFORM:
    case CanaryApplication::StatementType::SCATTER:
    case CanaryApplication::StatementType::GATHER:
      if (self_variable_group_id_ == statement_info.variable_group_id) {
        SpawnStageFromStatement(next_stage_to_spawn_, next_statement_to_spawn_,
                                statement_info);
      }
      ++next_statement_to_spawn_;
      ++next_stage_to_spawn_;
      break;
    case CanaryApplication::StatementType::LOOP:
      // Moves the program pointer only.
      if (is_inside_loop_) {
        ++spawned_loops_;
      } else {
        is_inside_loop_ = true;
        spawned_loops_ = 0;
      }
      if (spawned_loops_ >= statement_info.num_loop) {
        // Looping completes.
        next_statement_to_spawn_ = statement_info.branch_statement;
        is_inside_loop_ = false;
      } else {
        // More loop is needed.
        ++next_statement_to_spawn_;
      }
      ++next_stage_to_spawn_;
      break;
    case CanaryApplication::StatementType::WHILE:
      if (self_variable_group_id_ == statement_info.variable_group_id) {
        SpawnStageFromStatement(next_stage_to_spawn_, next_statement_to_spawn_,
                                statement_info);
      }
      is_blocked_by_control_flow_decision_ = true;
      break;
    case CanaryApplication::StatementType::END_LOOP:
    case CanaryApplication::StatementType::END_WHILE:
      next_statement_to_spawn_ = statement_info.branch_statement;
      break;
    default:
      LOG(FATAL) << "Unknown statement type!";
  }
  return true;
}

void StageGraph::SpawnStageFromStatement(
    StageId stage_id, StatementId statement_id,
    const CanaryApplication::StatementInfo& statement_info) {
  VLOG(1) << "Spawn stage " << get_value(stage_id) << " statement "
          << get_value(statement_id);
  std::set<StageId> before_set;
  for (const auto& pair : statement_info.variable_access_map) {
    if (pair.second == CanaryApplication::VariableAccess::READ) {
      if (variable_access_map_[pair.first].write_stage != StageId::INVALID) {
        before_set.insert(variable_access_map_[pair.first].write_stage);
      }
      variable_access_map_[pair.first].read_stages.insert(stage_id);
    } else {
      CHECK(pair.second == CanaryApplication::VariableAccess::WRITE);
      if (variable_access_map_[pair.first].write_stage != StageId::INVALID) {
        before_set.insert(variable_access_map_[pair.first].write_stage);
      }
      variable_access_map_[pair.first].write_stage = stage_id;
      for (auto read_stage : variable_access_map_[pair.first].read_stages) {
        before_set.insert(read_stage);
      }
      variable_access_map_[pair.first].read_stages.clear();
    }
  }
  auto& stage_record = uncomplete_stage_map_[stage_id];
  stage_record.statement_id = statement_id;
  stage_record.before_set_size = before_set.size();
  for (auto before_stage : before_set) {
    uncomplete_stage_map_.at(before_stage).after_set.insert(stage_id);
  }
  if (stage_record.before_set_size == 0) {
    VLOG(1) << "Self triggers " << get_value(stage_id);
    if (IsBlockedStage(stage_id)) {
      barrier_ready_stage_queue_.insert(stage_id);
    } else {
      ready_stage_queue_.insert(stage_id);
    }
  }
}

}  // namespace canary
