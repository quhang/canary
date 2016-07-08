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
 * @file src/worker/stage_graph.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class StageGraph.
 */

#ifndef CANARY_SRC_WORKER_STAGE_GRAPH_H_
#define CANARY_SRC_WORKER_STAGE_GRAPH_H_

#include <map>
#include <set>
#include <utility>

#include "shared/canary_internal.h"
#include "shared/canary_application.h"

namespace canary {

/**
 * The stage graph belonged to a partition, recording spawned stages and
 * tracking whether those stages are ready to execute.
 */
class StageGraph {
 public:
  //! Record of a stage.
  struct StageRecord {
    StatementId statement_id = StatementId::INVALID;
    int before_set_size = 0;
    std::set<StageId> after_set;
    template <typename Archive>
    void serialize(Archive& archive) {  // NOLINT
      archive(statement_id, before_set_size, after_set);
    }
  };

  //! Record of a variable.
  struct VariableRecord {
    std::set<StageId> read_stages;
    StageId write_stage = StageId::INVALID;
    template <typename Archive>
    void serialize(Archive& archive) {  // NOLINT
      archive(read_stages, write_stage);
    }
  };

 public:
  //! Constructor.
  StageGraph() {}
  //! Deconstructor.
  virtual ~StageGraph() {}

  //! Sets statement info map.
  void set_statement_info_map(
      const CanaryApplication::StatementInfoMap* statement_info_map) {
    statement_info_map_ = statement_info_map;
  }

  //! Initializes the stage graph when first launched.
  void Initialize(VariableGroupId self_variable_group_id) {
    CHECK_NOTNULL(statement_info_map_);
    self_variable_group_id_ = self_variable_group_id;
    CHECK(self_variable_group_id_ != VariableGroupId::INVALID);
    SpawnLocalStages();
  }

  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id) {
    auto iter = uncomplete_stage_map_.find(complete_stage_id);
    CHECK(iter != uncomplete_stage_map_.end());
    auto& stage_record = iter->second;
    const auto& statement_info =
        statement_info_map_->at(stage_record.statement_id);
    // Updates variable access map.
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
        ready_stage_queue_.insert(after_stage);
      }
    }
    SpawnLocalStages();
  }

  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage() {
    SpawnLocalStages();
    if (!ready_stage_queue_.empty()) {
      auto iter = ready_stage_queue_.begin();
      const StageId result_stage_id = *iter;
      ready_stage_queue_.erase(iter);
      return std::make_pair(
          result_stage_id,
          uncomplete_stage_map_.at(result_stage_id).statement_id);
    } else {
      if (no_more_statement_to_spawn_ && uncomplete_stage_map_.empty()) {
        return std::make_pair(StageId::COMPLETE, StatementId::INVALID);
      } else {
        return std::make_pair(StageId::INVALID, StatementId::INVALID);
      }
    }
  }

  //! Feeds a control flow decision.
  void FeedControlFlowDecision(StageId stage_id, bool control_decision) {
    CHECK(received_control_flow_decisions_.find(stage_id) ==
          received_control_flow_decisions_.end());
    received_control_flow_decisions_[stage_id] = control_decision;
    SpawnLocalStages();
  }

 private:
  //! Spawns local stages until there are enough uncomplete stages or there are
  // no more statement to execute.
  void SpawnLocalStages() {
    while (uncomplete_stage_map_.size() < kMaxUncompleteStages &&
           ExamineNextStatement()) {
      continue;
    }
  }

  //! Examines the next statement, and returns false if no progress can be made.
  bool ExamineNextStatement() {
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
          SpawnStageFromStatement(next_stage_to_spawn_,
                                  next_statement_to_spawn_, statement_info);
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
      case CanaryApplication::StatementType::END_LOOP:
        next_statement_to_spawn_ = statement_info.branch_statement;
        break;
      case CanaryApplication::StatementType::WHILE:
        if (self_variable_group_id_ == statement_info.variable_group_id) {
          SpawnStageFromStatement(next_stage_to_spawn_,
                                  next_statement_to_spawn_, statement_info);
        }
        is_blocked_by_control_flow_decision_ = true;
        break;
      case CanaryApplication::StatementType::END_WHILE:
        next_statement_to_spawn_ = statement_info.branch_statement;
        break;
      default:
        LOG(FATAL) << "Unknown statement type!";
    }
    return true;
  }

  //! Spawns a stage from a statement.
  void SpawnStageFromStatement(
      StageId stage_id, StatementId statement_id,
      const CanaryApplication::StatementInfo& statement_info) {
    std::set<StageId> before_set;
    for (const auto& pair : statement_info.variable_access_map) {
      if (pair.second == CanaryApplication::VariableAccess::READ) {
        if (variable_access_map_[pair.first].write_stage != StageId::INVALID) {
          before_set.insert(variable_access_map_[pair.first].write_stage);
        }
      } else {
        CHECK(pair.second == CanaryApplication::VariableAccess::WRITE);
        if (variable_access_map_[pair.first].write_stage != StageId::INVALID) {
          before_set.insert(variable_access_map_[pair.first].write_stage);
        }
        for (auto read_stage : variable_access_map_[pair.first].read_stages) {
          before_set.insert(read_stage);
        }
      }
    }

    auto& stage_record = uncomplete_stage_map_[stage_id];
    stage_record.statement_id = statement_id;
    stage_record.before_set_size = before_set.size();
    for (auto before_stage : before_set) {
      uncomplete_stage_map_.at(before_stage).after_set.insert(stage_id);
    }
    if (stage_record.before_set_size == 0) {
      ready_stage_queue_.insert(stage_id);
    }
  }

  //! The statement info map, which describes the program.
  const CanaryApplication::StatementInfoMap* statement_info_map_ = nullptr;

  //! Maximum uncomplete stages.
  const size_t kMaxUncompleteStages = 10;

  //! Self variable group id.
  VariableGroupId self_variable_group_id_ = VariableGroupId::INVALID;

  //! Spawned but uncomplete stages.
  std::map<StageId, StageRecord> uncomplete_stage_map_;
  //! Ready stages to be executed.
  std::set<StageId> ready_stage_queue_;

  //! Records the last stage that reads/writes a variable.
  std::map<VariableId, VariableRecord> variable_access_map_;

  //! Received control flow decisions.
  std::map<StageId, bool> received_control_flow_decisions_;

  //! Next statement to spawn.
  StatementId next_statement_to_spawn_ = StatementId::FIRST;
  //! Next stage to spawn.
  StageId next_stage_to_spawn_ = StageId::FIRST;
  //! No more statement to spawn.
  bool no_more_statement_to_spawn_ = false;

  //! Control flow states.
  bool is_blocked_by_control_flow_decision_ = false;
  bool is_inside_loop_ = false;
  int spawned_loops_ = 0;

  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(self_variable_group_id_);
    archive(uncomplete_stage_map_, ready_stage_queue_);
    archive(variable_access_map_);
    archive(received_control_flow_decisions_);
    archive(next_statement_to_spawn_, next_stage_to_spawn_,
            no_more_statement_to_spawn_);
    archive(is_blocked_by_control_flow_decision_, is_inside_loop_,
            spawned_loops_);
  }
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_STAGE_GRAPH_H_
