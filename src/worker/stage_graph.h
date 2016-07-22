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

#include "shared/canary_application.h"
#include "shared/canary_internal.h"

namespace canary {

/**
 * The stage graph stores spawned stages and tracks when those stages are ready
 * to execute. In the current implementation, each partition saves a seperate
 * copy of the stage graph, for simplicity reasons.
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
  void Initialize(VariableGroupId self_variable_group_id,
                  PartitionId self_partition_id);
  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id, double timestamp,
                     double cycles);
  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage();
  //! Feeds a control flow decision.
  void FeedControlFlowDecision(StageId stage_id, bool control_decision);

  //! Gets the earliest stage that is not finished.
  StageId get_earliest_unfinished_stage_id() {
    if (uncomplete_stage_map_.empty()) {
      return StageId::INVALID;
    } else {
      return uncomplete_stage_map_.begin()->first;
    }
  }
  //! Gets the latest stage that is finished.
  StageId get_last_finished_stage_id() {
    if (no_more_statement_to_spawn_ && uncomplete_stage_map_.empty()) {
      return StageId::COMPLETE;
    } else {
      return last_finished_stage_id_;
    }
  }
  //! The timestamp of critical stages.
  void retrieve_timestamp_stats(
      std::map<StageId, std::pair<StatementId, double>>* timestamp_storage) {
    timestamp_storage->swap(timestamp_storage_);
  }
  //! The cycles for stages.
  void retrieve_cycle_stats(
      std::map<StageId, std::pair<StatementId, double>>* cycle_storage_result);

  /*
   * Controls barrier bahavior.
   */
  bool InsertBarrier(StageId stage_id);
  void ReleaseBarrier();

 private:
  //! Whether a stage needs to be blocked.
  bool IsBlockedStage(StageId stage_id);
  //! Whether the barrier has been reached.
  bool HaveReachedBarrierStage();
  //! Updates cycle stats.
  void UpdateCycleStats(StageId stage_id, StatementId statement_id,
                        double cycles);
  //! Spawns local stages until there are enough uncomplete stages or there are
  // no more statement to execute.
  void SpawnLocalStages();
  //! Examines the next statement, and returns false if no progress can be made.
  bool ExamineNextStatement();
  //! Spawns a stage from a statement.
  void SpawnStageFromStatement(
      StageId stage_id, StatementId statement_id,
      const CanaryApplication::StatementInfo& statement_info);

 public:
  //! Serialization/deserialization.
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(self_variable_group_id_, self_partition_id_);
    archive(uncomplete_stage_map_, ready_stage_queue_);
    archive(variable_access_map_);
    archive(received_control_flow_decisions_);
    archive(next_statement_to_spawn_, next_stage_to_spawn_,
            no_more_statement_to_spawn_);
    archive(is_blocked_by_control_flow_decision_, is_inside_loop_,
            spawned_loops_);
    archive(last_finished_stage_id_, timestamp_storage_, cycle_storage_);
    archive(next_barrier_stage_id_, barrier_ready_stage_queue_);
  }

 private:
  //! The statement info map, which describes the program.
  // Caution: TRANSIENT.
  const CanaryApplication::StatementInfoMap* statement_info_map_ = nullptr;
  //! Maximum uncomplete stages.
  const size_t kMaxUncompleteStages = 10;

  //! Self variable group id.
  VariableGroupId self_variable_group_id_ = VariableGroupId::INVALID;
  //! Self partition id.
  PartitionId self_partition_id_ = PartitionId::INVALID;

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

  //! Running stats.
  StageId last_finished_stage_id_ = StageId::INVALID;
  std::map<StageId, std::pair<StatementId, double>> timestamp_storage_;
  std::map<StageId, std::pair<StatementId, double>> cycle_storage_;

  //! Barrier states.
  StageId next_barrier_stage_id_ = StageId::INVALID;
  std::set<StageId> barrier_ready_stage_queue_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_STAGE_GRAPH_H_
