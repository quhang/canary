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
namespace canary {

class StageGraph {
 public:
  // typedef std::map<VariableId, VariableInfo> VariableInfoMap;
  // typedef std::map<StatementId, StatementInfo> StatementInfoMap;
  // VariableGroupId.
  void Initialize();
  void CompleteStage(StageId stage_id) {}
  StageId NextReadyStage() {
    while (pending_stages < kMaxPendingStages && !no_stage_to_spawn) {
      // Spawn stages.
    }
  }
  void FeedCondition(StageId stage_id, bool control_decision) { SpawnStages(); }

 private:
  void SpawnStages() {
    while (pending_stages < kMaxPendingStages && !no_stage_to_spawn) {
      // Spawn stages.
    }
  }

  const int kMaxPendingStages = 100;
  int pending_stages_ = 0;
  int ready_stages_ = 0;
  bool no_stage_to_spawn_ = false;

  std::set<StageId> ready_stage_queue_;

  std::map<StageId, StageRecord> uncompleted_stage_map_;
  struct StageRecord {
    int before_set_size = 0;
    std::set<StageId> after_set;
  };
  std::map<VariableId, VariableRecord> variable_access_map_;
  struct VariableRecord {
    std::set<StageId> read_stages;
    StageId write_set = StageId::INVALID;
  };

  StatementId next_statement_to_spawn_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_STAGE_GRAPH_H_
