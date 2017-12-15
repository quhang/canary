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
 * @file src/worker/execute_engine.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ExecuteEngine.
 */

#ifndef CANARY_SRC_WORKER_EXECUTE_ENGINE_H_
#define CANARY_SRC_WORKER_EXECUTE_ENGINE_H_

#include <map>
#include <set>
#include <utility>

#include "shared/canary_application.h"
#include "shared/canary_internal.h"

namespace canary {

class ExecuteEngine {
 public:
  //! Constructor.
  ExecuteEngine() {}
  //! Deconstructor.
  virtual ~ExecuteEngine() {}

  //! Sets statement info map.
  virtual void set_statement_info_map(
      const CanaryApplication::StatementInfoMap* statement_info_map) = 0;
  //! Initializes the stage graph when first launched.
  virtual void Initialize(VariableGroupId self_variable_group_id,
                          PartitionId self_partition_id) = 0;
  //! Reports complete stage.
  virtual void CompleteStage(StageId complete_stage_id, double cycles) = 0;
  //! Gets the next ready stage.
  virtual std::pair<StageId, StatementId> GetNextReadyStage() = 0;
  //! Feeds a control flow decision.
  virtual void FeedControlFlowDecision(StageId stage_id,
                                       bool control_decision) = 0;

  //! Gets the earliest stage that is not finished.
  virtual StageId get_earliest_unfinished_stage_id() = 0;
  //! Gets the latest stage that is finished.
  virtual StageId get_last_finished_stage_id() = 0;
  //! The timestamp of critical stages.
  virtual void retrieve_timestamp_stats(
      std::map<StageId, std::pair<StatementId, double>>* timestamp_storage) = 0;
  //! The cycles for stages.
  virtual void retrieve_cycle_stats(
      std::map<StageId, std::pair<StatementId, double>>*
          cycle_storage_result) = 0;

  /*
   * Controls barrier bahavior.
   */
  virtual bool InsertBarrier(StageId stage_id) = 0;
  virtual void ReleaseBarrier() = 0;

  //! Deserialization function.
  virtual void load(CanaryInputArchive& archive) = 0;  // NOLINT
  //! Serialization function. Data are destroyed after serialization.
  virtual void save(CanaryOutputArchive& archive) const = 0;  // NOLINT
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_EXECUTE_ENGINE_H_
