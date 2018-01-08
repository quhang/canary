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
 * @file src/worker/recipe_helper.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeHelper.
 */

#ifndef CANARY_SRC_WORKER_RECIPE_HELPER_H_
#define CANARY_SRC_WORKER_RECIPE_HELPER_H_

#include "shared/canary_application.h"
#include "shared/canary_internal.h"

#include "worker/recipe.h"
#include "worker/recipe_block.h"

namespace canary {

/*
 * Store control flow decisions that are not consumed yet.
 */
class ControlFlowDecisionStorage {
 public:
  ControlFlowDecisionStorage() {}
  virtual ~ControlFlowDecisionStorage() {}
  NON_COPYABLE_AND_NON_MOVABLE(ControlFlowDecisionStorage);

  void Store(StageId stage_id, bool decision) {
    internal_storage_[stage_id] = decision;
  }
  enum class FetchResult { SUCCESS, NONE };
  FetchResult PopNext(StageId* stage_id, bool* decision) {
    return FetchInternal(stage_id, decision, true);
  }
  FetchResult PeekNext(StageId* stage_id, bool* decision) {
    return FetchInternal(stage_id, decision, false);
  }

 private:
  FetchResult FetchInternal(StageId* stage_id, bool* decision,
                            bool remove_next_entry) {
    if (internal_storage_.empty()) {
      return FetchResult::NONE;
    }
    auto result = internal_storage_.begin();
    if (stage_id) {
      *stage_id = result->first;
    }
    if (decision) {
      *decision = result->second;
    }
    if (remove_next_entry) {
      internal_storage_.erase(result);
    }
    return FetchResult::SUCCESS;
  }

  std::map<StageId, bool> internal_storage_;
};

/*
 * Store the access metadata of a set of collocated partitions.
 */
class PartitionMetadataStorage {
 public:
  PartitionMetadataStorage() {}
  virtual ~PartitionMetadataStorage() {}
  NON_COPYABLE_AND_NON_MOVABLE(PartitionMetadataStorage);

  std::unique_ptr<PartitionMetadataStorage> Clone() const {
    auto result = std::make_unique<PartitionMetadataStorage>();
    result->variable_id_to_access_metadata_ = variable_id_to_access_metadata_;
    return std::move(result);
  }

  struct AccessMetadata {
    StageId last_write_stage_id;
    int32_t num_read_stages;
  };
  AccessMetadata GetPartitionAccessMetadata(VariableId variable_id) const {
    // Throw exception if the access metadata is not available.
    return variable_id_to_access_metadata_.at(variable_id);
  }
  void InitializePartition(VariableId variable_id, StageId last_write_stage_id,
                           int num_read_stages) {
    auto& access = variable_id_to_access_metadata_[variable_id];
    access.last_write_stage_id = last_write_stage_id;
    access.num_read_stages = num_read_stages;
  }
  void WritePartition(VariableId variable_id, StageId last_write_stage_id) {
    auto& access = variable_id_to_access_metadata_.at(variable_id);
    access.last_write_stage_id = last_write_stage_id;
    access.num_read_stages = 0;
  }
  void ReadPartition(VariableId variable_id) {
    auto& access = variable_id_to_access_metadata_.at(variable_id);
    ++access.num_read_stages;
  }

 private:
  std::map<VariableId, AccessMetadata> variable_id_to_access_metadata_;
};

namespace recipe_helper {
/*
 * Match the recipe. Return whether there is a match, and the stage id of the
 * match.
 */
bool MatchRecipe(
    const Recipe& recipe, StageId begin_stage_id, StageId end_stage_id,
    int step_size, const PartitionMetadataStorage& partition_metadata_storage,
    const PartitionMetadataStorage& partition_metadata_storage_before_block,
    StageId* result_stage_id);

// Apply the recipe, whose stage id is "stage_id", to edit the partition
// metadata.
void ApplyRecipe(const Recipe& recipe, StageId stage_id,
                 PartitionMetadataStorage* partition_metadata_storage);

}  // namespace recipe_helper

}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_HELPER_H_
