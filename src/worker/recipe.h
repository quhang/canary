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
 * @file src/worker/recipe.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class Recipe.
 */

#ifndef CANARY_SRC_WORKER_RECIPE_H_
#define CANARY_SRC_WORKER_RECIPE_H_

#include <map>
#include <set>
#include <utility>

#include "shared/canary_application.h"
#include "shared/canary_internal.h"

#include "worker/execute_engine.h"

namespace canary {

struct AccessRequirement {
  VariableId variable_id;
  enum class AccessType { READ, WRITE } access_type;
  // The stage id is offset relative to the beginning of its recipe block.
  StageId last_write_stage_id_offset;
  int32_t num_reading_stages;
  // If what are executed before the recipe are data-dependent, the access
  // record has to be dynamically adjusted. In that case,
  // last_write_stage_id_offset is invalid, and num_reading_stages may need
  // adjustment as well.
  bool need_dynamic_adjustment;
};

struct Recipe {
  RecipeId recipe_id;
  // The stage id is offset relative to the beginning of its recipe block.
  StageId current_stage_id_offset;
  std::map<VariableId, AccessRequirement> variable_id_to_access_requirement;
  std::set<RecipeId> recipe_ids_to_fire;
};

// A recipe block contains a block of recipes to execute, while only the last
// recipe in the block may return a boolean value that determines what recipe
// block to run next.
struct RecipeBlock {
  RecipeBlockId recipe_block_id;
  std::list<RecipeId> recipe_ids;
  enum class RecipeBlockType {
    // The last recipe in the block determines what block to run next.
    DATA_DEPENDENT,
    // The next block to run is deterministic.
    NONE_DATA_DEPENDENT,
    // The block runs a fixed number of iterations.
    FIXED_ITERATIONS
  } recipe_block_type;

  // Valid for DATA_DEPENDENT.
  bool is_iterative;
  RecipeBlockId next_recipe_block_id_if_true;
  RecipeBlockId next_recipe_block_id_if_false;
  // Valid for NONE_DATA_DEPENDENT and FIXED_ITERATIONS.
  RecipeBlockId next_recipe_block_id;
  // Valid for FIXED_ITERATIONS.
  int32_t num_iterations;
};

struct ApplicationRecipes {
  std::map<RecipeId, Recipe> recipe_map;
  std::map<RecipeBlockId, RecipeBlock> recipe_block_map;
  RecipeBlockId begin_recipe_block_id, end_recipe_block_id;
  std::map<RecipeId, StatementId> recipe_id_to_statement_id;
};

class RecipeEngine : public ExecutionEngine {
 public:
  //! Constructor.
  RecipeEngine() {}
  //! Deconstructor.
  ~RecipeEngine() override {}

  /*
   * Core functionalities.
   */
  //! Sets statement info map.
  void set_statement_info_map(
      const CanaryApplication::StatementInfoMap* statement_info_map) override {}
  //! Initializes the stage graph when first launched.
  void Initialize(VariableGroupId self_variable_group_id,
                  PartitionId self_partition_id) override {}
  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id, double cycles) override {}
  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage() override {}
  //! Feeds a control flow decision.
  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {}

  /*
   * Migration related.
   */
  //! Deserialization function.
  void load(CanaryInputArchive& archive) override {}  // NOLINT
  //! Serialization function. Data are destroyed after serialization.
  void save(CanaryOutputArchive& archive) const override {}  // NOLINT

  /*
   * Debugging and monitoring related.
   */
  //! Gets the earliest stage that is not finished.
  StageId get_earliest_unfinished_stage_id() override {}
  //! Gets the latest stage that is finished.
  StageId get_last_finished_stage_id() override {}
  //! The timestamp of critical stages.
  void retrieve_timestamp_stats(
      std::map<StageId, std::pair<StatementId, double>>* timestamp_storage)
      override {}
  //! The cycles for stages.
  void retrieve_cycle_stats(std::map<StageId, std::pair<StatementId, double>>*
                                cycle_storage_result) override {}

  bool InsertBarrier(StageId stage_id) override {}
  void ReleaseBarrier() override {}

 protected:
  void InitializeRecipeBlock(RecipeBlockId recipe_block_id) {
    if (recipe_block_id == application_recipes_->end_recipe_block_id) {
      // FINISHED.
    }
    current_recipe_block_ =
        &application_recipes_->recipe_block_map.at[recipe_block_id];
    begin_stage_id_ = end_stage_id_;
    end_stage_id_ = begin_stage_id_ +
                    current_recipe_block_.recipe_ids.size() *
                        (current_recipe_block_->recipe_block_type ==
                                 RecipeBlock::RecipeBlockType::FIXED_ITERATIONS
                             ? current_recipe_block_->num_iterations
                             : 1);
    // Grow the end_stage_id_ using unused_control_flow_decisions_
    // Fire all recipes that are on the variable_group_id.
  }

  StageId GetNextReadyRecipe() {
    for (auto recipe_id : fired_recipe_ids_) {
      if (auto stage_id = CheckRecipe(recipe_id)) {
        // The recipe is not removed from the fired set.
        return stage_id;
      }
    }
    // Return some meaningful value.
    return -1;
  }

  void FireRecipe(RecipeId recipe_id) { fired_recipe_ids_.insert(recipe_id); }

  StageId CheckRecipe(RecipeId recipe_id) {
    const auto& recipe = application_recipes_->recipe_map.at(recipe_id);
    // recipe.current_stage_id_offset.
    const int step_size = current_recipe_block_.recipe_ids.size();
    bool matched = true;
    int32_t iteration_num = -1;
    for (const auto& key_value : variable_id_to_access_requirement) {
      const auto& access_requirement = key_value.second;
      const auto& partition_access_record =
          variable_id_to_partition_access_record_[access_requirement.first];
      // TODO
      CHECK(!access_requirment.need_dynamic_adjustment);
      auto iteration_num = (partition_access_record.last_write_stage_id -
                            access_requirment.last_write_stage_id_offset) /
                           step_size;
    }
    return iteration_num * step_size + recipe.current_stage_id_offset;
  }

  void CompleteRecipe(RecipeId recipe_id, StageId stage_id) {
    // Update variables, and fire recipes.
  }

 private:
  struct PartitionAccessRecord {
    StageId last_write_stage_id;
    int32_t num_reading_stages;
  };

  const RecipeBlock* current_recipe_block_;
  int32_t num_recipes_to_complete_;
  const ApplicationRecipes* application_recipes_;
  // Only the stages in [begin_stage_id_, end_stage_id_) can run.
  StageId begin_stage_id_, end_stage_id_;
  std::map<StageId, bool> unused_control_flow_decisions_;
  std::set<RecipeId> fired_recipes_;

  // Only stages in this range can run.
  StageId begin_stage_id_, end_stage_id_;
  int recipes_to_complete;
  RecipeBlockId next_block_to_run;
  std::set<RecipeId> active_recipes;
  std::map<VariableId, PartitionAccessRecord>
      variable_id_to_partition_access_record_before_block_;
  std::map<VariableId, PartitionAccessRecord>
      variable_id_to_partition_access_record_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_H_
