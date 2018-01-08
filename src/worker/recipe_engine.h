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
 * @file src/worker/recipe_engine.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeEngine.
 */

#ifndef CANARY_SRC_WORKER_RECIPE_ENGINE_H_
#define CANARY_SRC_WORKER_RECIPE_ENGINE_H_

#include "worker/execute_engine.h"
#include "worker/recipe.h"
#include "worker/recipe_block.h"
#include "worker/recipe_helper.h"

namespace canary {

/*
 * The interface of an executor that runs a recipe block.
 */
class RecipeBlockExecutor {
 public:
  // Initialize the executor.
  virtual void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) = 0;

  enum class RetrieveResult {
    SUCCESS,
    BLOCKED,
    COMPLETE,
  };
  // Get the next ready stage.
  virtual RetrieveResult GetNextReadyStage(StageId* stage_id,
                                           RecipeId* recipe_id) = 0;
  // Report the completion of a stage, and its computation cycles..
  virtual void CompleteStage(StageId stage_id, double cycles) = 0;
  // Feed a control flow decision.
  virtual void FeedControlFlowDecision(StageId stage_id,
                                       bool control_decision) = 0;
  // Return the next recipe block id.
  virtual RecipeBlockId WrapUp() = 0;
};

/*
 * In progress.
 */
class RecipeBlockExecutorNoneDataDependent : public RecipeBlockExecutor {
 public:
  RecipeBlockExecutorNoneDataDependent() {}
  virtual ~RecipeBlockExecutorNoneDataDependent() {}
  void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) override {
    application_recipes_ = application_recipes;
    current_recipe_block_id_ = recipe_block_id;
    current_recipe_block_ =
        &application_recipes->recipe_block_map.at(recipe_block_id);
    variable_group_id_ = variable_group_id;
    decision_storage_ = decision_storage;
    partition_metadata_storage_ = partition_metadata_storage;
    partition_metadata_storage_before_block_ =
        partition_metadata_storage->Clone();

    begin_stage_id_ = begin_stage_id;
    end_stage_id_ =
        get_next(begin_stage_id, current_recipe_block_->recipe_ids.size());
    // TODO: num_recipes_to_run_ =
    // CountRecipesInVariableGroup(variable_group_id);
    // Fire all recipes.
    for (auto recipe_id : current_recipe_block_->recipe_ids) {
      fired_recipe_ids_.insert(recipe_id);
    }
  }

  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      return RetrieveResult::COMPLETE;
    }
    StageId result_stage_id;
    auto iter = fired_recipe_ids_.begin();
    while (iter != fired_recipe_ids_.end()) {
      if (
          // a ready recipe will be skipped
          ready_recipe_ids_.find(*iter) == ready_recipe_ids_.end() &&
          // check the recipe
          CheckRecipe(*iter, &result_stage_id)) {
        *recipe_id = *iter;
        *stage_id = result_stage_id;
        ready_recipe_ids_.insert(*iter);
        // A ready recipe is still in the fired recipe queue.
        return RetrieveResult::SUCCESS;
      }
      iter = fired_recipe_ids_.erase(iter);
    }
    return RetrieveResult::BLOCKED;
  }
  void CompleteStage(StageId complete_stage_id, double cycles) override {
    CHECK(begin_stage_id_ <= complete_stage_id);
    CHECK(complete_stage_id < end_stage_id_);
    RecipeId recipe_id =
        current_recipe_block_
            ->recipe_ids[get_distance(complete_stage_id, begin_stage_id_)];
    const auto& recipe = application_recipes_->recipe_map.at(recipe_id);
    recipe_helper::ApplyRecipe(recipe, complete_stage_id,
                               partition_metadata_storage_.get());
    ready_recipe_ids_.erase(recipe_id);
    fired_recipe_ids_.insert(recipe_id);
    --num_recipes_to_run_;
  }

  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    decision_storage_->Store(stage_id, control_decision);
  }

  RecipeBlockId WrapUp() override {
    CHECK_EQ(num_recipes_to_run_, 0);
    return current_recipe_block_->next_recipe_block_id;
  }

 protected:
  bool CheckRecipe(RecipeId recipe_id, StageId* stage_id) {
    const auto& recipe = application_recipes_->recipe_map.at(recipe_id);
    int recipe_block_size = current_recipe_block_->recipe_ids.size();
    return recipe_helper::MatchRecipe(
        recipe, begin_stage_id_, end_stage_id_, recipe_block_size,
        *partition_metadata_storage_, *partition_metadata_storage_before_block_,
        stage_id);
  }

 private:
  const ApplicationRecipes* application_recipes_ = nullptr;
  RecipeBlockId current_recipe_block_id_;
  const RecipeBlock* current_recipe_block_ = nullptr;
  VariableGroupId variable_group_id_;
  std::shared_ptr<ControlFlowDecisionStorage> decision_storage_;
  std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage_;
  std::unique_ptr<PartitionMetadataStorage>
      partition_metadata_storage_before_block_;
  int32_t num_recipes_to_run_ = 0;
  // Only the stages in [begin_stage_id_, end_stage_id_) can run.
  StageId begin_stage_id_, end_stage_id_;
  // Recipes that are fired, and may be ready to run.
  std::set<RecipeId> fired_recipe_ids_;
  // Recipes that are ready to run, but might not receive all its messages.
  std::set<RecipeId> ready_recipe_ids_;
};

/* TODO

class RecipeBlockExecutorDataDependent : public RecipeBlockExecutor {
};

class RecipeBlockExecutorDataDependentAndIterative
    : public RecipeBlockExecutor {};

class RecipeBlockExecutorNoneDataDependentFixIterations
    : public RecipeBlockExecutor {};

    */

/*
 * A recipe engine manages the execution state of tasks reading or writing a set
 * of collocated partitions.
 */
class RecipeEngine : public ExecuteEngine {
 public:
  //! Constructor.
  RecipeEngine() {}
  //! Deconstructor.
  virtual ~RecipeEngine() {}

  /*
   * Core functionalities.
   */
  //! Sets statement info map.
  void set_statement_info_map(
      const CanaryApplication::StatementInfoMap* statement_info_map) override {
    statement_info_map_ = statement_info_map;
    // TODO: construct application_recipes from statement_info_map.
  }
  //! Initializes the engine when first launched.
  void Initialize(VariableGroupId self_variable_group_id,
                  PartitionId self_partition_id) override {
    // TODO: Initialize the recipe block executor.
  }
  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id, double cycles) override {
    ongoing_recipe_block_executor_->CompleteStage(complete_stage_id, cycles);
  }

  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage() override {
    StageId stage_id;
    RecipeId recipe_id;
    auto retrieve_result = ongoing_recipe_block_executor_->GetNextReadyStage(
        &stage_id, &recipe_id);
    while (retrieve_result == RecipeBlockExecutor::RetrieveResult::COMPLETE) {
      auto next_recipe_block_id = ongoing_recipe_block_executor_->WrapUp();
      if (next_recipe_block_id ==
          application_recipes_->end_recipe_block_id) {
        return std::make_pair(StageId::COMPLETE, StatementId::INVALID);
      }
      // Initilize the executor with the new block id.
      retrieve_result = ongoing_recipe_block_executor_->GetNextReadyStage(
          &stage_id, &recipe_id);
    }
    switch (retrieve_result) {
      case RecipeBlockExecutor::RetrieveResult::SUCCESS:
        return std::make_pair(
            stage_id,
            application_recipes_->recipe_id_to_statement_id.at(recipe_id));
      case RecipeBlockExecutor::RetrieveResult::BLOCKED:
        return std::make_pair(StageId::INVALID, StatementId::INVALID);
        break;
      default:
        LOG(FATAL) << "Internal error!";
    }
  }
  //! Feeds a control flow decision.
  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    ongoing_recipe_block_executor_->FeedControlFlowDecision(stage_id,
                                                            control_decision);
  }

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
      override {
    // TODO: fill in the stats.
    timestamp_storage->clear();
  }
  //! The cycles for stages.
  void retrieve_cycle_stats(std::map<StageId, std::pair<StatementId, double>>*
                                cycle_storage_result) override {
    // TODO: fill in the stats.
    cycle_storage_result->clear();
  }

  bool InsertBarrier(StageId stage_id) override {
    // TODO: insert a barrier.
    return true;
  }
  void ReleaseBarrier() override {
    // TODO: release a barrier.
  }

 private:
  const CanaryApplication::StatementInfoMap* statement_info_map_;
  const ApplicationRecipes* application_recipes_;
  // There is only one recipe block executor running at one time.
  std::unique_ptr<RecipeBlockExecutor> ongoing_recipe_block_executor_;
};


}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_ENGINE_H_
