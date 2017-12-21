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
  FetchResult FetchNext(StageId* stage_id, bool* decision) {
    return FetchInternal(stage_id, decision, true);
  }
  FetchResult ExamineNext(StageId* stage_id, bool* decision) {
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

class PartitionMetadataStorage {
 public:
  PartitionMetadataStorage() {}
  virtual ~PartitionMetadataStorage() {}
  NON_COPYABLE_AND_NON_MOVABLE(ControlFlowDecisionStorage);
  PartitionMetadataStorage* Clone() const {
    // TODO.
  }

  struct AccessMetadata {
    StageId last_write_stage_id;
    int32_t num_read_stages;
  };
  AccessMetadata RetrieveAccessMetadata(VariableId variable_id) {
    return variable_id_to_access_metadata_.at(variable_id);
  }
  void ApplyRecipe(const Recipe* recipe, StageId stage_id,
                   const PartitionMetadataStorage* metadata_before_block) {
    // TODO.
  }

 private:
  std::map<VariableId, AccessMetadata> variable_id_to_access_metadata_;
};

/*
 * The interface of an executor that runs a recipe block.
 */
class RecipeBlockExecutor {
 public:
  // Initialize the executor.
  virtual void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
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
    current_recipe_block_ = application_recipes->recipe_block_map.at[recipe_block_id];
    variable_group_id_ = variable_group_id;
    decision_storage_ = decision_storage;
    partition_metadata_storage_ = partition_metadata_storage;

    begin_stage_id_ = begin_stage_id;
    end_stage_id_ = begin_stage_id + current_recipe_block_.recipe_ids.size();
    num_recipes_to_run_ = current_recipe_block_->CountRecipesInVariableGroup(variable_group_id);
    // Fire all recipes.
    for (auto recipe_id : current_recipe_block_.recipe_ids) {
      FireRecipe(recipe_id);
    }
  }
  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      return RetrieveResult::COMPLETE;
    }
    StageId result_stage_id;
    auto iter = fired_recipes_.begin();
    while (iter != fired_recipes_.end()) {
      if (ready_recipes_.find(*iter) == ready_recipes_.end() &&
          CheckRecipe(*iter, &result_stage_id)) {
        *recipe_id = *iter;
        *stage_id = result_stage_id;
        ready_recipes_.insert(*recipe_id);
        return RetrieveResult::SUCCESS;
      }
      iter = fired_recipes_.erase(iter);
    }
    return RetrieveResult::BLOCKED;
  }
  void CompleteStage(StageId complete_stage_id, double cycles) override {
    CHECK_LE(begin_stage_id_, complete_stage_id);
    CHECK_LT(complete_stage_id, end_stage_id_);
    RecipeId recipe_id =
        current_recipe_block_->recipe_ids_[complete_stage_id - begin_stage_id_];
    // TODO.
    // Update the partition storage.
    ready_recipes_.erase(recipe_id);
    fired_recipes_.insert(recipe_id);
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
  void FireRecipe(RecipeId recipe_id) { fired_recipe_ids_.insert(recipe_id); }

  bool CheckRecipe(RecipeId recipe_id, StageId* stage_id) {
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

 private:
  const ApplicationRecipes* application_recipes_ = nullptr;
  RecipeBlockId current_recipe_block_id_;
  const RecipeBlock* current_recipe_block_ = nullptr;
  VariableGroupId variable_group_id_;
  std::shared_ptr<ControlFlowDecisionStorage> decision_storage_;
  std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage_;
  int32_t num_recipes_to_run_ = 0;
  // Only the stages in [begin_stage_id_, end_stage_id_) can run.
  StageId begin_stage_id_, end_stage_id_;
  std::set<RecipeId> fired_recipes_, ready_recipes_;
};

class RecipeBlockExecutorDataDependent : public RecipeBlockExecutor {
};

class RecipeBlockExecutorDataDependentAndIterative
    : public RecipeBlockExecutor {};

class RecipeBlockExecutorNoneDataDependentFixIterations
    : public RecipeBlockExecutor {};



/*
 * A recipe engine manages the execution state of tasks reading or writing a set
 * of collocated partitions.
 */
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
      const CanaryApplication::StatementInfoMap* statement_info_map) override {
    statement_info_map_ = statement_info_map;
    // TODO: construct application_recipes from statement_info_map.
  }
  //! Initializes the stage graph when first launched.
  void Initialize(VariableGroupId self_variable_group_id,
                  PartitionId self_partition_id) override {
    // TODO: Initialize the recipe block executor.
  }
  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id, double cycles) override {}
  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage() override {
    // TODO: forward to the recipe block executor, and replace the executor when
    // it is done.
  }
  //! Feeds a control flow decision.
  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    // TODO: forward it to the recipe block executor.
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
      override {}
  //! The cycles for stages.
  void retrieve_cycle_stats(std::map<StageId, std::pair<StatementId, double>>*
                                cycle_storage_result) override {}

  bool InsertBarrier(StageId stage_id) override {}
  void ReleaseBarrier() override {}

 private:
  const CanaryApplication::StatementInfoMap* statement_info_map_;
  const ApplicationRecipes* application_recipes_;
  // There is only one recipe block executor running at one time.
  std::unique_ptr<RecipeBlockExecutor> ongoing_recipe_block_executor_;
  const ApplicationRecipes* application_recipes_;
};


}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_ENGINE_H_
