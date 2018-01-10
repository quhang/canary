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
  // Factory method.
  static std::unique_ptr<RecipeBlockExecutor> Create(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id, VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id);

  RecipeBlockExecutor() {}
  virtual ~RecipeBlockExecutor() {}

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
  virtual void CompleteStage(StageId stage_id, double cycles) {
    CHECK(begin_stage_id_ <= stage_id);
    CHECK(stage_id < end_stage_id_);
    RecipeId recipe_id =
        current_recipe_block_
            ->recipe_ids[get_distance(begin_stage_id_, stage_id) %
                         current_recipe_block_->recipe_ids.size()];
    CompleteRecipe(recipe_id, stage_id);
  }
  // Feed a control flow decision.
  virtual void FeedControlFlowDecision(StageId stage_id,
                                       bool control_decision) = 0;
  // Wrap up the state.
  virtual void WrapUp(RecipeBlockId* next_recipe_block_id,
              StageId* next_stage_id) = 0;

 protected:
  /*
   * The protected methods are for the child to use.
   */
  // Initialize internal common data structures.
  void InitializeInternal(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id, VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage);

 protected:
  const ApplicationRecipes* application_recipes_ = nullptr;
  RecipeBlockId current_recipe_block_id_;
  const RecipeBlock* current_recipe_block_ = nullptr;
  VariableGroupId variable_group_id_;
  // Metadata storage.
  std::shared_ptr<ControlFlowDecisionStorage> decision_storage_;
  std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage_;
  std::unique_ptr<PartitionMetadataStorage>
      partition_metadata_storage_before_block_;
  // Recipe counts.
  int32_t num_recipes_in_one_block_, num_recipes_to_run_;
  // Only the stages in [begin_stage_id_, end_stage_id_) can run.
  StageId begin_stage_id_, end_stage_id_;
  // Recipes that are fired, and may be ready to run.
  std::set<RecipeId> fired_recipe_ids_;
  // Recipes that are ready to run, but might not receive all its messages.
  std::set<RecipeId> ready_recipe_ids_;

 protected:
  // Check whether a recipe can run.
  bool CheckRecipe(RecipeId recipe_id, StageId* stage_id);
  // Retrieve a ready stage.
  bool RetrieveReadyStageIfPossible(StageId* stage_id, RecipeId* recipe_id);
  // Complete a recipe.
  void CompleteRecipe(RecipeId recipe_id, StageId complete_stage_id);
};

class RecipeBlockExecutorNoneDataDependent : public RecipeBlockExecutor {
 public:
  RecipeBlockExecutorNoneDataDependent() {}
  ~RecipeBlockExecutorNoneDataDependent() override {}
  void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) override {
    InitializeInternal(application_recipes, recipe_block_id, variable_group_id,
                       decision_storage, partition_metadata_storage);
    begin_stage_id_ = begin_stage_id;
    end_stage_id_ =
        get_next(begin_stage_id, current_recipe_block_->recipe_ids.size());
    num_recipes_to_run_ = num_recipes_in_one_block_;
  }

  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      return RetrieveResult::COMPLETE;
    }
    if (RetrieveReadyStageIfPossible(stage_id, recipe_id)) {
      return RetrieveResult::SUCCESS;
    } else {
      return RetrieveResult::BLOCKED;
    }
  }

  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    decision_storage_->Store(stage_id, control_decision);
  }

  void WrapUp(RecipeBlockId* next_recipe_block_id,
              StageId* next_stage_id) override {
    CHECK_EQ(num_recipes_to_run_, 0);
    *next_recipe_block_id = current_recipe_block_->next_recipe_block_id;
    *next_stage_id = end_stage_id_;
  }
};

class RecipeBlockExecutorFixedIterations : public RecipeBlockExecutor {
 public:
  RecipeBlockExecutorFixedIterations() {}
  ~RecipeBlockExecutorFixedIterations() override {}
  void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) override {
    InitializeInternal(application_recipes, recipe_block_id, variable_group_id,
                       decision_storage, partition_metadata_storage);
    begin_stage_id_ = begin_stage_id;
    end_stage_id_ =
        get_next(begin_stage_id, current_recipe_block_->num_iterations *
                                     current_recipe_block_->recipe_ids.size());
    num_recipes_to_run_ =
        num_recipes_in_one_block_ * current_recipe_block_->num_iterations;
  }

  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      return RetrieveResult::COMPLETE;
    }
    if (RetrieveReadyStageIfPossible(stage_id, recipe_id)) {
      return RetrieveResult::SUCCESS;
    } else {
      return RetrieveResult::BLOCKED;
    }
  }

  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    decision_storage_->Store(stage_id, control_decision);
  }

  void WrapUp(RecipeBlockId* next_recipe_block_id,
              StageId* next_stage_id) override {
    CHECK_EQ(num_recipes_to_run_, 0);
    *next_recipe_block_id = current_recipe_block_->next_recipe_block_id;
    *next_stage_id = end_stage_id_;
  }
};

class RecipeBlockExecutorDataDependent : public RecipeBlockExecutor {
 public:
  RecipeBlockExecutorDataDependent() {}
  ~RecipeBlockExecutorDataDependent() override {}
  void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) override {
    InitializeInternal(application_recipes, recipe_block_id, variable_group_id,
                       decision_storage, partition_metadata_storage);
    begin_stage_id_ = begin_stage_id;
    end_stage_id_ =
        get_next(begin_stage_id, current_recipe_block_->recipe_ids.size());
    num_recipes_to_run_ = num_recipes_in_one_block_;

    TryFillInControlDecision();
  }

  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      if (control_decision_received_) {
        return RetrieveResult::COMPLETE;
      } else {
        return RetrieveResult::BLOCKED;
      }
    }
    if (RetrieveReadyStageIfPossible(stage_id, recipe_id)) {
      return RetrieveResult::SUCCESS;
    } else {
      return RetrieveResult::BLOCKED;
    }
  }

  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    decision_storage_->Store(stage_id, control_decision);
    TryFillInControlDecision();
  }

  void WrapUp(RecipeBlockId* next_recipe_block_id,
              StageId* next_stage_id) override {
    CHECK_EQ(num_recipes_to_run_, 0);
    CHECK(control_decision_received_);
    *next_recipe_block_id = next_recipe_block_id_;
    *next_stage_id = end_stage_id_;
  }

 private:
  bool TryFillInControlDecision() {
    if (!control_decision_received_ &&
        decision_storage_->Query(get_prev(end_stage_id_))) {
      StageId control_stage_id;
      bool control_decision;
      CHECK(decision_storage_->PopNext(&control_stage_id, &control_decision) ==
            ControlFlowDecisionStorage::FetchResult::SUCCESS);
      CHECK(control_stage_id == get_prev(end_stage_id_));
      control_decision_received_ = true;
      next_recipe_block_id_ =
          (control_decision
               ? current_recipe_block_->next_recipe_block_id_if_true
               : current_recipe_block_->next_recipe_block_id_if_false);
      return true;
    }
    return false;
  }
  bool control_decision_received_ = false;
  RecipeBlockId next_recipe_block_id_;
};

class RecipeBlockExecutorDataDependentAndIterative
    : public RecipeBlockExecutor {
 public:
  RecipeBlockExecutorDataDependentAndIterative() {}
  ~RecipeBlockExecutorDataDependentAndIterative() override {}
  void Initialize(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id, VariableGroupId variable_group_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
      StageId begin_stage_id) override {
    InitializeInternal(application_recipes, recipe_block_id, variable_group_id,
                       decision_storage, partition_metadata_storage);
    begin_stage_id_ = begin_stage_id;
    end_stage_id_ =
        get_next(begin_stage_id, current_recipe_block_->recipe_ids.size());
    num_recipes_to_run_ = num_recipes_in_one_block_;

    TryFillInControlDecision();
  }

  RetrieveResult GetNextReadyStage(StageId* stage_id,
                                   RecipeId* recipe_id) override {
    if (num_recipes_to_run_ == 0) {
      if (control_decision_received_) {
        return RetrieveResult::COMPLETE;
      } else {
        return RetrieveResult::BLOCKED;
      }
    }
    if (RetrieveReadyStageIfPossible(stage_id, recipe_id)) {
      return RetrieveResult::SUCCESS;
    } else {
      return RetrieveResult::BLOCKED;
    }
  }

  void FeedControlFlowDecision(StageId stage_id,
                               bool control_decision) override {
    decision_storage_->Store(stage_id, control_decision);
    TryFillInControlDecision();
  }

  void WrapUp(RecipeBlockId* next_recipe_block_id,
              StageId* next_stage_id) override {
    CHECK_EQ(num_recipes_to_run_, 0);
    CHECK(control_decision_received_);
    *next_recipe_block_id = current_recipe_block_->next_recipe_block_id;
    *next_stage_id = end_stage_id_;
  }

 private:
  bool TryFillInControlDecision() {
    while (!control_decision_received_ &&
           decision_storage_->Query(get_prev(end_stage_id_))) {
      StageId control_stage_id;
      bool control_decision;
      CHECK(decision_storage_->PopNext(&control_stage_id, &control_decision) ==
            ControlFlowDecisionStorage::FetchResult::SUCCESS);
      CHECK(control_stage_id == get_prev(end_stage_id_));
      if (control_decision) {
        num_recipes_to_run_ += num_recipes_in_one_block_;
        end_stage_id_ =
            get_next(end_stage_id_, current_recipe_block_->recipe_ids.size());
      } else {
        control_decision_received_ = true;
        return true;
      }
    }
    return false;
  }
  bool control_decision_received_ = false;
};

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
      const CanaryApplication::StatementInfoMap* statement_info_map) override;
  //! Initializes the engine when first launched.
  void Initialize(VariableGroupId self_variable_group_id,
                  PartitionId self_partition_id) override;
  //! Reports complete stage.
  void CompleteStage(StageId complete_stage_id, double cycles) override {
    ongoing_recipe_block_executor_->CompleteStage(complete_stage_id, cycles);
  }
  //! Gets the next ready stage.
  std::pair<StageId, StatementId> GetNextReadyStage() override;
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
  StageId get_earliest_unfinished_stage_id() override {
    return StageId::INVALID;
  }
  //! Gets the latest stage that is finished.
  StageId get_last_finished_stage_id() override {
    return StageId::INVALID;
  }
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
  std::unique_ptr<ApplicationRecipes> application_recipes_;
  // There is only one recipe block executor running at one time.
  std::unique_ptr<RecipeBlockExecutor> ongoing_recipe_block_executor_;
  VariableGroupId variable_group_id_;
  PartitionId partition_id_;
  std::shared_ptr<ControlFlowDecisionStorage> decision_storage_;
  std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage_;
};


}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_ENGINE_H_
