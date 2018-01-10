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
 * @file src/worker/recipe_engine.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeEngine.
 */

#include "worker/recipe_engine.h"

#include "worker/recipe_constructor.h"

namespace canary {

std::unique_ptr<RecipeBlockExecutor> RecipeBlockExecutor::Create(
    const ApplicationRecipes* application_recipes,
    RecipeBlockId recipe_block_id, VariableGroupId variable_group_id,
    PartitionId partition_id,
    std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
    std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
    StageId begin_stage_id) {
  std::unique_ptr<RecipeBlockExecutor> result;
  switch (application_recipes->recipe_block_map.at(recipe_block_id)
              .recipe_block_type) {
    case RecipeBlock::RecipeBlockType::NONE_DATA_DEPENDENT:
      result = std::make_unique<RecipeBlockExecutorNoneDataDependent>();
      break;
    case RecipeBlock::RecipeBlockType::FIXED_ITERATIONS:
      result = std::make_unique<RecipeBlockExecutorFixedIterations>();
      break;
    case RecipeBlock::RecipeBlockType::DATA_DEPENDENT:
      result = std::make_unique<RecipeBlockExecutorDataDependent>();
      break;
    case RecipeBlock::RecipeBlockType::DATA_DEPENDENT_AND_INNER_ITERATIVE:
      result = std::make_unique<RecipeBlockExecutorDataDependentAndIterative>();
      break;
    default:
      LOG(FATAL) << "Internal error!";
  }
  result->Initialize(application_recipes, recipe_block_id, variable_group_id,
                     partition_id, decision_storage, partition_metadata_storage,
                     begin_stage_id);
  VLOG(1) << "Created: " << result->Print();
  return std::move(result);
}

void RecipeBlockExecutor::InitializeInternal(
      const ApplicationRecipes* application_recipes,
      RecipeBlockId recipe_block_id,
      VariableGroupId variable_group_id,
      PartitionId partition_id,
      std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
      std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage) {
    application_recipes_ = application_recipes;
    current_recipe_block_id_ = recipe_block_id;
    current_recipe_block_ =
        &application_recipes->recipe_block_map.at(recipe_block_id);
    variable_group_id_ = variable_group_id;
    partition_id_ = partition_id;
    decision_storage_ = decision_storage;
    partition_metadata_storage_ = partition_metadata_storage;
    partition_metadata_storage_before_block_ =
        partition_metadata_storage->Clone();
    // Count the number of recipes in one block.
    {
    num_recipes_in_one_block_ = 0;
    for (auto recipe_id : current_recipe_block_->recipe_ids) {
      if (application_recipes_->recipe_map.at(recipe_id).variable_group_id ==
          variable_group_id_) {
        ++num_recipes_in_one_block_;
      }
    }
    FireAllRecipes();
  }
}

bool RecipeBlockExecutor::CheckRecipe(RecipeId recipe_id, StageId* stage_id) {
  const auto& recipe = application_recipes_->recipe_map.at(recipe_id);
  int recipe_block_size = current_recipe_block_->recipe_ids.size();
  return recipe_helper::MatchRecipe(
      recipe, begin_stage_id_, end_stage_id_, recipe_block_size,
      *partition_metadata_storage_, *partition_metadata_storage_before_block_,
      stage_id);
}

bool RecipeBlockExecutor::RetrieveReadyStageIfPossible(StageId* stage_id,
                                                       RecipeId* recipe_id) {
  StageId result_stage_id;
  auto iter = fired_recipe_ids_.begin();
  while (iter != fired_recipe_ids_.end()) {
    if (ready_recipe_ids_.find(*iter) == ready_recipe_ids_.end() &&
        CheckRecipe(*iter, &result_stage_id)) {
      *recipe_id = *iter;
      *stage_id = result_stage_id;
      ready_recipe_ids_.insert(*iter);
      fired_recipe_ids_.erase(iter);
      return true;
    }
    iter = fired_recipe_ids_.erase(iter);
  }
  return false;
}

void RecipeBlockExecutor::CompleteRecipe(RecipeId recipe_id,
                                         StageId complete_stage_id) {
  --num_recipes_to_run_;
  const auto& recipe = application_recipes_->recipe_map.at(recipe_id);
  recipe_helper::ApplyRecipe(recipe, complete_stage_id,
                             partition_metadata_storage_.get());
  ready_recipe_ids_.erase(recipe_id);
  for (auto recipe_id_to_fire : recipe.recipe_ids_to_fire) {
    fired_recipe_ids_.insert(recipe_id_to_fire);
  }
}


void RecipeEngine::set_statement_info_map(
    const CanaryApplication::StatementInfoMap* statement_info_map) {
  statement_info_map_ = statement_info_map;
  application_recipes_ = RecipeConstructor(statement_info_map).RetrieveResult();
}

void RecipeEngine::Initialize(VariableGroupId self_variable_group_id,
                              PartitionId self_partition_id) {
  variable_group_id_ = self_variable_group_id;
  partition_id_ = self_partition_id;
  decision_storage_ = std::make_shared<ControlFlowDecisionStorage>();
  partition_metadata_storage_ = std::make_shared<PartitionMetadataStorage>();
  ongoing_recipe_block_executor_ = RecipeBlockExecutor::Create(
      application_recipes_.get(), application_recipes_->begin_recipe_block_id,
      variable_group_id_, partition_id_, decision_storage_,
      partition_metadata_storage_, StageId::FIRST);
}

std::pair<StageId, StatementId> RecipeEngine::GetNextReadyStage() {
  StageId stage_id;
  RecipeId recipe_id;
  auto retrieve_result =
      ongoing_recipe_block_executor_->GetNextReadyStage(&stage_id, &recipe_id);
  while (retrieve_result == RecipeBlockExecutor::RetrieveResult::COMPLETE) {
    RecipeBlockId next_recipe_block_id;
    StageId next_stage_id;
    ongoing_recipe_block_executor_->WrapUp(&next_recipe_block_id,
                                           &next_stage_id);
    if (next_recipe_block_id == application_recipes_->end_recipe_block_id) {
      ongoing_recipe_block_executor_.reset();
      return std::make_pair(StageId::COMPLETE, StatementId::INVALID);
    } else {
      ongoing_recipe_block_executor_ = RecipeBlockExecutor::Create(
          application_recipes_.get(), next_recipe_block_id, variable_group_id_,
          partition_id_, decision_storage_, partition_metadata_storage_,
          next_stage_id);
    }
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
    default:
      LOG(FATAL) << "Internal error!";
  }
}

}  // namespace canary
