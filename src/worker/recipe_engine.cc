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

namespace canary {

std::unique_ptr<RecipeBlockExecutor> RecipeBlockExecutor::Create(
    const ApplicationRecipes* application_recipes,
    RecipeBlockId recipe_block_id, VariableGroupId variable_group_id,
    std::shared_ptr<ControlFlowDecisionStorage> decision_storage,
    std::shared_ptr<PartitionMetadataStorage> partition_metadata_storage,
    StageId begin_stage_id) {
  std::unique_ptr<RecipeBlockExecutor> result;
  switch (application_recipes->recipe_block_map.at(recipe_block_id)
              .recipe_block_type) {
    case RecipeBlock::RecipeBlockType::NONE_DATA_DEPENDENT:
      result = std::make_unique<RecipeBlockExecutorNoneDataDependent>();
      break;
    // case RecipeBlock::RecipeBlockType::DATA_DEPENDENT_AND_NONE_ITERATIVE:
    //   break;
    // case RecipeBlock::RecipeBlockType::DATA_DEPENDENT_AND_ITERATIVE:
    //   break;
    // case RecipeBlock::RecipeBlockType::FIXED_ITERATIONS:
    //   break;
    default:
      LOG(FATAL) << "Internal error!";
  }
  result->Initialize(application_recipes, recipe_block_id, variable_group_id,
                     decision_storage, partition_metadata_storage,
                     begin_stage_id);
  return std::move(result);
}

void RecipeEngine::set_statement_info_map(
    const CanaryApplication::StatementInfoMap* statement_info_map) {
  statement_info_map_ = statement_info_map;
  application_recipes_ = ConstructRecipeFromStatementInfo(statement_info_map);
}

std::unique_ptr<ApplicationRecipes> RecipeEngine::ConstructRecipeFromStatementInfo(
    const CanaryApplication::StatementInfoMap* statement_info_map) {
  auto result = std::make_unique<ApplicationRecipes>();

  std::set<StatementId> inner_loop_statements;
  ComputeInnerLoopStatements(statement_info_map, &inner_loop_statements);
  std::map<StatementId, RecipeBlockId> statement_id_to_recipe_block_id;
  result->begin_recipe_block_id = RecipeBlockId::FIRST;
  result->end_recipe_block_id = ComputeRecipeBlockId(
      statement_info_map, inner_loop_statements, result->begin_recipe_block_id,
      &statement_id_to_recipe_block_id);

  std::list<StatementId> temp_statements;
  int num_iterations = 0;
  // Caution: nested fixed-iteration looping is not supported.
  for (const auto& key_value : *statement_info_map) {
    StatementId statement_id = key_value.first;
    const auto& statement_info = key_value.second;
    switch (statement_info.statement_type) {
      case CanaryApplication::StatementType::LOOP:
        if (!temp_statements.empty()) {
          auto current_recipe_block_id = 
              statement_id_to_recipe_block_id.at(temp_statements.back());
          ConstructRecipeBlockNoneDataDependent(
              temp_statements, current_recipe_block_id,
              get_next(current_recipe_block_id), result.get());
          temp_statements.clear();
        }
        num_iterations = statement_info.num_loop;
        CHECK(inner_loop_statements.find(statement_id) !=
              inner_loop_statements.end())
            << "Outer looping with fixed iterations is not supported";
        break;
      case CanaryApplication::StatementType::END_LOOP:
        if (!temp_statements.empty()) {
          auto current_recipe_block_id = 
              statement_id_to_recipe_block_id.at(temp_statements.back());
          ConstructRecipeBlockFixedItertations(
              temp_statements, num_iterations, current_recipe_block_id,
              get_next(current_recipe_block_id, 2), result.get());
          temp_statements.clear();
        }
        break;
      case CanaryApplication::StatementType::WHILE:
        temp_statements.push_back(statement_id);
        ConstructRecipeBlockDataDependent(
            temp_statements, statement_id_to_recipe_block_id.at(statement_id),
            get_next(statement_id_to_recipe_block_id.at(statement_id)),
            statement_id_to_recipe_block_id.at(statement_info.branch_statement),
            result.get());
        temp_statements.clear();
        break;
      case CanaryApplication::StatementType::END_WHILE:
        {
          StatementId while_statement_id = statement_info.branch_statement;
          const auto& while_statement_info =
              statement_info_map->at(while_statement_id);
          temp_statements.push_back(while_statement_id);
          if (inner_loop_statements.find(statement_id) !=
              inner_loop_statements.end()) {
            ConstructRecipeBlockDataDependentAndInnerIterative(
                temp_statements,
                statement_id_to_recipe_block_id.at(statement_id),
                get_next(statement_id_to_recipe_block_id.at(statement_id), 2),
                result.get());
          } else {
            ConstructRecipeBlockDataDependent(
                temp_statements,
                statement_id_to_recipe_block_id.at(statement_id),
                get_next(
                    statement_id_to_recipe_block_id.at(while_statement_id)),
                statement_id_to_recipe_block_id.at(
                    while_statement_info.branch_statement),
                result.get());
          }
          temp_statements.clear();
        }
        break;
      default:
        temp_statements.push_back(statement_id);
    }
  }
  return std::move(result);
}

/*
 * Compute all the statements that are in the inner loop.
 */
void RecipeEngine::ComputeInnerLoopStatements(
    const CanaryApplication::StatementInfoMap* statement_info_map,
    std::set<StatementId>* inner_loop_statements) {
  std::list<StatementId> temp_statements;
  bool is_inner_loop = false;
  for (const auto& key_value : *statement_info_map) {
    StatementId statement_id = key_value.first;
    const auto& statement_info = key_value.second;
    switch (statement_info.statement_type) {
      case CanaryApplication::StatementType::LOOP:
      case CanaryApplication::StatementType::WHILE:
        temp_statements.clear();
        is_inner_loop = true;
        temp_statements.push_back(statement_id);
        break;
      case CanaryApplication::StatementType::END_LOOP:
      case CanaryApplication::StatementType::END_WHILE:
        temp_statements.push_back(statement_id);
        if (is_inner_loop) {
          // Found an inner loop.
          inner_loop_statements->insert(temp_statements.begin(),
                                        temp_statements.end());
        }
        is_inner_loop = false;
        temp_statements.clear();
        break;
      default:
        temp_statements.push_back(statement_id);
    }
  }
}

/*
 * Compute the recipe block id for each statement. Note that a statement can be
 * in multiple recipe blocks, and only the first recipe block id is given..
 */
RecipeBlockId RecipeEngine::ComputeRecipeBlockId(
    const CanaryApplication::StatementInfoMap* statement_info_map,
    const std::set<StatementId>& inner_loop_statements,
    RecipeBlockId begin_recipe_block_id,
    std::map<StatementId, RecipeBlockId>* statement_id_to_recipe_block_id) {
  for (const auto& key_value : *statement_info_map) {
    StatementId statement_id = key_value.first;
    const auto& statement_info = key_value.second;
    switch (statement_info.statement_type) {
      case CanaryApplication::StatementType::LOOP:
        ++begin_recipe_block_id;
        (*statement_id_to_recipe_block_id)[statement_id] =
            begin_recipe_block_id;
        break;
      case CanaryApplication::StatementType::WHILE:
        (*statement_id_to_recipe_block_id)[statement_id] =
            begin_recipe_block_id++;
        break;
      case CanaryApplication::StatementType::END_LOOP:
      case CanaryApplication::StatementType::END_WHILE:
        (*statement_id_to_recipe_block_id)[statement_id] =
            begin_recipe_block_id;
        if (inner_loop_statements.find(statement_id) !=
            inner_loop_statements.end()) {
          // Every inner most loop corresponds to two blocks.
          begin_recipe_block_id = get_next(begin_recipe_block_id, 2);
        } else {
          ++begin_recipe_block_id;
        }
        break;
      default:
        (*statement_id_to_recipe_block_id)[statement_id] =
            begin_recipe_block_id;
    }
  }
  return begin_recipe_block_id;
}

void RecipeEngine::Initialize(VariableGroupId self_variable_group_id,
                              PartitionId self_partition_id) {
  variable_group_id_ = self_variable_group_id;
  partition_id_ = self_partition_id;
  decision_storage_ = std::make_shared<ControlFlowDecisionStorage>();
  partition_metadata_storage_ = std::make_shared<PartitionMetadataStorage>();
  // TODO: initialize the partitions.
  // partition_metadata_storage_before_block_->InitializePartition();
  ongoing_recipe_block_executor_ = RecipeBlockExecutor::Create(
      application_recipes_.get(), application_recipes_->begin_recipe_block_id,
      variable_group_id_, decision_storage_, partition_metadata_storage_,
      StageId::FIRST);
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
          decision_storage_, partition_metadata_storage_, next_stage_id);
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
