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
 * @file src/worker/recipe_constructor.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeConstructor.
 */

#include "worker/recipe_constructor.h"

namespace canary {

void RecipeConstructor::Construct() {
  result_ = std::make_unique<ApplicationRecipes>();

  recipe_block_id_counter_ = RecipeBlockId::FIRST;
  recipe_id_counter_ = RecipeId::FIRST;
  ComputeInnerLoopStatements();
  ComputeRecipeBlockId();

  std::list<StatementId> temp_statements;
  int num_iterations = 0;
  // Caution: nested fixed-iteration looping is not supported.
  for (const auto& key_value : *statement_info_map_) {
    StatementId statement_id = key_value.first;
    const auto& statement_info = key_value.second;
    switch (statement_info.statement_type) {
      case CanaryApplication::StatementType::LOOP:
        if (!temp_statements.empty()) {
          auto current_recipe_block_id = 
              statement_id_to_recipe_block_id_.at(temp_statements.back());
          ConstructRecipeBlockNoneDataDependent(
              temp_statements, current_recipe_block_id,
              get_next(current_recipe_block_id), result_.get());
          temp_statements.clear();
        }
        num_iterations = statement_info.num_loop;
        CHECK(inner_loop_statements_.find(statement_id) !=
              inner_loop_statements_.end())
            << "Outer looping with fixed iterations is not supported";
        break;
      case CanaryApplication::StatementType::END_LOOP:
        if (!temp_statements.empty()) {
          auto current_recipe_block_id = 
              statement_id_to_recipe_block_id_.at(temp_statements.back());
          ConstructRecipeBlockFixedItertations(
              temp_statements, num_iterations, current_recipe_block_id,
              get_next(current_recipe_block_id, 2), result_.get());
          temp_statements.clear();
        }
        break;
      case CanaryApplication::StatementType::WHILE:
        temp_statements.push_back(statement_id);
        ConstructRecipeBlockDataDependent(
            temp_statements, statement_id_to_recipe_block_id_.at(statement_id),
            get_next(statement_id_to_recipe_block_id_.at(statement_id)),
            statement_id_to_recipe_block_id_.at(
                statement_info.branch_statement),
            result_.get());
        temp_statements.clear();
        break;
      case CanaryApplication::StatementType::END_WHILE:
        {
          StatementId while_statement_id = statement_info.branch_statement;
          const auto& while_statement_info =
              statement_info_map_->at(while_statement_id);
          temp_statements.push_back(while_statement_id);
          if (inner_loop_statements_.find(statement_id) !=
              inner_loop_statements_.end()) {
            ConstructRecipeBlockDataDependentAndInnerIterative(
                temp_statements,
                statement_id_to_recipe_block_id_.at(statement_id),
                get_next(statement_id_to_recipe_block_id_.at(statement_id), 2),
                result_.get());
          } else {
            ConstructRecipeBlockDataDependent(
                temp_statements,
                statement_id_to_recipe_block_id_.at(statement_id),
                get_next(
                    statement_id_to_recipe_block_id_.at(while_statement_id)),
                statement_id_to_recipe_block_id_.at(
                    while_statement_info.branch_statement),
                result_.get());
          }
          temp_statements.clear();
        }
        break;
      default:
        temp_statements.push_back(statement_id);
    }
  }
}

/*
 * Compute all the statements that are in the inner loop.
 */
void RecipeConstructor::ComputeInnerLoopStatements() {
  inner_loop_statements_.clear();
  std::list<StatementId> temp_statements;
  bool is_inner_loop = false;
  for (const auto& key_value : *statement_info_map_) {
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
          inner_loop_statements_.insert(temp_statements.begin(),
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
void RecipeConstructor::ComputeRecipeBlockId() {
  result_->begin_recipe_block_id = recipe_block_id_counter_;
  for (const auto& key_value : *statement_info_map_) {
    StatementId statement_id = key_value.first;
    const auto& statement_info = key_value.second;
    switch (statement_info.statement_type) {
      case CanaryApplication::StatementType::LOOP:
        ++recipe_block_id_counter_;
        statement_id_to_recipe_block_id_[statement_id] =
            recipe_block_id_counter_;
        break;
      case CanaryApplication::StatementType::WHILE:
        statement_id_to_recipe_block_id_[statement_id] =
            recipe_block_id_counter_++;
        break;
      case CanaryApplication::StatementType::END_LOOP:
      case CanaryApplication::StatementType::END_WHILE:
        statement_id_to_recipe_block_id_[statement_id] =
            recipe_block_id_counter_;
        if (inner_loop_statements_.find(statement_id) !=
            inner_loop_statements_.end()) {
          // Every inner most loop corresponds to two blocks.
          recipe_block_id_counter_= get_next(recipe_block_id_counter_, 2);
        } else {
          ++recipe_block_id_counter_;
        }
        break;
      default:
        statement_id_to_recipe_block_id_[statement_id] =
            recipe_block_id_counter_;
    }
  }
  result_->end_recipe_block_id = recipe_block_id_counter_;
}

void RecipeConstructor::ConstructRecipeBlockNoneDataDependent(
    const std::list<StatementId>& statement_ids,
    RecipeBlockId recipe_block_id,
    RecipeBlockId next_recipe_block_id,
    ApplicationRecipes* result) {
  auto& recipe_block = result->recipe_block_map[recipe_block_id];
  recipe_block.recipe_block_id = recipe_block_id;
  recipe_block.recipe_block_type =
      RecipeBlock::RecipeBlockType::NONE_DATA_DEPENDENT;
  recipe_block.next_recipe_block_id = next_recipe_block_id;
  int stage_id_offset = 0;
  for (auto statement_id : statement_ids) {
    auto& recipe = result->recipe_map[recipe_id_counter_];
    recipe.recipe_id = recipe_id_counter_;
    result_->recipe_id_to_statement_id[recipe_id_counter_] = statement_id;
    recipe_block.recipe_ids.push_back(recipe_id_counter_);
    ++recipe_id_counter_;
    recipe.current_stage_id_offset = stage_id_offset++;
    const auto& statement_info = statement_info_map_->at(statement_id);
    recipe.variable_group_id = statement_info.variable_group_id;
    // TODO: Compute to_fire and access_requirements.
  }
}

}  // namespace canary
