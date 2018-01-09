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
 * @file src/worker/recipe_block.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeBlock.
 */

#ifndef CANARY_SRC_WORKER_RECIPE_BLOCK_H_
#define CANARY_SRC_WORKER_RECIPE_BLOCK_H_

#include "worker/recipe.h"

namespace canary {

/*
 * A recipe block contains recipes to execute, while only the last recipe in
 * the block may return a boolean value that determines what recipe block to
 * run next.
 */
struct RecipeBlock {
  RecipeBlockId recipe_block_id;
  std::vector<RecipeId> recipe_ids;
  enum class RecipeBlockType {
    // The next block to run is deterministic.
    NONE_DATA_DEPENDENT,
    // The last recipe in the block determines what block to run next.
    DATA_DEPENDENT,
    // The last recipe in the block determines what block to run next, and the
    // block is iterative.
    DATA_DEPENDENT_AND_INNER_ITERATIVE,
    // The block runs a fixed number of iterations.
    FIXED_ITERATIONS
  } recipe_block_type;

  // Valid for NONE_DATA_DEPENDENT, FIXED_ITERATIONS and
  // DATA_DEPENDENT_AND_INNER_ITERATIVE.
  RecipeBlockId next_recipe_block_id;
  // Valid for DATA_DEPENDENT.
  RecipeBlockId next_recipe_block_id_if_true;
  RecipeBlockId next_recipe_block_id_if_false;
  // Valid for FIXED_ITERATIONS.
  int32_t num_iterations;
  std::string Print() const {
    std::stringstream ss;
    ss << "recipe block #" << get_value(recipe_block_id) << ", ";
    switch (recipe_block_type) {
      case RecipeBlockType::NONE_DATA_DEPENDENT:
        ss << " next block = " << get_value(next_recipe_block_id) << ", ";
        break;
      case RecipeBlockType::DATA_DEPENDENT:
        ss << " next block = " << get_value(next_recipe_block_id_if_true) << "/"
           << get_value(next_recipe_block_id_if_false) << ", ";
        break;
      case RecipeBlockType::DATA_DEPENDENT_AND_INNER_ITERATIVE:
        ss << " iterative, next block = " << get_value(next_recipe_block_id)
           << ", ";
        break;
      case RecipeBlockType::FIXED_ITERATIONS:
        ss << num_iterations
           << " loops, next block = " << get_value(next_recipe_block_id)
           << ", ";
        break;
    }

    return ss.str();
  }
};

/*
 * Describe all the recipes in an application.
 */
struct ApplicationRecipes {
  std::map<RecipeId, Recipe> recipe_map;
  std::map<RecipeBlockId, RecipeBlock> recipe_block_map;
  RecipeBlockId begin_recipe_block_id, end_recipe_block_id;
  std::map<RecipeId, StatementId> recipe_id_to_statement_id;
  std::string Print() const {
    std::stringstream ss;
    for (const auto& block_key_value : recipe_block_map) {
      ss << block_key_value.second.Print() << std::endl;
      for (auto recipe_id : block_key_value.second.recipe_ids) {
        ss << "  statement #"
           << get_value(recipe_id_to_statement_id.at(recipe_id)) << ", "
           << recipe_map.at(recipe_id).Print() << std::endl;
      }
    }
    ss << "start block #" << get_value(begin_recipe_block_id) << std::endl;
    ss << "end block #" << get_value(end_recipe_block_id) << std::endl;
    return ss.str();
  }
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_BLOCK_H_
