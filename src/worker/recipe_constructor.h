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
 * @file src/worker/recipe_constructor.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeConstructor.
 */

#ifndef CANARY_SRC_WORKER_RECIPE_CONSTRUCTOR_H_
#define CANARY_SRC_WORKER_RECIPE_CONSTRUCTOR_H_

#include "worker/recipe.h"
#include "worker/recipe_block.h"
#include "worker/recipe_helper.h"

namespace canary {

/*
 * A helper class that constructs recipes for a job.
 */
class RecipeConstructor {
 public:
  explicit RecipeConstructor(
      const CanaryApplication::StatementInfoMap* statement_info_map)
      : statement_info_map_(statement_info_map) {}
  virtual ~RecipeConstructor() {}

  // Return the constructed recipes.
  std::unique_ptr<ApplicationRecipes> RetrieveResult() {
    if (!result_) {
      Construct();
    }
    return std::move(result_);
  }

 private:
  // Start constructing the recipes.
  void Construct();

  // Compute which statement is in the inner most loop.
  void ComputeInnerLoopStatements();
  // COmpute the recipe block id fo each recipe.
  void ComputeRecipeBlockId();

  void ConstructRecipeBlockNoneDataDependent(
      const std::list<StatementId>& statement_ids,
      RecipeBlockId recipe_block_id, RecipeBlockId next_recipe_block_id);
  void ConstructRecipeBlockDataDependent(
      const std::list<StatementId>& statement_ids,
      RecipeBlockId recipe_block_id, RecipeBlockId true_next_recipe_block_id,
      RecipeBlockId false_next_recipe_block_id);
  void ConstructRecipeBlockDataDependentAndInnerIterative(
      const std::list<StatementId>& statement_ids,
      RecipeBlockId recipe_block_id, RecipeBlockId next_recipe_block_id) {}
  void ConstructRecipeBlockFixedItertations(
      const std::list<StatementId>& statement_ids, int num_iterations,
      RecipeBlockId recipe_block_id, RecipeBlockId next_recipe_block_id) {}

  void ComputeRecipesInBlock(
    const std::list<StatementId>& statement_ids,
    RecipeBlock* recipe_block,
    PartitionMetadataStorage* partition_metadata);
  void FillInFireRelation(RecipeBlock* recipe_block);

  const CanaryApplication::StatementInfoMap* statement_info_map_;
  std::unique_ptr<ApplicationRecipes> result_;
  std::set<StatementId> inner_loop_statements_;
  std::map<StatementId, RecipeBlockId> statement_id_to_recipe_block_id_;
  RecipeBlockId recipe_block_id_counter_;
  RecipeId recipe_id_counter_;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_CONSTRUCTOR_H_
