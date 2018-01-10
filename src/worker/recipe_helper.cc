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
 * @file src/worker/recipe_helper.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class RecipeHelper.
 */

#include "worker/recipe_helper.h"

namespace canary {
namespace recipe_helper {

namespace {

AccessRequirement AdjustDynamicAccessRequirement(
    const AccessRequirement& access_requirement, StageId begin_stage_id,
    const PartitionMetadataStorage& partition_metadata_storage_before_block) {
  if (!access_requirement.need_dynamic_adjustment) {
    return access_requirement;
  }
  auto result = access_requirement;
  const auto& access_metadata =
      partition_metadata_storage_before_block.GetPartitionAccessMetadata(
          access_requirement.variable_id);
  result.last_write_stage_id_offset =
      get_distance(begin_stage_id, access_metadata.last_write_stage_id);
  if (access_requirement.access_type == AccessRequirement::AccessType::WRITE) {
    result.num_read_stages += access_metadata.num_read_stages;
  }
  return result;
}

/*
 * Match the access requirement. Return whether there
 * is a match, and the stage id of the match.
 */
bool MatchAccessRequirement(
    const AccessRequirement& access_requirement, StageId begin_stage_id,
    StageId end_stage_id, int step_size,
    const PartitionMetadataStorage& partition_metadata_storage,
    const PartitionMetadataStorage& partition_metadata_storage_before_block,
    StageId* result_stage_id) {
  const auto access_requirement_adjusted =
      AdjustDynamicAccessRequirement(access_requirement, begin_stage_id,
                                     partition_metadata_storage_before_block);
  const auto& access_metadata =
      partition_metadata_storage.GetPartitionAccessMetadata(
          access_requirement.variable_id);
  // If the access type is WRITE, the number of reading stages needs to be
  // checked.
  if (access_requirement_adjusted.access_type ==
          AccessRequirement::AccessType::WRITE &&
      access_metadata.num_read_stages !=
          access_requirement_adjusted.num_read_stages) {
    return false;
  }
  int32_t difference = get_distance(
      get_next(begin_stage_id,
               access_requirement_adjusted.last_write_stage_id_offset),
      access_metadata.last_write_stage_id);
  if (difference % step_size != 0) {
    return false;
  }
  StageId candidate_stage_id = get_next(begin_stage_id, difference);
  if (candidate_stage_id < begin_stage_id ||
      candidate_stage_id >= end_stage_id) {
    return false;
  }
  *result_stage_id = candidate_stage_id;
  return true;
}

}  // namespace

bool MatchRecipe(
    const Recipe& recipe, StageId begin_stage_id, StageId end_stage_id,
    int step_size, const PartitionMetadataStorage& partition_metadata_storage,
    const PartitionMetadataStorage& partition_metadata_storage_before_block,
    StageId* result_stage_id) {
  StageId candidate_stage_id;
  bool first_entry = true;
  CHECK(!recipe.variable_id_to_access_requirement.empty());
  for (const auto& key_value : recipe.variable_id_to_access_requirement) {
    const auto& access_requirement = key_value.second;
    StageId temp_stage_id;
    bool matched = MatchAccessRequirement(
        access_requirement, begin_stage_id, end_stage_id, step_size,
        partition_metadata_storage, partition_metadata_storage_before_block,
        &temp_stage_id);
    if (!matched) {
      return false;
    }
    if (first_entry) {
      first_entry = false;
      candidate_stage_id = temp_stage_id;
    } else {
      if (candidate_stage_id != temp_stage_id) {
        return false;
      }
    }
  }
  *result_stage_id = candidate_stage_id;
  return true;
}

void ApplyRecipe(const Recipe& recipe, StageId stage_id,
                 PartitionMetadataStorage* partition_metadata_storage) {
  for (const auto& key_value : recipe.variable_id_to_access_requirement) {
    auto variable_id = key_value.first;
    const auto& access_requirement = key_value.second;
    if (access_requirement.access_type ==
        AccessRequirement::AccessType::WRITE) {
      partition_metadata_storage->WritePartition(variable_id, stage_id);
    } else {
      // access_requirement.access_type ==
      // AccessRequirement::AccessRequirement::READ;
      partition_metadata_storage->ReadPartition(variable_id);
    }
  }
}

bool IsDependentRecipes(const Recipe& first_recipe,
                        const Recipe& second_recipe) {
  for (const auto& key_value : first_recipe.variable_id_to_access_requirement) {
    if (second_recipe.variable_id_to_access_requirement.count(
            key_value.first) != 0 &&
        (second_recipe.variable_id_to_access_requirement.at(key_value.first)
                 .access_type == AccessRequirement::AccessType::WRITE ||
         key_value.second.access_type ==
             AccessRequirement::AccessType::WRITE)) {
      return true;
    }
  }
  return false;
}

}  // namespace recipe_helper
}  // namespace canary
