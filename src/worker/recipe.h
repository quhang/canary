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

#include "shared/canary_application.h"
#include "shared/canary_internal.h"

namespace canary {

struct AccessRequirement {
  VariableId variable_id;
  enum class AccessType { READ, WRITE } access_type;
  // The stage id is offset relative to the beginning of its recipe block.
  StageId last_write_stage_id_offset;
  int32_t num_read_stages;
  // If what are executed before the recipe are data-dependent, the access
  // record may need dynamic adjustment. This happens when the last write stage
  // is before the begining of the recipe block and unpredictable.
  bool need_dynamic_adjustment;
};

struct Recipe {
  RecipeId recipe_id;
  VariableGroupId variable_group_id;
  // The stage id is offset relative to the beginning of its recipe block.
  StageId current_stage_id_offset;
  std::map<VariableId, AccessRequirement> variable_id_to_access_requirement;
  std::set<RecipeId> recipe_ids_to_fire;
};

}  // namespace canary
#endif  // CANARY_SRC_WORKER_RECIPE_H_
