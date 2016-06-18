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
 * @file src/shared/partition_map.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class PartitionMap.
 */

#include "shared/partition_map.h"

namespace canary {

WorkerId PerApplicationPartitionMap::QueryWorkerId(
    VariableId variable_id, PartitionId partition_id) const {
  const auto variable_id_value = get_value(variable_id);
  CHECK_GE(variable_id_value, 0);
  const auto partition_id_value = get_value(partition_id);
  CHECK_GE(partition_id_value, 0);
  if (variable_id_value >= static_cast<int>(application_map_.size())) {
    return WorkerId::INVALID;
  }
  const auto& per_variable_map = application_map_[variable_id_value];
  if (partition_id_value >= static_cast<int>(per_variable_map.size())) {
    return WorkerId::INVALID;
  }
  return per_variable_map[partition_id_value];
}

int PerApplicationPartitionMap::QueryVariablePartitioning(
    VariableId variable_id) const {
  const auto variable_id_value = get_value(variable_id);
  CHECK_GE(variable_id_value, 0);
  if (variable_id_value >= static_cast<int>(application_map_.size())) {
    return -1;
  }
  return static_cast<int>(application_map_.at(variable_id_value).size());
}

void PerApplicationPartitionMap::SetNumVariable(int num_variable) {
  application_map_.clear();
  application_map_.resize(num_variable);
}

void PerApplicationPartitionMap::SetVariablePartitioning(VariableId variable_id,
                                                         int partitioning) {
  application_map_.at(get_value(variable_id)).clear();
  application_map_.at(get_value(variable_id))
      .resize(partitioning, WorkerId::INVALID);
}

void PerApplicationPartitionMap::SetWorkerId(VariableId variable_id,
                                             PartitionId partition_id,
                                             WorkerId worker_id) {
  const auto variable_id_value = get_value(variable_id);
  const auto partition_id_value = get_value(partition_id);
  application_map_.at(variable_id_value).at(partition_id_value) = worker_id;
}

PerApplicationPartitionMap* PartitionMap::GetPerApplicationPartitionMap(
    ApplicationId application_id) {
  auto iter = partition_map_.find(application_id);
  if (iter == partition_map_.end()) {
    return nullptr;
  } else {
    return &iter->second;
  }
}

PerApplicationPartitionMap* PartitionMap::AddPerApplicationPartitionMap(
    ApplicationId application_id) {
  return &partition_map_[application_id];
}

bool PartitionMap::DeletePerApplicationPartitionMap(
    ApplicationId application_id) {
  auto iter = partition_map_.find(application_id);
  if (iter == partition_map_.end()) {
    return false;
  } else {
    partition_map_.erase(iter);
    return true;
  }
}

void PartitionMap::MergeUpdate(const PartitionMapUpdate& update) {
  for (auto& pair : update) {
    auto& full_partition_id = pair.first;
    auto& per_application_partition_map =
        partition_map_.at(full_partition_id.application_id);
    per_application_partition_map.SetWorkerId(full_partition_id.variable_id,
                                              full_partition_id.partition_id,
                                              pair.second);
  }
}

WorkerId PartitionMap::QueryWorkerId(ApplicationId application_id,
                                     VariableId variable_id,
                                     PartitionId partition_id) const {
  auto iter = partition_map_.find(application_id);
  if (iter == partition_map_.end()) {
    return WorkerId::INVALID;
  } else {
    return iter->second.QueryWorkerId(variable_id, partition_id);
  }
}

}  // namespace canary
