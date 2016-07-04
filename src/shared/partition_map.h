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
 * @file src/shared/partition_map.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class PartitionMap.
 */

#ifndef CANARY_SRC_SHARED_PARTITION_MAP_H_
#define CANARY_SRC_SHARED_PARTITION_MAP_H_

#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shared/canary_internal.h"

namespace canary {

//! A partition map update.
typedef std::list<std::pair<FullPartitionId, WorkerId>> PartitionMapUpdate;

/**
 * Per-application partition map.
 */
class PerApplicationPartitionMap {
 public:
  //! Queries the worker id of a partition. Returns INVALID if not available.
  WorkerId QueryWorkerId(VariableGroupId variable_group_id,
                         PartitionId partition_id) const;

  //! Queries the partitioning of a variable group. Returns -1 if not available.
  int QueryPartitioning(VariableGroupId variable_group_id) const;

  //! Sets the number of variable groups.
  void SetNumVariableGroup(int num_variable);

  //! Sets the partitioning of a variable.
  void SetPartitioning(VariableGroupId variable_group_id, int partitioning);

  //! Sets the worker id of a partition.
  void SetWorkerId(VariableGroupId variable_group_id, PartitionId partition_id,
                   WorkerId worker_id);

  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_map_);
  }

 private:
  std::vector<std::vector<WorkerId>> application_map_;
};

/**
 * The partition map of all applications.
 */
class PartitionMap {
 public:
  //! Gets a per-application partition map.
  PerApplicationPartitionMap* GetPerApplicationPartitionMap(
      ApplicationId application_id);

  //! Adds a per-application partition map.
  PerApplicationPartitionMap* AddPerApplicationPartitionMap(
      ApplicationId application_id);

  //! Deletes a per-application partition map. Returns whether the deletion
  // succeeds.
  bool DeletePerApplicationPartitionMap(ApplicationId application_id);

  //! Merges the updates.
  void MergeUpdate(const PartitionMapUpdate& update);

  //! Queries worker id.
  WorkerId QueryWorkerId(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id) const;

  //! Queries worker id.
  WorkerId QueryWorkerId(const FullPartitionId& full_partition_id) const;

  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(partition_map_);
  }

 private:
  std::unordered_map<ApplicationId, PerApplicationPartitionMap> partition_map_;
};

}  // namespace canary
#endif  // CANARY_SRC_SHARED_PARTITION_MAP_H_
