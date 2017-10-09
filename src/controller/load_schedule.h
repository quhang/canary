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
 * @file src/controller/load_schedule.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class LoadSchedule.
 */

#ifndef CANARY_SRC_CONTROLLER_LOAD_SCHEDULE_H_
#define CANARY_SRC_CONTROLLER_LOAD_SCHEDULE_H_

#include <map>
#include <set>
#include <string>

#include "shared/canary_internal.h"

#include "controller/scheduling_info.h"

namespace canary {

/**
 * A load schedule algorithm migrates partitions between workers for the best
 * performance or resource requirement.
 */
class LoadSchedule {
 protected:
  //! Constructor.
  explicit LoadSchedule(SchedulingInfo* scheduling_info)
      : scheduling_info_(scheduling_info) {}
  //! Destructor.
  virtual ~LoadSchedule() {}

 public:
  //! Factory method.
  static LoadSchedule* ConstructLoadSchedule(
      SchedulingInfo* scheduling_info, const std::string& name = "default");
  //! Invokes the load balancing algorithm.
  virtual void BalanceLoad() = 0;

 protected:
  SchedulingInfo* scheduling_info_ = nullptr;
};

/**
 * A test load balancing algorithm, which swaps two partitions.
 */
class TestLoadSchedule : public LoadSchedule {
 public:
  //! Constructor.
  explicit TestLoadSchedule(SchedulingInfo* scheduling_info)
      : LoadSchedule(scheduling_info) {}
  //! Destructor.
  virtual ~TestLoadSchedule() {}
  //! Invokes the load balancing algorithm.
  void BalanceLoad() override;
};

/**
 * A test load balancing algorithm, which moves one partition.
 */
class TestMoveLoadSchedule : public LoadSchedule {
 public:
  //! Constructor.
  explicit TestMoveLoadSchedule(SchedulingInfo* scheduling_info)
      : LoadSchedule(scheduling_info) {}
  //! Destructor.
  virtual ~TestMoveLoadSchedule() {}
  //! Invokes the load balancing algorithm.
  void BalanceLoad() override;
};

/**
 * A simple load balancing algorithm, which balances the number of partitions
 * across workers.
 */
class BalancedPartitionNumberLoadSchedule : public LoadSchedule {
 public:
  //! Constructor.
  explicit BalancedPartitionNumberLoadSchedule(SchedulingInfo* scheduling_info)
      : LoadSchedule(scheduling_info) {}
  //! Destructor.
  virtual ~BalancedPartitionNumberLoadSchedule() {}
  //! Invokes the load balancing algorithm.
  void BalanceLoad() override;

 protected:
  struct WorkerInfoRecord {
    WorkerId worker_id = WorkerId::INVALID;
    std::set<FullPartitionId> owned_partitions;
    int num_cores = 0;
    int expected_num_partitions = 0;
  };

  void GrabWorkerInfo();
  void CalculateExpectedNumPartitions();
  void IssuePartitionPlacementDecision();
  bool OffloadPartition(WorkerInfoRecord* worker_info_record);

  std::map<WorkerId, WorkerInfoRecord> worker_info_;
  int total_cores_ = 0;
  int total_partitions_ = 0;
};

/**
 * Mitigates straggler by killing the worker with the most stolen CPU.
 */
class StragglerMitigationLoadSchedule
    : public BalancedPartitionNumberLoadSchedule {
 public:
  //! Constructor.
  explicit StragglerMitigationLoadSchedule(SchedulingInfo* scheduling_info)
      : BalancedPartitionNumberLoadSchedule(scheduling_info) {}
  //! Destructor.
  virtual ~StragglerMitigationLoadSchedule() {}
  //! Invokes the load balancing algorithm.
  void BalanceLoad() override;

 private:
  void FigureOutStraggler();
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_LOAD_SCHEDULE_H_
