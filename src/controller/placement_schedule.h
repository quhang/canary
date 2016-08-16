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
 * @file src/controller/placement_schedule.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class PlacementSchedule.
 */

#ifndef CANARY_SRC_CONTROLLER_PLACEMENT_SCHEDULE_H_
#define CANARY_SRC_CONTROLLER_PLACEMENT_SCHEDULE_H_

#include <string>

#include "shared/canary_internal.h"

#include "controller/scheduling_info.h"

namespace canary {

/**
 * The base class of a placement scheduling algorithm.
 */
class PlacementSchedule {
 protected:
  //! Constructor.
  explicit PlacementSchedule(SchedulingInfo* scheduling_info)
      : scheduling_info_(CHECK_NOTNULL(scheduling_info)) {}
  //! Destructor.
  virtual ~PlacementSchedule() {}

 public:
  //! Factory class.
  static PlacementSchedule* ConstructPlacementSchedule(
      SchedulingInfo* scheduling_info, const std::string& name = "default");
  //! Interface for invoking the placement algorithm.
  virtual void PlaceApplication(ApplicationId application_id) = 0;

 protected:
  SchedulingInfo* scheduling_info_ = nullptr;
};

/**
 * The default placement algorithm that places partitions evenly on workers.
 */
class EvenlyPlacementSchedule : public PlacementSchedule {
 public:
  //! Constructor.
  explicit EvenlyPlacementSchedule(SchedulingInfo* scheduling_info)
      : PlacementSchedule(scheduling_info) {}
  //! Destructor.
  virtual ~EvenlyPlacementSchedule() {}

 public:
  //! Invoking the placement algorithm.
  void PlaceApplication(ApplicationId application_id) override;

 private:
  //! Returns NUM_SLOT worker id, by assigning load to workers in a round-robin
  // manner using the number of cores as a weight.
  void GetWorkerAssignment(int num_slot, std::vector<WorkerId>* assignment);
  //! Gets the next assigned worker id.
  WorkerId NextAssignWorkerId();

 private:
  //! The last worker that was assigned a partition.
  WorkerId last_assigned_worker_id_ = WorkerId::INVALID;
  //! How many partitions were assigned to the last worker.
  int last_assigned_partitions_ = 0;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_PLACEMENT_SCHEDULE_H_
