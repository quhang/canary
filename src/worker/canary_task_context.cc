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
 * @file src/worker/canary_task_context.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryTaskContext.
 */

#include "worker/canary_task_context.h"

#include "message/message_include.h"
#include "worker/stage_graph.h"
#include "worker/worker_communication_interface.h"

namespace canary {

void CanaryTaskContext::BroadcastInternal(struct evbuffer* buffer) {
  CHECK_NOTNULL(send_data_interface_)
      ->BroadcastDataToPartition(application_id_, gather_variable_group_id_,
                                 gather_stage_id_, buffer);
}

void CanaryTaskContext::ScatterInternal(int partition_id,
                                        struct evbuffer* buffer) {
  CHECK_NOTNULL(send_data_interface_)
      ->SendDataToPartition(application_id_, gather_variable_group_id_,
                            static_cast<PartitionId>(partition_id),
                            gather_stage_id_, buffer);
}

// NOTE: By chinmayee
double CanaryTaskContext::GetComputeTime() const {
  return stage_graph_->compute_time();
}

// NOTE: By chinmayee
void CanaryTaskContext::SetComputeTime(double ct) {
  stage_graph_->set_compute_time(ct);
}

// NOTE: By chinmayee
double CanaryTaskContext::GetScatterGatherTime() const {
  return stage_graph_->scatter_gather_time();
}

// NOTE: By chinmayee
void CanaryTaskContext::SetScatterGatherTime(double sgt) {
  stage_graph_->set_scatter_gather_time(sgt);
}

// NOTE: By chinmayee
void CanaryTaskContext::SendComputedPartitionHistory(
  const PartitionHistory& history) const {
  int num_partitions = history.GetNumPartitions();
  int history_len = history.GetHistoryLen();
  CHECK(worker_id_ != WorkerId::INVALID);
  message::ControllerSendPartitionHistory history_message;
  history_message.from_worker_id = worker_id_;
  history_message.application_id = application_id_;
  history_message.num_partitions = num_partitions;
  history_message.history_len = history_len;
  history_message.history.clear();
  history_message.times.clear();
  history_message.last_time = history.GetLastTime();
  if (history_len > 0) {
    history_message.history.resize(history_len);
    history_message.times.resize(history_len);
    for (int i = 0; i < history_len; ++i) {
      const PartitionPlacement& placement = history.GetPlacement(i);
      CHECK_EQ(placement.GetNumPartitions(), num_partitions) <<
        "Placement size != Expected number of partitions" << std::endl;
      history_message.history[i] = placement.GetPlacementData();
      history_message.times[i] = history.GetTime(i);
      VLOG(1) << "Adding partition for time " << history_message.times[i];
    }  // for i
  }  // if history_len > 0
  LOG(INFO) << "Sending partition history of length " <<  history_len <<
             " to controller ...";
  CHECK_NOTNULL(send_command_interface_)->SendCommandToController(
    message::SerializeMessageWithControlHeader(history_message));
  VLOG(1) << "Done sending partition history to controller";
}  // SendComputedPartitionHistory

// NOTE: By chinmayee
void CanaryTaskContext::UpdatePlacement(float t) {
  VLOG(1) << "Sending update placement for " << t <<
             " to controller from partition " << self_partition_id_ <<
             " variable " << get_value(variable_group_id_) << " ...";
  CHECK(worker_id_ != WorkerId::INVALID);
  message::ControllerUpdatePlacementForTime update;
  update.from_worker_id = worker_id_;
  update.num_partitions = parallelism_;
  update.application_id = application_id_;
  update.variable_group_id = variable_group_id_;
  update.partition_id = self_partition_id_;
  update.time = t;
  CHECK_NOTNULL(send_command_interface_)->SendCommandToController(
    message::SerializeMessageWithControlHeader(update));
  VLOG(1) << "Sent update placement for " << t <<
             " to controller from partition " << self_partition_id_ <<
             " variable " << get_value(variable_group_id_);
}  // UpdatePlacement

}  // namespace canary
