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
 * @file include/canary/canary_task_context.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryTaskContext.
 */

#ifndef CANARY_INCLUDE_CANARY_CANARY_TASK_CONTEXT_H_
#define CANARY_INCLUDE_CANARY_CANARY_TASK_CONTEXT_H_

#include <list>
#include <map>
#include <vector>

#include "canary/canary_internal.h"

#include "canary/canary_application.h"

namespace canary {

class WorkerSendDataInterface;
class PartitionData;
template <typename T>
class TypedPartitionData {};

/**
 * The context of a task.
 */
class CanaryTaskContext {
 public:
  CanaryTaskContext() {}
  virtual ~CanaryTaskContext() {}
  NON_COPYABLE_AND_NON_MOVABLE(CanaryTaskContext);

  //! Broadcasts a data.
  template <typename T>
  void Broadcast(const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(data);
    }
    BroadcastInternal(buffer);
  }

  //! Reduces data.
  template <typename T, typename Function>
  std::enable_if_t<
      std::is_same<std::decay_t<std::result_of_t<Function(T, T)>>, T>::value, T>
  Reduce(T initial, Function combiner) {
    std::vector<T> receive_data(Gather<T>());
    for (auto& element : receive_data) {
      initial = combiner(initial, element);
    }
    return initial;
  }

  //! Reduces data.
  template <typename T, typename Function>
  std::enable_if_t<std::is_void<std::result_of_t<Function(T, T*)>>::value, void>
  Reduce(T* initial, Function combiner) {
    CHECK_NOTNULL(initial);
    std::vector<T> receive_data(Gather<T>());
    for (auto& element : receive_data) {
      combiner(element, initial);
    }
  }

  //! Scatters a data.
  template <typename T>
  void Scatter(int partition_id, const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(data);
    }
    ScatterInternal(partition_id, buffer);
  }

  //! Gathers data.
  template <typename T>
  std::vector<T> Gather() {
    std::vector<T> receive_data(receive_buffer_.size());
    auto iter = receive_data.front();
    for (auto buffer : receive_buffer_) {
      {
        CanaryInputArchive archive(buffer);
        archive(*iter);
      }
      evbuffer_free(buffer);
      ++iter;
    }
    receive_buffer_.clear();
    return std::move(receive_data);
  }

  //! Ordered scatters a data.
  template <typename T>
  void OrderedScatter(int partition_id, const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(GetPartitionId());
      archive(data);
    }
    ScatterInternal(partition_id, buffer);
  }

  //! Ordered gathers data.
  template <typename T>
  std::map<int, T> OrderedGather() {
    std::map<int, T> receive_buffer;
    int src_partition_id;
    for (auto buffer : receive_buffer_) {
      CanaryInputArchive archive(buffer);
      archive(src_partition_id);
      archive(receive_buffer[src_partition_id]);
    }
    return std::move(receive_buffer);
  }

  //! Reads a variable.
  template <typename T>
  const T& ReadVariable(CanaryApplication::VariableHandle<T> handle) {
    auto pointer =
        dynamic_cast<TypedPartitionData<T>*>(ReadVariableInternal(handle));
    CHECK(pointer != nullptr) << "Invalid variable read.";
    return *pointer->get_data();
  }

  //! Writes a variable.
  template <typename T>
  T* WriteVariable(CanaryApplication::VariableHandle<T> handle) {
    auto pointer =
        dynamic_cast<TypedPartitionData<T>*>(WriteVariableInternal(handle));
    CHECK(pointer != nullptr) << "Invalid variable read.";
    return pointer->get_data();
  }

  int GetGatherSize() const { return static_cast<int>(receive_buffer_.size()); }
  int GetScatterParallelism() const { return scatter_partitioning_; }
  int GetGatherParallelism() const { return gather_partitioning_; }
  int GetPartitionId() const { return self_partition_id_; }

 private:
  void BroadcastInternal(struct evbuffer* buffer);
  void ScatterInternal(int partition_id, struct evbuffer* buffer);
  void* ReadVariableInternal(VariableId variable_id);
  void* WriteVariableInternal(VariableId variable_id);

  WorkerSendDataInterface* send_data_interface_;
  std::list<struct evbuffer*> receive_buffer_;
  std::map<VariableId, PartitionData*> read_partition_data_map_;
  std::map<VariableId, PartitionData*> write_partition_data_map_;
  int self_partition_id_;
  int scatter_partitioning_ = -1;
  int gather_partitioning_ = -1;
  ApplicationId application_id_;
  VariableGroupId gather_variable_group_id_;
  StageId gather_stage_id_;
};

}  // namespace canary
#endif  // CANARY_INCLUDE_CANARY_CANARY_TASK_CONTEXT_H_
