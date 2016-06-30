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
 * @file src/worker/worker_light_thread_context.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class WorkerLightThreadContext.
 */

#include "worker/worker_light_thread_context.h"

namespace canary {

WorkerLightThreadContext::WorkerLightThreadContext() {
  PCHECK(pthread_mutex_init(&internal_lock_, nullptr) == 0);
}

WorkerLightThreadContext::~WorkerLightThreadContext() {
  pthread_mutex_destroy(&internal_lock_);
}

bool WorkerLightThreadContext::Enter() {
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  const bool success =
      (!running_) && (!ready_stages_.empty() || !command_list_.empty());
  running_ = true;
  pthread_mutex_unlock(&internal_lock_);
  return success;
}

bool WorkerLightThreadContext::Exit() {
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  CHECK(running_);
  const bool success = (ready_stages_.empty() && command_list_.empty());
  if (success) {
    running_ = false;
  }
  pthread_mutex_unlock(&internal_lock_);
  return success;
}

void WorkerLightThreadContext::ForceExit() {
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  const bool to_activate = (!ready_stages_.empty() || command_list_.empty());
  running_ = false;
  pthread_mutex_unlock(&internal_lock_);

  if (to_activate && activate_callback_) {
    activate_callback_();
  }
}

void WorkerLightThreadContext::DeliverMessage(StageId stage_id,
                                              struct evbuffer* buffer) {
  bool to_activate = false;
  if (stage_id >= StageId::FIRST) {
    // Normal data routed to a stage.
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    auto& stage_buffer = stage_buffer_map_[stage_id];
    stage_buffer.buffer_list.push_back(buffer);
    // If enough messages are received for the stage.
    if (stage_buffer.expected_buffer ==
        static_cast<int>(stage_buffer.buffer_list.size())) {
      ready_stages_.push_back(stage_id);
      if (!running_) {
        to_activate = true;
      }
    }
    pthread_mutex_unlock(&internal_lock_);
  } else {
    // A command requiring attention.
    PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
    command_list_.resize(command_list_.size() + 1);
    auto& command_buffer = command_list_.back();
    command_buffer.stage_id = stage_id;
    command_buffer.command = buffer;
    to_activate = !running_;
    pthread_mutex_unlock(&internal_lock_);
  }

  if (to_activate && activate_callback_) {
    activate_callback_();
  }
}

void WorkerLightThreadContext::RegisterReceivingData(StageId stage_id,
                                                     int num_message) {
  CHECK(stage_id >= StageId::FIRST);
  bool to_activate = false;

  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  auto& stage_buffer = stage_buffer_map_[stage_id];
  stage_buffer.expected_buffer = num_message;
  if (num_message == static_cast<int>(stage_buffer.buffer_list.size())) {
    ready_stages_.push_back(stage_id);
    if (!running_) {
      to_activate = true;
    }
  }
  pthread_mutex_unlock(&internal_lock_);

  if (to_activate && activate_callback_) {
    activate_callback_();
  }
}

bool WorkerLightThreadContext::RetrieveCommand(StageId* stage_id,
                                               struct evbuffer** command) {
  bool result = false;
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  if (!command_list_.empty()) {
    auto& command_buffer = command_list_.front();
    *stage_id = command_buffer.stage_id;
    if (command != nullptr) {
      *command = command_buffer.command;
    }
    command_list_.pop_front();
    result = true;
  }
  pthread_mutex_unlock(&internal_lock_);
  return result;
}

bool WorkerLightThreadContext::RetrieveData(
    StageId* stage_id, std::list<struct evbuffer*>* buffer_list) {
  bool result = false;
  PCHECK(pthread_mutex_lock(&internal_lock_) == 0);
  if (!ready_stages_.empty()) {
    const auto ready_stage = ready_stages_.front();
    ready_stages_.pop_front();
    auto iter = stage_buffer_map_.find(ready_stage);
    CHECK(iter != stage_buffer_map_.end());
    *stage_id = ready_stage;
    buffer_list->swap(iter->second.buffer_list);
    stage_buffer_map_.erase(iter);
    result = true;
  }
  pthread_mutex_unlock(&internal_lock_);
  return result;
}

}  // namespace canary
