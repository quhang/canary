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
 * @file src/controller/controller_scheduler.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerScheduler.
 */

#include "controller/controller_scheduler.h"

namespace canary {

void ControllerSchedulerBase::Initialize(
    network::EventMainThread* event_main_thread,
    ControllerSendCommandInterface* send_command_interface) {
  event_main_thread_ = CHECK_NOTNULL(event_main_thread);
  send_command_interface_ = CHECK_NOTNULL(send_command_interface);
}

void ControllerSchedulerBase::ReceiveCommand(struct evbuffer* buffer) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalReceiveCommand, this, buffer));
}

void ControllerSchedulerBase::NotifyWorkerIsDown(WorkerId worker_id) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalNotifyWorkerIsDown, this, worker_id));
}

void ControllerSchedulerBase::NotifyWorkerIsUp(WorkerId worker_id) {
  event_main_thread_->AddInjectedEvent(std::bind(
      &ControllerSchedulerBase::InternalNotifyWorkerIsUp, this, worker_id));
}

#define PROCESS_MESSAGE(TYPE, METHOD)                                 \
  case MessageCategory::TYPE: {                                       \
    auto message =                                                    \
        new message::get_message_type<MessageCategory::TYPE>::Type(); \
    message::RemoveControlHeader(buffer);                             \
    message::DeserializeMessage(buffer, message);                     \
    METHOD(message);                                                  \
    break;                                                            \
  }
void ControllerScheduler::InternalReceiveCommand(struct evbuffer* buffer) {
  CHECK_NOTNULL(buffer);
  auto header = CHECK_NOTNULL(message::ExamineControlHeader(buffer));
  using message::MessageCategoryGroup;
  using message::MessageCategory;
  switch (header->category_group) {
    case MessageCategoryGroup::CONTROLLER_COMMAND:
      switch (header->category) {
        default:
          LOG(FATAL) << "Unexpected message type!";
      }  // switch category.
      break;
    case MessageCategoryGroup::LAUNCH_COMMAND:
      switch (header->category) {
        PROCESS_MESSAGE(LAUNCH_APPLICATION, ProcessLaunchApplication);
        default:
          LOG(FATAL) << "Unexpected message type!";
      }  // switch category.
      break;
    default:
      LOG(FATAL) << "Invalid message header!";
  }  // switch category group.
}
#undef PROCESS_MESSAGE

void ControllerScheduler::ProcessLaunchApplication(
    message::LaunchApplication* launch_message) {
  ApplicationId assigned_application_id = (next_application_id_++);
  auto& application_record = application_map_[assigned_application_id];
  application_record.binary_location = launch_message->binary_location;
  application_record.application_parameter =
      launch_message->application_parameter;
  application_record.loaded_application = CanaryApplication::LoadApplication(
      application_record.binary_location,
      application_record.application_parameter,
      &application_record.loading_handle);
  // TODO.
  delete launch_message;
}

void ControllerScheduler::InternalNotifyWorkerIsDown(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
}

void ControllerScheduler::InternalNotifyWorkerIsUp(WorkerId worker_id) {
  CHECK(worker_id != WorkerId::INVALID);
}

}  // namespace canary
