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
 * @file src/controller/controller_scheduler.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class ControllerScheduler.
 */

#ifndef CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
#define CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "shared/canary_internal.h"

#include "controller/controller_communication_interface.h"
#include "controller/launch_communication_interface.h"
#include "controller/scheduling_info.h"
#include "message/message_include.h"
#include "shared/canary_application.h"
#include "shared/network.h"
#include "shared/partition_map.h"

namespace canary {

/**
 * The base class of a controller scheduler, which hooks the scheduler with the
 * communication layer so that it can exchange messages with workers or
 * launchers.
 */
class ControllerSchedulerBase : public ControllerReceiveCommandInterface,
                                public LaunchReceiveCommandInterface {
 public:
  //! Constructor.
  ControllerSchedulerBase() {}
  //! Destructor.
  virtual ~ControllerSchedulerBase() {}
  //! Initialize.
  void Initialize(network::EventMainThread* event_main_thread,
                  ControllerSendCommandInterface* send_command_interface,
                  LaunchSendCommandInterface* launch_send_command_interface);
  /*
   * Receives commands from the communication layer.
   */
  //! Called when receiving a launching command. The message header is kept, and
  // the buffer ownership is transferred.
  void ReceiveLaunchCommand(LaunchCommandId launch_command_id,
                            struct evbuffer* buffer) override;
  //! Called when receiving a command. The message header is kept, and the
  // buffer ownership is transferred.
  void ReceiveCommand(struct evbuffer* buffer) override;
  //! Called when a worker is down, even if it is shut down by the controller.
  void NotifyWorkerIsDown(WorkerId worker_id) override;
  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void NotifyWorkerIsUp(WorkerId worker_id) override;

 protected:
  /*
   * Internal synchonous calls.
   */
  //! Called when receiving launching commands from a launcher.
  virtual void InternalReceiveLaunchCommand(LaunchCommandId launch_command_id,
                                            struct evbuffer* buffer) = 0;
  //! Called when receiving commands from a worker.
  virtual void InternalReceiveCommand(struct evbuffer* buffer) = 0;
  //! Called when a worker is down, even if it is shut down by the controller.
  virtual void InternalNotifyWorkerIsDown(WorkerId worker_id) = 0;
  //! Called when a worker is up. The up notification and down notification are
  // paired.
  virtual void InternalNotifyWorkerIsUp(WorkerId worker_id) = 0;

 protected:
  network::EventMainThread* event_main_thread_ = nullptr;
  ControllerSendCommandInterface* send_command_interface_ = nullptr;
  LaunchSendCommandInterface* launch_send_command_interface_ = nullptr;
};

class LoadSchedule;
class PlacementSchedule;

/**
 * The controller scheduler implements most execution management
 * functionalities. It processes commands from workers or launchers, and tracks
 * the running states of workers and applications. But it does not specify the
 * scheduling algorithmx. All method calls in this class are synchronous due to
 * the wrapper in ControllerSchedulerBase.
 */
class ControllerScheduler : public ControllerSchedulerBase,
                            public SchedulingInfo {
 public:
  ControllerScheduler() {}
  virtual ~ControllerScheduler() {}

 private:
  //! Called when receiving commands from a launcher.
  void InternalReceiveLaunchCommand(LaunchCommandId launch_command_id,
                                    struct evbuffer* buffer) override;
  //! Called when receiving commands from a worker.
  void InternalReceiveCommand(struct evbuffer* buffer) override;
  //! Called when a worker is down, even if it is shut down by the controller.
  void InternalNotifyWorkerIsDown(WorkerId worker_id) override;
  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void InternalNotifyWorkerIsUp(WorkerId worker_id) override;
  /*
   * Processes messages received from the launcher through the RPC interface.
   * All the commands are checked for correctness since user inputs are not
   * trusted.
   */
  bool CheckLaunchApplicationMessage(
      const message::LaunchApplication& launch_message,
      message::LaunchApplicationResponse* response);
  void ProcessLaunchApplication(
      LaunchCommandId launch_command_id,
      const message::LaunchApplication& launch_message);

  void ProcessPauseApplication(LaunchCommandId launch_command_id,
                               const message::PauseApplication& pause_message);

  bool CheckResumeApplicationMessage(
      const message::ResumeApplication& resume_message,
      message::ResumeApplicationResponse* response);
  void ProcessResumeApplication(
      LaunchCommandId launch_command_id,
      const message::ResumeApplication& resume_message);

  bool CheckControlApplicationPriorityMessage(
      const message::ControlApplicationPriority& control_message,
      message::ControlApplicationPriorityResponse* response);
  void ProcessControlApplicationPriority(
      LaunchCommandId launch_command_id,
      const message::ControlApplicationPriority& control_message);

  bool CheckRequestApplicationStatMessage(
      const message::RequestApplicationStat& request_message,
      message::RequestApplicationStatResponse* response);
  void ProcessRequestApplicationStat(
      LaunchCommandId launch_command_id,
      const message::RequestApplicationStat& request_message);

  void ProcessRequestShutdownWorker(
      LaunchCommandId launch_command_id,
      const message::RequestShutdownWorker& request_message);

  /*
   * Processes messages received from workers.
   */
  void ProcessMigrationInPrepared(
      const message::ControllerRespondMigrationInPrepared& respond_message);
  void ProcessMigrationInDone(
      const message::ControllerRespondMigrationInDone& respond_message);
  void ProcessPartitionDone(
      const message::ControllerRespondPartitionDone& respond_message);
  void ProcessStatusOfPartition(
      const message::ControllerRespondStatusOfPartition& respond_message);
  void ProcessStatusOfWorker(
      const message::ControllerRespondStatusOfWorker& respond_message);
  void ProcessReachBarrier(
      const message::ControllerRespondReachBarrier& respond_message);

  /*
   * Handling the state of an application.
   */
  //! Initializes an application record.
  bool InitializeApplicationRecord(
      ApplicationId application_id,
      const message::LaunchApplication& launch_message,
      message::LaunchApplicationResponse* response);
  //! Processes running stats.
  void UpdateRunningStats(WorkerId worker_id, ApplicationId application_id,
                          VariableGroupId variable_group_id,
                          PartitionId partition_id,
                          const message::RunningStats& running_stats);
  //! Cleans up an application after it is complete.
  void CleanUpApplication(ApplicationId application_id);

  /*
   * Application launching related.
   */
  //! Checks whether the partition map is filled in correctly by the scheduling
  // algorithm.
  bool CheckPartitionMapIntegrityAndMerge(ApplicationId application_id);
  //! Updates the partitions each workers own, and initializes the partition
  // record.
  void UpdateApplicationStateBasedOnPartitionMap(ApplicationId application_id);
  //! Loads partitions and applications.
  void InitializePartitionsOfApplicationOnWorkers(ApplicationId application_id);

  /*
   * Helper functions.
   */
  //! Checks if the application id is valid, and fills in the response.
  template <typename InputMessage, typename OutputMessage>
  bool CheckValidApplicationId(const InputMessage& input_message,
                               OutputMessage* output_message);
  //! Sends a command to every partition in the application.
  template <typename T>
  void SendCommandToPartitionInApplication(ApplicationId application_id,
                                           T* template_command);

  /*
   * Logging facility.
   */
  //! Initializes logging file if haven't.
  void InitializeLoggingFile();
  //! Flushes logging file.
  void FlushLoggingFile();
  //! Logs an application.
  void LogApplication(ApplicationId application_id,
                      const std::string& binary_location,
                      const std::string& application_parameter);
  //! Transforms the application parameter string to printable string.
  static std::string TransformString(const std::string& input);
  //! Logs a running stat report.
  void LogRunningStats(WorkerId worker_id, ApplicationId application_id,
                       VariableGroupId variable_group_id,
                       PartitionId partition_id,
                       const message::RunningStats& running_stats);

  /*
   * Implements the SchedulingInfo interface.
   */
  const std::map<WorkerId, WorkerRecord>& get_worker_map() const override {
    return worker_map_;
  }
  const std::map<ApplicationId, ApplicationRecord>& get_application_map()
      const override {
    return application_map_;
  }
  const std::map<FullPartitionId, PartitionRecord>& get_partition_record_map()
      const override {
    return partition_record_map_;
  }

  /*
   * Constructs scheduling algorithms.
   */
  PlacementSchedule* get_placement_schedule(const std::string& name);
  LoadSchedule* get_load_schedule(const std::string& name);

 private:
  //! Logging file handler.
  FILE* log_file_ = nullptr;
  //! The next application id to assign.
  ApplicationId next_application_id_ = ApplicationId::FIRST;
  //! Scheduling info.
  std::map<WorkerId, WorkerRecord> worker_map_;
  std::map<ApplicationId, ApplicationRecord> application_map_;
  std::map<FullPartitionId, PartitionRecord> partition_record_map_;
  //! Scheduling algorithms.
  std::map<std::string, PlacementSchedule*> placement_schedule_algorithms_;
  std::map<std::string, LoadSchedule*> load_schedule_algorithms_;

 protected:
  bool MigratePartition(FullPartitionId full_partition_id,
                        WorkerId to_worker_id) {
    // MigrateIn=>MigrateIn prepared=>MigrateOut=>MigrateIn done=>Change
    // partition map and everything.
    return true;
  }
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
