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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "shared/canary_internal.h"

#include "controller/controller_communication_interface.h"
#include "controller/launch_communication_interface.h"
#include "message/message_include.h"
#include "shared/canary_application.h"
#include "shared/network.h"
#include "shared/partition_map.h"

namespace canary {

/*
 * The skeleton of the controller scheduler implementation, which wraps the
 * asynchronous call into synchounous call into subclass.
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
   * Callbacks exposed to the synchronous context.
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

class ControllerScheduler : public ControllerSchedulerBase {
 protected:
  //! Represents an active worker.
  struct WorkerRecord {
    int num_cores = -1;
    std::map<ApplicationId, std::set<FullPartitionId>> owned_partitions;
    std::set<ApplicationId> loaded_applications;
  };
  //! Represents the execution state of an application.
  enum class ApplicationState {
    INVALID = -1,
    RUNNING = 0,
    AT_BARRIER,
    COMPLETE
  };
  //! Represents an active application.
  struct ApplicationRecord {
    //! Binary location of the application.
    std::string binary_location;
    //! The application parameter.
    std::string application_parameter;
    //! The first barrier stage, at which all partitions should pause and wait
    // for resuming.
    StageId first_barrier_stage = StageId::INVALID;
    //! Priority level of the application, lower means higher priority.
    PriorityLevel priority_level;
    //! Internal usage, the dynamic loading handle of the application.
    void* loading_handle = nullptr;
    //! The loaded application.
    CanaryApplication* loaded_application = nullptr;
    //! Describing the variables in the application.
    const CanaryApplication::VariableGroupInfoMap* variable_group_info_map =
        nullptr;
    //! The application's partition map.
    PerApplicationPartitionMap per_app_partition_map;
    //! The total number of partitions.
    int total_partition = 0;
    //! The total number of complete partitions.
    int complete_partition = 0;
    //! The total number of partitions blocked at a barrier.
    int blocked_partition = 0;
    //! The execution state of an application.
    ApplicationState application_state = ApplicationState::INVALID;
    //! The total of cycles spent for the application.
    double total_used_cycles = 0;
    //! The identifier of a triggered report.
    int report_id = -1;
    //! The number of partitions that have reported since the last time the
    // report id was changed.
    std::set<FullPartitionId> report_partition_set;
    //! Lauchers that wait for reporting the running stats.
    std::vector<LaunchCommandId> report_command_list;
  };

 public:
  ControllerScheduler() {}
  virtual ~ControllerScheduler() {}

 protected:
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

 private:
  /*
   * Processes messages received from the launcher through the RPC interface.
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
   * Application launching related.
   */
  //! Fills in initial info in the application record.
  bool InitializeApplicationRecord(
      const message::LaunchApplication& launch_message,
      ApplicationRecord* application_record);
  //! Assigns partitions to workers.
  void DecidePartitionMap(ApplicationRecord* application_record);
  //! Returns NUM_SLOT worker id, by assigning load to workers in a round-robin
  // manner using the number of cores as a weight.
  void GetWorkerAssignment(int num_slot, std::vector<WorkerId>* assignment);
  //! Gets the next assigned worker id.
  WorkerId NextAssignWorkerId();
  //! Loads an application on all workers.
  void RequestLoadApplicationOnAllWorkers(
      ApplicationId application_id,
      const ApplicationRecord& application_record);
  //! Updates the partitions each workers own.
  void UpdateWorkerOwnedPartitions(
      ApplicationId application_id,
      const PerApplicationPartitionMap& per_app_partition_map);
  //! Requests workers to load partitions.
  void RequestLoadPartitions(ApplicationId application_id);

  /*
   * Misc functions.
   */
  //! Checks if the application id is valid, and fills in the response.
  template <typename InputMessage, typename OutputMessage>
  bool CheckValidApplicationId(const InputMessage& input_message,
                               OutputMessage* output_message);
  //! Sends a command to every partition in the application.
  template <typename T>
  void SendCommandToPartitionInApplication(ApplicationId application_id,
                                           T* template_command);
  //! Cleans up an application after it is complete.
  void CleanUpApplication(ApplicationId application_id,
                          ApplicationRecord* application_record);
  //! Processes running stats.
  void UpdateRunningStats(WorkerId worker_id, ApplicationId application_id,
                          VariableGroupId variable_group_id,
                          PartitionId partition_id,
                          const message::RunningStats& running_stats);

 protected:
  /*
   * Logging related.
   */
  //! Initializes logging file.
  void InitializeLoggingFile();
  //! Flushes logging file.
  void FlushLoggingFile();
  //! Transforms the application parameter string to printable string.
  static std::string TransformString(const std::string& input);

 private:
  WorkerId last_assigned_worker_id_ = WorkerId::INVALID;
  int last_assigned_partitions_ = 0;
  //! The next application id to assign.
  ApplicationId next_application_id_ = ApplicationId::FIRST;

 protected:
  FILE* log_file_ = nullptr;
  std::map<WorkerId, WorkerRecord> worker_map_;
  std::map<ApplicationId, ApplicationRecord> application_map_;
};

}  // namespace canary
#endif  // CANARY_SRC_CONTROLLER_CONTROLLER_SCHEDULER_H_
