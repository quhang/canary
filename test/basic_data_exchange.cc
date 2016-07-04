/*
 * Tests when the worker keeps reducing a value.
 */
#include <cstdlib>
#include <list>
#include <thread>

#include "shared/canary_internal.h"

#include "shared/initialize.h"
#include "controller/controller_communication_manager.h"
#include "controller/controller_scheduler.h"
#include "worker/worker_communication_manager.h"
#include "worker/worker_light_thread_context.h"
#include "worker/worker_scheduler.h"
#include "message/message_include.h"

DEFINE_int32(num_worker, 2, "Number of workers.");
DEFINE_int32(num_partition_per_worker, 2, "Number of partitions per worker.");
DEFINE_bool(controller_only, false, "Controller only.");
DEFINE_bool(worker_only, false, "Worker only.");
DEFINE_bool(trigger_swap, false, "Triggers swapping.");

using namespace canary;  // NOLINT

const VariableGroupId reduce_variable = VariableGroupId::FIRST;
const VariableGroupId distribute_variable = get_next(VariableGroupId::FIRST);

class TestControllerScheduler : public ControllerSchedulerBase {
 public:
  TestControllerScheduler() {}
  virtual ~TestControllerScheduler() {}

 protected:
  //! Called when receiving commands from a worker.
  void InternalReceiveCommand(struct evbuffer* buffer) override {
    CHECK_NOTNULL(buffer);
    LOG(FATAL) << "Not implemented.";
  }

  //! Called when a worker is down, even if it is shut down by the controller.
  void InternalNotifyWorkerIsDown(WorkerId worker_id) override {
    CHECK(worker_id != WorkerId::INVALID);
    LOG(FATAL) << "Worker is down (" << get_value(worker_id) << ").";
  }

  //! Called when a worker is up. The up notification and down notification are
  // paired.
  void InternalNotifyWorkerIsUp(WorkerId worker_id) override {
    CHECK(worker_id != WorkerId::INVALID);
    if (++num_workers_ == FLAGS_num_worker) {
      const auto total_partitions = FLAGS_num_worker *
          FLAGS_num_partition_per_worker;
      const ApplicationId application_id = ApplicationId::FIRST;
      per_app_partition_map_.SetNumVariableGroup(2);
      per_app_partition_map_.SetPartitioning(reduce_variable, 1);
      per_app_partition_map_.SetWorkerId(
          reduce_variable, PartitionId::FIRST, WorkerId::FIRST);
      per_app_partition_map_.SetPartitioning(
          distribute_variable, total_partitions);
      // Set partition map.
      for (int worker_index = 0; worker_index < FLAGS_num_worker;
           ++worker_index) {
             for (int partition_index
                  = worker_index * FLAGS_num_partition_per_worker;
                  partition_index
                  < (worker_index + 1) * FLAGS_num_partition_per_worker;
                  ++partition_index) {
          per_app_partition_map_.SetWorkerId(
              distribute_variable,
              static_cast<PartitionId>(partition_index),
              static_cast<WorkerId>(worker_index));
        }
      }
      // Load application.
      for (int worker_index = 0; worker_index < FLAGS_num_worker;
           ++worker_index) {
        message::WorkerLoadApplication load_application;
        load_application.application_id = application_id;
        send_command_interface_->SendCommandToWorker(
            static_cast<WorkerId>(worker_index),
            message::SerializeMessageWithControlHeader(load_application));
      }
      // Send partition map.
      auto send_map = new PerApplicationPartitionMap(per_app_partition_map_);
      send_command_interface_->AddApplication(application_id, send_map);
      // Load partitions.
      for (int worker_index = 0; worker_index < FLAGS_num_worker;
           ++worker_index) {
        message::WorkerLoadPartitions load_partitions;
        load_partitions.application_id = application_id;
        load_partitions.load_partitions.clear();
        for (int partition_index
             = worker_index * FLAGS_num_partition_per_worker;
             partition_index
             < (worker_index + 1) * FLAGS_num_partition_per_worker;
             ++partition_index) {
          load_partitions.load_partitions.emplace_back(
              distribute_variable, static_cast<PartitionId>(partition_index));
        }
        send_command_interface_->SendCommandToWorker(
            static_cast<WorkerId>(worker_index),
            message::SerializeMessageWithControlHeader(load_partitions));
      }
      {
        message::WorkerLoadPartitions load_partitions;
        load_partitions.application_id = application_id;
        load_partitions.load_partitions.clear();
        load_partitions.load_partitions.emplace_back(
            reduce_variable, PartitionId::FIRST);
        send_command_interface_->SendCommandToWorker(
            WorkerId::FIRST,
            message::SerializeMessageWithControlHeader(load_partitions));
      }
      if (FLAGS_trigger_swap) {
        event_main_thread_->AddDelayInjectedEvent(std::bind(
                &TestControllerScheduler::Switch, this));
      }
    }
  }

  void Switch() {
    PartitionMapUpdate update;
    const auto total_partitions
        = FLAGS_num_worker * FLAGS_num_partition_per_worker;
    const FullPartitionId first_partition{ApplicationId::FIRST,
      distribute_variable, PartitionId::FIRST};
    const auto last_partition_id = static_cast<PartitionId>(total_partitions-1);
    const FullPartitionId second_partition{ApplicationId::FIRST,
      distribute_variable, last_partition_id};
    const auto first_worker_id =
        per_app_partition_map_.QueryWorkerId(
            distribute_variable, PartitionId::FIRST);
    const auto second_worker_id = per_app_partition_map_.QueryWorkerId(
        distribute_variable, last_partition_id);

    update.emplace_back(first_partition, second_worker_id);
    update.emplace_back(second_partition, first_worker_id);
    send_command_interface_->UpdatePartitionMap(new PartitionMapUpdate(update));
    per_app_partition_map_.SetWorkerId(
        distribute_variable, PartitionId::FIRST, second_worker_id);
    per_app_partition_map_.SetWorkerId(
        distribute_variable, last_partition_id, first_worker_id);

    event_main_thread_->AddDelayInjectedEvent(std::bind(
            &TestControllerScheduler::Switch, this));
  }

 private:
  int num_workers_ = 0;
  PerApplicationPartitionMap per_app_partition_map_;
};

class TestWorkerLightThreadContext : public WorkerLightThreadContext {
 public:
  TestWorkerLightThreadContext() {}
  virtual ~TestWorkerLightThreadContext() {}

  //! Initializes the light thread.
  void Initialize() override {}

  //! Finalizes the light thread.
  void Finalize() override {}

  void Run() override {
   struct evbuffer* command;
   StageId command_stage_id;
   if (RetrieveCommand(&command_stage_id, &command)) {
     ProcessCommand(command_stage_id, command);
     return;
   }
   std::list<struct evbuffer*> buffer_list;
   StageId stage_id;
   if (RetrieveData(&stage_id, &buffer_list)) {
     ProcessData(stage_id, &buffer_list);
     return;
   }
 }

 private:
  void ProcessCommand(StageId command_stage_id, struct evbuffer* command) {
    CHECK(command_stage_id < StageId::INVALID);
    switch (command_stage_id) {
      case StageId::INIT:
        if (get_variable_group_id() == reduce_variable) {
          struct evbuffer* buffer = evbuffer_new();
          {
            int data = 1;
            CanaryOutputArchive archive(buffer);
            archive(data);
          }
          get_send_data_interface()->BroadcastDataToPartition(
              get_application_id(), distribute_variable,
              StageId::FIRST, buffer);
          RegisterReceivingData(
              get_next(StageId::FIRST),
              FLAGS_num_worker * FLAGS_num_partition_per_worker);
        } else if (get_variable_group_id() == distribute_variable) {
          RegisterReceivingData(StageId::FIRST, 1);
        }
        break;
      default:
        LOG(FATAL) << "Unknown command stage id!";
    }
    // The command might be empty.
    if (command) {
      evbuffer_free(command);
    }
  }
  void ProcessData(StageId stage_id, std::list<struct evbuffer*>* buffer_list) {
    CHECK(stage_id > StageId::INVALID);
    const auto total_partitions
        = FLAGS_num_worker * FLAGS_num_partition_per_worker;
    if (get_variable_group_id() == reduce_variable) {
      CHECK_EQ(static_cast<int>(buffer_list->size()), total_partitions);
      int data = 0;
      for (auto in_buffer : *buffer_list) {
        int temp = 0;
        {
          CanaryInputArchive archive(in_buffer);
          archive(temp);
        }
        data += temp;
        evbuffer_free(in_buffer);
      }
      data /= total_partitions;
      LOG(INFO) << "Reduces: " << data;
      struct evbuffer* send_buffer = evbuffer_new();
      {
        CanaryOutputArchive archive(send_buffer);
        archive(data);
      }
      get_send_data_interface()->BroadcastDataToPartition(
          ApplicationId::FIRST, distribute_variable,
          get_next(stage_id), send_buffer);
      RegisterReceivingData(get_next(stage_id, 2), total_partitions);
    } else if (get_variable_group_id() == distribute_variable) {
      CHECK_EQ(buffer_list->size(), 1u);
      int data = 0;
      {
        CanaryInputArchive archive(buffer_list->front());
        archive(data);
      }
      evbuffer_free(buffer_list->front());
      struct evbuffer* send_buffer = evbuffer_new();
      {
        CanaryOutputArchive archive(send_buffer);
        archive(data);
      }
      get_send_data_interface()->SendDataToPartition(
          get_application_id(), reduce_variable, PartitionId::FIRST,
          get_next(stage_id), send_buffer);
      RegisterReceivingData(get_next(stage_id, 2), 1);
    }
  }
};

class TestWorkerScheduler : public WorkerSchedulerBase {
 public:
  TestWorkerScheduler() {}
  virtual ~TestWorkerScheduler() {}

  void StartExecution() {
    // Reads global variable: the number of worker threads.
    thread_handle_list_.resize(FLAGS_worker_threads);
    for (auto& handle : thread_handle_list_) {
      // TODO(quhang): set thread priority.
      PCHECK(pthread_create(&handle, nullptr,
                            &WorkerSchedulerBase::ExecutionRoutine, this) == 0);
    }
  }

  void LoadApplicationBinary(ApplicationRecord* application_record) {
    CHECK_NOTNULL(application_record);
    LOG(INFO) << "Test: load application binary.";
  }

  void UnloadApplicationBinary(ApplicationRecord* application_record) {
    CHECK_NOTNULL(application_record);
    LOG(INFO) << "Test: unload application binary.";
  }

  WorkerLightThreadContext* LoadPartition(FullPartitionId) {
    return new TestWorkerLightThreadContext();
  }

  void UnloadPartition(WorkerLightThreadContext* thread_context) {
    CHECK_NOTNULL(thread_context);
    return;
  }
};

void LaunchController() {
  network::EventMainThread event_main_thread;
  ControllerCommunicationManager manager;
  TestControllerScheduler scheduler;
  manager.Initialize(&event_main_thread,
                     &scheduler);  // command receiver.
  scheduler.Initialize(&event_main_thread,  // command sender.
                       &manager);  // data sender.
  // The main thread runs both the manager and the scheduler.
  event_main_thread.Run();
}

void LaunchWorker(int index) {
  network::EventMainThread event_main_thread;
  WorkerCommunicationManager manager;
  TestWorkerScheduler scheduler;
  manager.Initialize(&event_main_thread,
                     &scheduler,
                     &scheduler,
                     FLAGS_controller_host,
                     FLAGS_controller_service,
                     std::to_string(std::stoi(FLAGS_worker_service) + index));
  scheduler.Initialize(&manager, manager.GetDataRouter());
  event_main_thread.Run();
}

int main(int argc, char** argv) {
  InitializeCanaryWorker(&argc, &argv);
  CHECK(!FLAGS_worker_only || !FLAGS_controller_only);
  if (FLAGS_worker_only) {
    LaunchWorker(0);
  } else if (FLAGS_controller_only) {
    LaunchController();
  } else {
    std::thread controller_thread(LaunchController);
    std::list<std::thread> thread_vector;
    for (int i = 0; i < FLAGS_num_worker; ++i) {
      std::thread worker_thread(LaunchWorker, i);
      thread_vector.push_back(std::move(worker_thread));
    }
    controller_thread.join();
  }
  LOG(WARNING) << "Exited.";

  return 0;
}
