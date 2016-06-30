#include <cstdlib>
#include <list>
#include <thread>

#include "shared/internal.h"

#include "shared/initialize.h"
#include "controller/controller_communication_manager.h"
#include "controller/controller_scheduler.h"
#include "worker/worker_communication_manager.h"
#include "worker/worker_light_thread_context.h"
#include "worker/worker_scheduler.h"
#include "message/message_include.h"

DEFINE_int32(num_worker, 2, "Number of workers.");
DEFINE_int32(num_partition_per_worker, 2, "Number of partitions per worker.");

using namespace canary;  // NOLINT

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
    if (++num_workers_ == FLAGS_num_worker) {
      const ApplicationId application_id = ApplicationId::FIRST;
      const VariableGroupId reduce_variable = VariableGroupId::FIRST;
      const VariableGroupId distribute_variable =
          get_next(VariableGroupId::FIRST);
      PerApplicationPartitionMap per_app_partition_map;
      per_app_partition_map.SetNumVariableGroup(2);
      per_app_partition_map.SetPartitioning(reduce_variable, 1);
      per_app_partition_map.SetWorkerId(
          reduce_variable, PartitionId::FIRST, WorkerId::FIRST);
      per_app_partition_map.SetPartitioning(
          distribute_variable,
          FLAGS_num_worker * FLAGS_num_partition_per_worker);
      // Set partition map.
      for (int worker_index = 0;
           worker_index < FLAGS_num_worker;
           ++worker_index) {
        for (int partition_index
             = worker_index * FLAGS_num_partition_per_worker;
            partition_index
            < (worker_index + 1) * FLAGS_num_partition_per_worker;
            ++partition_index) {
          per_app_partition_map.SetWorkerId(
              distribute_variable,
              static_cast<PartitionId>(partition_index),
              static_cast<WorkerId>(worker_index));
        }
      }
      // Load application.
      for (int worker_index = 0;
           worker_index < FLAGS_num_worker;
           ++worker_index) {
        message::WorkerLoadApplication load_application;
        load_application.application_id = application_id;
        send_command_interface_->SendCommandToWorker(
            static_cast<WorkerId>(worker_index),
            message::SerializeMessageWithControlHeader(load_application));
      }
      // Send partition map.
      auto send_map = new PerApplicationPartitionMap(per_app_partition_map);
      send_command_interface_->AddApplication(application_id, send_map);
      // Load partitions.
      for (int worker_index = 0;
           worker_index < FLAGS_num_worker;
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
    }
  }

 private:
  int num_workers_ = 0;
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
    const VariableGroupId reduce_variable = VariableGroupId::FIRST;
    const VariableGroupId distribute_variable =
        get_next(VariableGroupId::FIRST);
    CHECK(command_stage_id < StageId::INVALID);
    switch (command_stage_id) {
      case StageId::INIT:
        if (get_variable_group_id() == reduce_variable) {
          VLOG(3) << "INIT reduce";
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
              StageId::FIRST,
              FLAGS_num_worker * FLAGS_num_partition_per_worker);
          VLOG(3) << "INIT reduce done";
        } else if (get_variable_group_id() == distribute_variable) {
          RegisterReceivingData(StageId::FIRST, 1);
          VLOG(3) << "INIT distribute: " << get_value(get_partition_id());
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
    const VariableGroupId reduce_variable = VariableGroupId::FIRST;
    const VariableGroupId distribute_variable =
        get_next(VariableGroupId::FIRST);
    if (get_variable_group_id() == reduce_variable) {
      CHECK_EQ(static_cast<int>(buffer_list->size()),
               FLAGS_num_worker * FLAGS_num_partition_per_worker);
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
      data /= FLAGS_num_worker * FLAGS_num_partition_per_worker;
      LOG(INFO) << "Reduces: " << data;
      struct evbuffer* send_buffer = evbuffer_new();
      {
        CanaryOutputArchive archive(send_buffer);
        archive(data);
      }
      get_send_data_interface()->BroadcastDataToPartition(
          ApplicationId::FIRST, distribute_variable,
          StageId::FIRST, send_buffer);
      RegisterReceivingData(
          StageId::FIRST,
          FLAGS_num_worker * FLAGS_num_partition_per_worker);
    } else if (get_variable_group_id() == distribute_variable) {
      CHECK_EQ(buffer_list->size(), 1u);
      int data = 0;
      {
        CanaryInputArchive archive(buffer_list->front());
        archive(data);
      }
      LOG(INFO) << "Bounces: " << data;
      evbuffer_free(buffer_list->front());
      struct evbuffer* send_buffer = evbuffer_new();
      {
        CanaryOutputArchive archive(send_buffer);
        archive(data);
      }
      get_send_data_interface()->SendDataToPartition(
          get_application_id(), reduce_variable, PartitionId::FIRST,
          StageId::FIRST, send_buffer);
      RegisterReceivingData(StageId::FIRST, 1);
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

int main(int argc, char** argv) {
  InitializeCanaryWorker(&argc, &argv);

  std::thread controller_thread(
      []{
      network::EventMainThread event_main_thread;
      ControllerCommunicationManager manager;
      TestControllerScheduler scheduler;

      manager.Initialize(&event_main_thread,
                         &scheduler);  // command receiver.
      scheduler.Initialize(&event_main_thread,  // command sender.
                           &manager);  // data sender.
      // The main thread runs both the manager and the scheduler.
      event_main_thread.Run();
      });


  std::list<std::thread> thread_vector;
  for (int i = 0; i < FLAGS_num_worker; ++i) {
    std::thread worker_thread(
        [i] {
        network::EventMainThread event_main_thread;
        WorkerCommunicationManager manager;
        TestWorkerScheduler scheduler;
        manager.Initialize(&event_main_thread, &scheduler, &scheduler,
                           FLAGS_controller_host,
                           FLAGS_controller_service,
                           std::to_string(
                               std::stoi(FLAGS_worker_service) + i));
        scheduler.Initialize(&manager, &manager);
        event_main_thread.Run();
        });
    thread_vector.push_back(std::move(worker_thread));
  }

  controller_thread.join();

  return 0;
}
