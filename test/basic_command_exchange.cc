#include <cstdlib>
#include <thread>
#include "shared/internal.h"
#include "shared/initialize.h"

#include "controller/controller_communication_manager.h"
#include "worker/worker_communication_manager.h"

DEFINE_int32(num_worker, 1, "Number of workers.");

using namespace canary;  // NOLINT

class ControllerTestReceiver : public ControllerReceiveCommandInterface {
 public:
  void ReceiveCommand(struct evbuffer* buffer) override {
    {
      auto header = message::StripControlHeader(buffer);
      CHECK(header->category ==
            message::MessageCategory::TEST_CONTROLLER_COMMAND);
      delete header;
    }
    message::TestControllerCommand in_command;
    message::DeserializeMessage(buffer, &in_command);
    const WorkerId from_worker_id = in_command.from_worker_id;
    CHECK_EQ(in_command.test_string,
             std::to_string(initial_command_.at(from_worker_id)))
        << get_value(from_worker_id);
    ++initial_command_.at(from_worker_id);
    TestSend(from_worker_id);
    LOG(INFO) << "Send to:" << get_value(from_worker_id) << " value= " <<
        initial_command_[from_worker_id];
  }
  void NotifyWorkerIsDown(WorkerId worker_id) override {
    LOG(FATAL);
  }
  void NotifyWorkerIsUp(WorkerId worker_id) override {
    CHECK(initial_command_.find(worker_id) == initial_command_.end());
    initial_command_[worker_id] = std::rand();
    TestSend(worker_id);
    LOG(INFO) << "Controller starts sending to: " << get_value(worker_id)
        << " value= " << initial_command_[worker_id];
  }

  void set_manager(ControllerCommunicationManager* manager) {
    manager_ = manager;
  }

 private:
  void TestSend(WorkerId worker_id) {
    message::TestWorkerCommand command;
    command.test_string = std::to_string(initial_command_[worker_id]);
    auto buffer = message::SerializeMessageWithControlHeader(command);
    manager_->SendCommandToWorker(worker_id, buffer);
  }
  ControllerCommunicationManager* manager_ = nullptr;
  std::map<WorkerId, int> initial_command_;
};

class WorkerTestReceiver : public WorkerReceiveCommandInterface {
 public:
  void set_manager(WorkerCommunicationManager* manager) {
    manager_ = manager;
  }
  void ReceiveCommandFromController(struct evbuffer* buffer) override {
    {
      auto header = message::StripControlHeader(buffer);
      CHECK(header->category ==
            message::MessageCategory::TEST_WORKER_COMMAND);
      delete header;
    }
    message::TestWorkerCommand in_command;
    message::DeserializeMessage(buffer, &in_command);

    message::TestControllerCommand out_command;
    out_command.test_string = in_command.test_string;
    out_command.from_worker_id = worker_id_;
    LOG(INFO) << "Worker " << get_value(worker_id_) << " send "
        << out_command.test_string;

    auto send_buffer = message::SerializeMessageWithControlHeader(out_command);
    manager_->SendCommandToController(send_buffer);
  }

  void AssignWorkerId(WorkerId worker_id) {
    LOG(INFO) << "Worker get id:" << get_value(worker_id);
    worker_id_ = worker_id;
  }

 private:
  WorkerCommunicationManager* manager_ = nullptr;
  WorkerId worker_id_ = WorkerId::INVALID;
  int total_commands = 0;
};

class TestDataReceiver : public WorkerReceiveDataInterface {
 public:
  //! Called when receiving data from a partition. The routed data might be
  // rejected, which means this is not the right destination. The header is
  // stripped.
  bool ReceiveRoutedData(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id,
                         struct evbuffer* buffer) override {};
  //! Called when receiving data from a worker, which is sent directly. The
  // header is stripped.
  void ReceiveDirectData(struct evbuffer* buffer) override {};
};

int main(int argc, char* argv[]) {
  InitializeCanaryWorker(&argc, &argv);

  std::thread controller_thread([]{
    network::EventMainThread event_main_thread;
    ControllerCommunicationManager manager;
    ControllerTestReceiver command_receiver;
    command_receiver.set_manager(&manager);
    manager.Initialize(&event_main_thread, &command_receiver);
    event_main_thread.Run();
  });

  for (int i = 0; i < FLAGS_num_worker; ++i) {
    auto worker_thread = new std::thread([i]{
        WorkerCommunicationManager manager;
        WorkerTestReceiver command_receiver;
        TestDataReceiver data_receiver;
        command_receiver.set_manager(&manager);
        network::EventMainThread main_thread;
        manager.Initialize(&main_thread, &command_receiver, &data_receiver,
                           FLAGS_controller_host,
                           FLAGS_controller_service,
                           std::to_string(std::stoi(FLAGS_worker_service) + i));
        main_thread.Run();
    });
  }

  controller_thread.join();

  return 0;
}
