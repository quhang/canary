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
    auto in_command =
        message::ControlHeader::UnpackMessage<
        message::MessageCategory::TEST_CONTROLLER_COMMAND>(buffer);
    const WorkerId from_worker_id = in_command->from_worker_id;
    CHECK_EQ(in_command->test_string,
             std::to_string(initial_command_.at(from_worker_id)))
        << get_value(from_worker_id);
    ++initial_command_.at(from_worker_id);
    TestSend(from_worker_id);
    LOG(INFO) << "Send to:" << get_value(from_worker_id) << " value= " <<
        initial_command_[from_worker_id];
    delete in_command;
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
    auto buffer = message::ControlHeader::PackMessage(command);
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
    auto message =
        message::ControlHeader::UnpackMessage<
        message::MessageCategory::TEST_WORKER_COMMAND>(buffer);
    message::TestControllerCommand command;
    command.test_string = message->test_string;
    command.from_worker_id = worker_id_;
    LOG(INFO) << "Worker " << get_value(worker_id_) << " send "
        << message->test_string;
    auto send_buffer = message::ControlHeader::PackMessage(command);
    manager_->SendCommandToController(send_buffer);
    delete message;
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
    auto worker_thread = new std::thread([]{
        WorkerCommunicationManager manager;
        WorkerTestReceiver command_receiver;
        command_receiver.set_manager(&manager);
        network::EventMainThread main_thread;
        manager.Initialize(&main_thread, &command_receiver, nullptr);
        main_thread.Run();
    });
  }

  controller_thread.join();

  return 0;
}
