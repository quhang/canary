#include <thread>

#include "shared/initialize.h"
#include "controller/controller_communication_manager.h"

using namespace canary;

class TestReceiver : public ControllerReceiveCommandInterface {
 public:
  void ReceiveCommand(struct evbuffer*) override {
    LOG(INFO) << "Receive command.";
  }
  void NotifyWorkerIsDown(WorkerId worker_id) override {
    LOG(INFO) << "Worker is down:" << get_value(worker_id);
  }
  void NotifyWorkerIsUp(WorkerId worker_id) override {
    LOG(INFO) << "Worker is up:" << get_value(worker_id);
    for (int i = 0; i < 1000; ++i) {
      message::TestWorkerCommand command;
      command.test_string = std::to_string(i);
      auto buffer = message::SerializeControlMessageWithHeader(command);
      manager_->SendCommandToWorker(worker_id, buffer);
    }
  }

  void set_manager(ControllerCommunicationManager* manager) {
    manager_ = manager;
  }

 private:
  ControllerCommunicationManager* manager_ = nullptr;

};

int main(int argc, char* argv[]) {
  InitializeCanaryController(&argc, &argv);

  network::EventMainThread event_main_thread;
  ControllerCommunicationManager manager;
  TestReceiver command_receiver;
  command_receiver.set_manager(&manager);

  manager.Initialize(&event_main_thread, &command_receiver);

  event_main_thread.Run();

  return 0;
}
