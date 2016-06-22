#include "shared/internal.h"
#include "shared/initialize.h"

#include "worker/worker_communication_manager.h"

using namespace canary;

class TestReceiver : public WorkerReceiveCommandInterface {
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
    message::TestWorkerCommand command;
    message::DeserializeMessage(buffer, &command);
    LOG(INFO) << "Received :" << command.test_string;
  }

  void AssignWorkerId(WorkerId worker_id) override {
    LOG(INFO) << "Assign worker id:" << get_value(worker_id);
  }
 private:
  WorkerCommunicationManager* manager_ = nullptr;
};

int main(int argc, char* argv[]) {
  InitializeCanaryWorker(&argc, &argv);

  WorkerCommunicationManager manager;
  TestReceiver command_receiver;
  command_receiver.set_manager(&manager);
  network::EventMainThread main_thread;
  manager.Initialize(&main_thread, &command_receiver, nullptr);
  main_thread.Run();
  return 0;
}
