/**
 * Tests a controll sending FLAGS_num_command test commands to each of the
 * worker.
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <list>
#include <thread>
#include "shared/internal.h"
#include "shared/initialize.h"

#include "controller/controller_communication_manager.h"
#include "worker/worker_communication_manager.h"

DEFINE_int32(num_worker, 10, "Number of workers.");
DEFINE_int32(num_command, 100, "Number of commands.");

using namespace canary;  // NOLINT

class TestControllerReceiver : public ControllerReceiveCommandInterface {
 public:
  void ReceiveCommand(struct evbuffer* buffer) override {
    std::string test_string;
    WorkerId from_worker_id;
    ++total_commands_;
    CHECK_LE(total_commands_, FLAGS_num_worker * FLAGS_num_command);
    AnalyzeMessage(buffer, &test_string, &from_worker_id);
    CHECK_EQ(test_string, std::to_string(ongoing_sequence_.at(from_worker_id)))
        << "Wrong message from worker " << get_value(from_worker_id);

    ++ongoing_sequence_.at(from_worker_id);
    if (sent_commands_.at(from_worker_id) == FLAGS_num_command) {
      LOG(INFO) << "Down W" << get_value(from_worker_id);
      manager_->ShutDownWorker(from_worker_id);
    } else {
      ++sent_commands_.at(from_worker_id);
      SendMessage(from_worker_id);
      LOG(INFO) << "C to W" << get_value(from_worker_id)
          << " : " << ongoing_sequence_[from_worker_id];
    }
  }

  void NotifyWorkerIsDown(WorkerId worker_id) override {
    ++done_workers_;
    if (done_workers_ == FLAGS_num_worker) {
      CHECK_EQ(FLAGS_num_worker * FLAGS_num_command, total_commands_);
      manager_->Finalize();
      success_ = true;
    }
  }

  void NotifyWorkerIsUp(WorkerId worker_id) override {
    CHECK(ongoing_sequence_.find(worker_id) == ongoing_sequence_.end());
    ongoing_sequence_[worker_id] = std::rand();
    sent_commands_[worker_id] = 1;
    SendMessage(worker_id);
    LOG(INFO) << "Up W" << get_value(worker_id);
    LOG(INFO) << "C to W" << get_value(worker_id)
        << " : " << ongoing_sequence_[worker_id];
  }

  void set_manager(ControllerCommunicationManager* manager) {
    manager_ = manager;
  }

  bool get_success() {
    return success_;
  }

 private:
  void AnalyzeMessage(struct evbuffer* buffer,
                      std::string* result_string,
                      WorkerId* result_worker_id) {
    auto header = message::StripControlHeader(buffer);
    CHECK(header->category ==
          message::MessageCategory::TEST_CONTROLLER_COMMAND);
    delete header;
    message::TestControllerCommand command;
    message::DeserializeMessage(buffer, &command);
    *result_string = command.test_string;
    *result_worker_id = command.from_worker_id;
  }
  void SendMessage(WorkerId worker_id) {
    message::TestWorkerCommand command;
    command.test_string = std::to_string(ongoing_sequence_[worker_id]);
    auto buffer = message::SerializeMessageWithControlHeader(command);
    manager_->SendCommandToWorker(worker_id, buffer);
  }
  ControllerCommunicationManager* manager_ = nullptr;
  std::map<WorkerId, int> ongoing_sequence_;
  std::map<WorkerId, int> sent_commands_;
  int total_commands_ = 0;
  int done_workers_ = 0;
  bool success_ = false;
};

class TestWorkerReceiver : public WorkerReceiveCommandInterface {
 public:
  void set_manager(WorkerCommunicationManager* manager) {
    manager_ = manager;
  }
  void ReceiveCommandFromController(struct evbuffer* buffer) override {
    std::string test_string;
    AnalyzeMessage(buffer, &test_string);

    message::TestControllerCommand out_command;
    out_command.test_string = test_string;
    out_command.from_worker_id = worker_id_;
    LOG(INFO) << "W" << get_value(worker_id_) << " to C: " << test_string;

    auto send_buffer = message::SerializeMessageWithControlHeader(out_command);
    manager_->SendCommandToController(send_buffer);
  }

  void AssignWorkerId(WorkerId worker_id) {
    worker_id_ = worker_id;
  }

 private:
  void AnalyzeMessage(struct evbuffer* buffer,
                      std::string* result_string) {
    auto header = message::StripControlHeader(buffer);
    CHECK(header->category ==
          message::MessageCategory::TEST_WORKER_COMMAND);
    delete header;
    message::TestWorkerCommand command;
    message::DeserializeMessage(buffer, &command);
    *result_string = command.test_string;
  }

  WorkerCommunicationManager* manager_ = nullptr;
  WorkerId worker_id_ = WorkerId::INVALID;
};

class TestDataReceiver : public WorkerReceiveDataInterface {
 public:
  //! Called when receiving data from a partition. The routed data might be
  // rejected, which means this is not the right destination. The header is
  // stripped.
  bool ReceiveRoutedData(ApplicationId application_id,
                         VariableGroupId variable_group_id,
                         PartitionId partition_id,
                         StageId stage_id,
                         struct evbuffer* buffer) override {};
  //! Called when receiving data from a worker, which is sent directly. The
  // header is stripped.
  void ReceiveDirectData(struct evbuffer* buffer) override {};
};

TEST(basic, basic_command_exchange) {
  // Pointers are used to avoid deallocation.
  std::thread controller_thread([]{
    auto event_main_thread = new network::EventMainThread();
    auto manager = new ControllerCommunicationManager();
    auto command_receiver = new TestControllerReceiver();
    command_receiver->set_manager(manager);
    manager->Initialize(event_main_thread, command_receiver);
    event_main_thread->Run();
    LOG(INFO) << "Exits controller thread.";
    EXPECT_TRUE(command_receiver->get_success());
  });

  std::list<std::thread> thread_vector;
  for (int i = 0; i < FLAGS_num_worker; ++i) {
    thread_vector.emplace_back([i]{
        auto event_main_thread = new network::EventMainThread();
        auto manager = new WorkerCommunicationManager();
        auto command_receiver = new TestWorkerReceiver();
        auto data_receiver = new TestDataReceiver();
        command_receiver->set_manager(manager);
        manager->Initialize(event_main_thread,
                            command_receiver, data_receiver,
                            FLAGS_controller_host,
                            FLAGS_controller_service,
                            std::to_string(
                                std::stoi(FLAGS_worker_service) + i));
        event_main_thread->Run();
        LOG(INFO) << "Exit worker thread " << i;
    });
  }
  for (auto& thread_handle : thread_vector) {
    thread_handle.join();
  }
  controller_thread.join();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::canary::InitializeCanaryWorker(&argc, &argv);
  return RUN_ALL_TESTS();
}
