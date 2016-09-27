#include <cereal/archives/xml.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "canary/canary.h"

static int FLAG_app_partitions = 2;   // Number of partitions.
static int FLAG_app_iterations = 10;  // Number of iterations.

namespace canary {

class ShuffleTestApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    // Declares variables.
    auto d_component = DeclareVariable<std::vector<int>>(FLAG_app_partitions);

    WriteAccess(d_component);
    Transform([=](CanaryTaskContext* task_context) {
      auto input = task_context->WriteVariable(d_component);
      for (int i = 0; i < FLAG_app_partitions; ++i) {
        input->push_back(i);
      }
    });

    Loop(FLAG_app_iterations);

    ReadAccess(d_component);
    Scatter([=](CanaryTaskContext* task_context) {
      std::vector<std::vector<int>> buffer(FLAG_app_partitions);
      for (auto number : task_context->ReadVariable(d_component)) {
        buffer[number % FLAG_app_partitions].push_back(number);
      }
      for (int i = 0; i < FLAG_app_partitions; ++i) {
        task_context->Scatter(i, buffer[i]);
      }
      if (task_context->GetPartitionId() == 0) {
        printf("%.9f\n",
               time::timepoint_to_double(time::WallClock::now()));
        fflush(stdout);
      }
    });

    WriteAccess(d_component);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(FLAG_app_partitions);
      auto result = task_context->Gather<std::vector<int>>();
      std::vector<int> shuffled_result;
      for (auto& vector : result) {
        shuffled_result.insert(shuffled_result.end(),
                               vector.begin(), vector.end());
      }
      std::sort(shuffled_result.begin(), shuffled_result.end());
      return 0;
    });

    EndLoop();
  }

  // Loads parameter.
  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::XMLInputArchive archive(ss);
      LoadFlag("partitions", FLAG_app_partitions, archive);
      LoadFlag("iterations", FLAG_app_iterations, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::ShuffleTestApplication);
