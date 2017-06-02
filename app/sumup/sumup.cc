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
static int FLAG_app_intermediate = 4;  // Number of intermediate combiners.
static long long FLAG_app_doubles = 100; // Number of doubles to sum.

namespace canary {

/**
 * A microbenchmark that iteratively broadcasts a value and sums up the values.
 */
class BarrierTestApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    // Declares variables.
    auto d_component = DeclareVariable<std::vector<double>>(FLAG_app_partitions);
    auto d_inter = DeclareVariable<double>(FLAG_app_intermediate);
    auto d_sum = DeclareVariable<double>(1);

    WriteAccess(d_sum);
    Transform([=](CanaryTaskContext* task_context) {
      *task_context->WriteVariable(d_sum) = 1;
    });

    WriteAccess(d_component);
    Transform([=](CanaryTaskContext* task_context) {
      task_context->WriteVariable(d_component)->assign(
          FLAG_app_doubles / FLAG_app_partitions, 1);
    });

    Loop(FLAG_app_iterations);

    TrackNeeded();
    ReadAccess(d_sum);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Broadcast(task_context->ReadVariable(d_sum));
    });

    WriteAccess(d_component);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(1);
      double temp = 0;
      task_context->GatherSingle(&temp);
      return 0;
    });

    // Layered reduction.
    ReadAccess(d_component);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& vec = task_context->ReadVariable(d_component);
      auto size = vec.size();
      auto ptr = vec.data();
      double sum = 0;
      const auto end = ptr + size;
      auto p = ptr;
      while (p < end) {
        sum += *(p++);
      }
      task_context->Scatter(
          task_context->GetPartitionId() % task_context->GetGatherParallelism(),
          sum);
    });

    WriteAccess(d_inter);
    Gather([=](CanaryTaskContext* task_context) -> int {
      const int remain = task_context->GetScatterParallelism() %
                         task_context->GetGatherParallelism();
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism() /
                         task_context->GetGatherParallelism() +
                         (task_context->GetPartitionId() < remain ? 1 : 0));
      double* sum = task_context->WriteVariable(d_inter);
      *sum = task_context->Reduce(double(0), std::plus<double>());
      return 0;
    });

    ReadAccess(d_inter);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(0, task_context->ReadVariable(d_inter));
    });

    WriteAccess(d_sum);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      double* sum = task_context->WriteVariable(d_sum);
      *sum = task_context->Reduce(double(0), std::plus<double>());
      printf("%.9f %f\n",
             time::timepoint_to_double(time::WallClock::now()),
             *sum);
      fflush(stdout);
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
      LoadFlag("intermediate", FLAG_app_intermediate, archive);
      LoadFlag("doubles", FLAG_app_doubles, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::BarrierTestApplication);
