#include <cereal/archives/xml.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "canary/canary.h"

#include "cuda_helper.h"

static int FLAG_app_partitions = 1;  // Number of partitions.
static int FLAG_app_iterations = 10;  // Number of iterations.
static int FLAG_app_samples = 1000;  // Number of total samples.
static int FLAG_app_intermediate = 4;  // Number of intermediate combiners.

constexpr int DIMENSION = 20;

namespace canary {

class LogisticLoopApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    typedef std::vector<double> Point;
    const Point reference{1., -1., 1., 1., -1., 2.,  2.,
                          2., 1.,  1., 0., 1.,  -1., 1.,
                          1., 1.,  2., 1., 1.,  1.};

    // Declares variables.

    // Placed on a computation node.
    auto d_x = DeclareVariable<GpuTensorStore<double, 2>>(FLAG_app_partitions);
    auto d_y = DeclareVariable<GpuTensorStore<double, 1>>();
    auto d_local_gradient = DeclareVariable<GpuTensorStore<double, 1>>();
    auto d_local_w = DeclareVariable<GpuTensorStore<double, 1>>();

    // Placed on a parameter node.
    auto d_global_gradient = DeclareVariable<Point>();
    auto d_global_w = DeclareVariable<Point>(1);

    WriteAccess(d_x);
    WriteAccess(d_y);
    Transform([=](CanaryTaskContext* task_context) {
      auto x_data = task_context->WriteVariable(d_x);
      size_t samples_per_partition = FLAG_app_samples / FLAG_app_partitions;
      x_data->Resize({DIMENSION, samples_per_partition});
      auto y_data = task_context->WriteVariable(d_y);
      y_data->Resize({samples_per_partition});
      app::GenerateRandomData(reference, x_data, y_data);
    });

    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      task_context->WriteVariable(d_global_w)->assign(DIMENSION, 1.);
    });

    Loop(FLAG_app_iterations);

    TrackNeeded();
    ReadAccess(d_global_w);
    Scatter([=](CanaryTaskContext* task_context) {
      // Broadcast the global weight.
      task_context->Broadcast(task_context->ReadVariable(d_global_w));
    });

    WriteAccess(d_local_w);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(1);
      // Receive the global weight and load it into GPU.
      std::vector<double> buffer;
      task_context->GatherSingle(&buffer);
      task_context->WriteVariable(d_local_w)->ToDevice(buffer);
      return 0;
    });

    ReadAccess(d_x);
    ReadAccess(d_y);
    ReadAccess(d_local_w);
    WriteAccess(d_local_gradient);
    Transform([=](CanaryTaskContext* task_context) {
      app::UpdateWeightTuned(task_context->ReadVariable(d_x),
                             task_context->ReadVariable(d_y),
                             task_context->ReadVariable(d_local_w),
                             task_context->WriteVariable(d_local_gradient));
    });

    ReadAccess(d_local_gradient);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(
          0, task_context->ReadVariable(d_local_gradient).ToHost());
    });

    WriteAccess(d_global_gradient);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto global_gradient = task_context->WriteVariable(d_global_gradient);
      global_gradient->assign(DIMENSION, 0);
      task_context->Reduce(
          global_gradient,
          [=](const std::vector<double>& left, std::vector<double>* right) {
            for (int i = 0; i < DIMENSION; ++i) { (*right)[i] += left[i]; }
          });
      return 0;
    });

    ReadAccess(d_global_gradient);
    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_gradient = task_context->ReadVariable(d_global_gradient);
      auto global_w = task_context->WriteVariable(d_global_w);
      for (int i = 0; i < DIMENSION; ++i) {
        (*global_w)[i] -= global_gradient[i];
      }
      return 0;
    });

    EndLoop();

    ReadAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_w = task_context->ReadVariable(d_global_w);
      Point output(DIMENSION);
      for (int i = 0; i < DIMENSION; ++i) {
        output[i] = global_w[i] / global_w.front();
      }
      printf("w=");
      for (auto data : output) {
        printf("%f ", data);
      }
      printf("\n");
      fflush(stdout);
    });
  }

  // Loads parameter.
  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::XMLInputArchive archive(ss);
      LoadFlag("partitions", FLAG_app_partitions, archive);
      LoadFlag("iterations", FLAG_app_iterations, archive);
      LoadFlag("samples", FLAG_app_samples, archive);
      LoadFlag("intermediate", FLAG_app_intermediate, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::LogisticLoopApplication);
