#include <cereal/archives/xml.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "canary/canary.h"
#include "../helper.h"

static int FLAG_app_partitions = 1;  // Number of partitions.
static int FLAG_app_iterations = 10;  // Number of iterations.
static int FLAG_app_samples = 1000;  // Number of total samples.
static int FLAG_app_intermediate = 4;  // Number of intermediate combiners.
static double FLAG_app_rates = -1;    // The task execution rate per worker.

constexpr int DIMENSION = 20;

namespace canary {

class LogisticLoopApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    rate_limiter_.Initialize(FLAG_app_rates);
    typedef std::array<double, DIMENSION> Point;
    typedef std::vector<std::pair<Point, bool>> FeatureVector;
    constexpr Point reference{1.f, -1.f, 1.f, 1.f, -1.f, 2.f,  2.f,
                              2.f, 1.f,  1.f, 0.f, 1.f,  -1.f, 1.f,
                              1.f, 1.f,  2.f, 1.f, 1.f,  1.f};

    // Declares variables.
    auto d_feature = DeclareVariable<FeatureVector>(FLAG_app_partitions);
    auto d_local_gradient = DeclareVariable<Point>();
    auto d_local_w = DeclareVariable<Point>();
    auto d_inter_gradient = DeclareVariable<Point>(FLAG_app_intermediate);
    auto d_global_gradient = DeclareVariable<Point>();
    auto d_global_w = DeclareVariable<Point>(1);

    WriteAccess(d_feature);
    Transform([=](CanaryTaskContext* task_context) {
      auto feature = task_context->WriteVariable(d_feature);
      feature->resize(FLAG_app_samples / FLAG_app_partitions);
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::uniform_real_distribution<> dis(-1, 1);
      auto generator = [&dis, &gen] { return dis(gen); };
      for (auto& pair : *feature) {
        Point& point = pair.first;
        std::generate_n(point.begin(), point.size() - 1, generator);
        point.back() = 1;
        pair.second = (helper::array_dot(point, reference) > 0);
      }
    });

    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      auto global_w = task_context->WriteVariable(d_global_w);
      std::fill(global_w->begin(), global_w->end(), 0);
    });

    Loop(FLAG_app_iterations);

    TrackNeeded();
    ReadAccess(d_global_w);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Broadcast(task_context->ReadVariable(d_global_w));
    });

    WriteAccess(d_local_w);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(1);
      task_context->GatherSingle(task_context->WriteVariable(d_local_w));
      return 0;
    });

    ReadAccess(d_feature);
    ReadAccess(d_local_w);
    WriteAccess(d_local_gradient);
    Transform([=](CanaryTaskContext* task_context) {
      this->rate_limiter_.Join();
      const auto& feature = task_context->ReadVariable(d_feature);
      const auto& local_w = task_context->ReadVariable(d_local_w);
      auto local_gradient = task_context->WriteVariable(d_local_gradient);
      std::fill(local_gradient->begin(), local_gradient->end(), 0);
      for (const auto& pair : feature) {
        const Point& point = pair.first;
        const bool flag = pair.second;
        const auto dot = helper::array_dot(local_w, point);
        const double factor =
            flag ? +(1. / (1. + std::exp(-dot)) - 1.)
                 : -(1. / (1. + std::exp(+dot)) - 1.);
        helper::array_acc(point, factor, local_gradient);
      }
    });

    // Layered reduction.
    ReadAccess(d_local_gradient);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(
          task_context->GetPartitionId() % task_context->GetGatherParallelism(),
          task_context->ReadVariable(d_local_gradient));
    });

    WriteAccess(d_inter_gradient);
    Gather([=](CanaryTaskContext* task_context) -> int {
      const int remain = task_context->GetScatterParallelism() %
                         task_context->GetGatherParallelism();
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism() /
                         task_context->GetGatherParallelism() +
                         (task_context->GetPartitionId() < remain ? 1 : 0));
      auto inter_gradient = task_context->WriteVariable(d_inter_gradient);
      std::fill(inter_gradient->begin(), inter_gradient->end(), 0);
      task_context->Reduce(inter_gradient,
                           helper::array_add<double, DIMENSION>);
      return 0;
    });

    ReadAccess(d_inter_gradient);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(0, task_context->ReadVariable(d_inter_gradient));
    });

    WriteAccess(d_global_gradient);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto global_gradient = task_context->WriteVariable(d_global_gradient);
      std::fill(global_gradient->begin(), global_gradient->end(), 0);
      task_context->Reduce(global_gradient,
                           helper::array_add<double, DIMENSION>);
      return 0;
    });

    ReadAccess(d_global_gradient);
    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_gradient =
          task_context->ReadVariable(d_global_gradient);
      auto global_w = task_context->WriteVariable(d_global_w);
      helper::array_sub(global_gradient, global_w);
      return 0;
    });

    EndLoop();

    ReadAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_w = task_context->ReadVariable(d_global_w);
      Point output;
      std::fill(output.begin(), output.end(), 0);
      helper::array_acc(global_w, 1. / global_w.front(), &output);
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
      LoadFlag("rates", FLAG_app_rates, archive);
    }
  }

 private:
  RateLimiter rate_limiter_;
};

}  // namespace canary

REGISTER_APPLICATION(::canary::LogisticLoopApplication);
