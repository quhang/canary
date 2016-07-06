#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
#include <cereal/archives/binary.hpp>

#include "canary/canary.h"

DEFINE_int32(app_partitions, 2, "Number of partitions.");
DEFINE_int32(app_iterations, 50, "Number of iterations.");
DEFINE_int32(app_samples, 100, "Number of total samples.");

constexpr int DIMENSION = 20;

namespace {

// Helper function for dot multiply.
template <typename T, size_t size>
T array_dot(const std::array<T, size>& left, const std::array<T, size>& right) {
  T result = 0;
  auto left_iter = left.cbegin();
  auto right_iter = right.cbegin();
  while (left_iter != left.cend() && right_iter != right.cend()) {
    result += (*left_iter) * (*right_iter);
    ++left_iter;
    ++right_iter;
  }
  return result;
}

// Helper function: output += input_multi * input.
template <typename T, size_t size>
void array_accumulate(const std::array<T, size>& input, T input_multi,
                      std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend() && output_iter != output->end()) {
    *output_iter += input_multi*(*input_iter);
    ++input_iter;
    ++output_iter;
  }
}

// Helper function: output += input_multi * input.
template <typename T, size_t size>
void array_substract(const std::array<T, size>& input,
                     std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend() && output_iter != output->end()) {
    *output_iter -= *input_iter;
    ++input_iter;
    ++output_iter;
  }
}

// Helper function: output += input_multi * input.
template <typename T, size_t size>
void array_add(const std::array<T, size>& input,
               std::array<T, size>* output) {
  auto input_iter = input.cbegin();
  auto output_iter = output->begin();
  while (input_iter != input.cend() && output_iter != output->end()) {
    *output_iter += *input_iter;
    ++input_iter;
    ++output_iter;
  }
}

}  // namespace

namespace canary {

class LogisticLoopApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    typedef std::array<double, DIMENSION> Point;
    typedef std::vector<std::pair<Point, bool>> FeatureVector;
    constexpr Point reference{1.f, -1.f, 1.f, 1.f, -1.f, 2.f,  2.f,
                              2.f, 1.f,  1.f, 0.f, 1.f,  -1.f, 1.f,
                              1.f, 1.f,  2.f, 1.f, 1.f,  1.f};

    // Declares variables.
    auto d_feature = DeclareVariable<FeatureVector>(FLAGS_app_partitions);
    auto d_local_gradient = DeclareVariable<Point>();
    auto d_local_w = DeclareVariable<Point>();
    auto d_global_w = DeclareVariable<Point>();
    auto d_global_gradient = DeclareVariable<Point>();

    WriteAccess(d_feature);
    Transform([=](CanaryTaskContext* task_context) {
      auto feature = task_context->WriteVariable(d_feature);
      feature->resize(FLAGS_app_samples / FLAGS_app_partitions);
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::uniform_real_distribution<> dis(-1, 1);
      auto generator = [&dis, &gen] { return dis(gen); };
      for (auto& pair : *feature) {
        Point& point = pair.first;
        std::generate_n(point.begin(), point.size(), generator);
        pair.second = (array_dot(point, reference) > 0);
      }
    });

    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      auto global_w = task_context->WriteVariable(d_global_w);
      std::fill(global_w->begin(), global_w->end(), 0);
    });

    WriteAccess(d_global_gradient);
    Transform([=](CanaryTaskContext* task_context) {
      auto global_gradient = task_context->WriteVariable(d_global_gradient);
      std::fill(global_gradient->begin(), global_gradient->end(), 1000);
    });

    Loop(FLAGS_app_iterations);

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
      const auto& feature = task_context->ReadVariable(d_feature);
      const auto& local_w = task_context->ReadVariable(d_local_w);
      auto local_gradient = task_context->WriteVariable(d_local_gradient);
      std::fill(local_gradient->begin(), local_gradient->end(), 0);
      for (const auto& pair : feature) {
        const Point& point = pair.first;
        const bool flag = pair.second;
        const double multi =
            flag ? +(1. / (1. + std::exp(-array_dot(local_w, point))) - 1.)
                 : -(1. / (1. + std::exp(+array_dot(local_w, point))) - 1.);
        array_accumulate(point, multi, local_gradient);
      }
    });

    ReadAccess(d_local_gradient);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(0, task_context->ReadVariable(d_local_gradient));
    });

    WriteAccess(d_global_gradient);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto global_gradient = task_context->WriteVariable(d_global_gradient);
      std::fill(global_gradient->begin(), global_gradient->end(), 0);
      task_context->Reduce(global_gradient, array_add<double, DIMENSION>);
      return 0;
    });

    ReadAccess(d_global_gradient);
    WriteAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_gradient
          = task_context->ReadVariable(d_global_gradient);
      auto global_w = task_context->WriteVariable(d_global_w);
      array_substract(global_gradient, global_w);
      return 0;
    });

    EndLoop();

    ReadAccess(d_global_w);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_w = task_context->ReadVariable(d_global_w);
      Point output;
      std::fill(output.begin(), output.end(), 0);
      array_accumulate(global_w, 1. / global_w.front(), &output);
      printf("w=");
      for (auto data : output) {
        printf("%f ", data);
      }
      printf("\n");
    });
  }

  // Loads parameter.
  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::BinaryInputArchive archive(ss);
      archive(FLAGS_app_partitions);
      archive(FLAGS_app_iterations);
      archive(FLAGS_app_samples);
    }
  }

  // Saves parameter.
  std::string SaveParameter() override {
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive archive(ss);
      archive(FLAGS_app_partitions);
      archive(FLAGS_app_iterations);
      archive(FLAGS_app_samples);
    }
    return ss.str();
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::LogisticLoopApplication);
