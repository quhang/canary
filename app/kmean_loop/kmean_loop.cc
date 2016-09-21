#include <cereal/archives/xml.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "canary/canary.h"
#include "../helper.h"

static int FLAG_app_partitions = 1;   // Number of partitions.
static int FLAG_app_iterations = 10;  // Number of iterations.
static int FLAG_app_samples = 1000;    // Number of total samples.
static double FLAG_app_rates = -1;    // The task execution rate per worker.

constexpr int DIMENSION = 20;
constexpr int NUM_CLUSTER = 10;

namespace {

//! Finds the closest cluster.
template <typename T, size_t SIZE, size_t CLUSTER_SIZE>
int find_closest_cluster(
    const std::array<T, SIZE>& point,
    const std::array<std::array<T, SIZE>, CLUSTER_SIZE>& cluster_center) {
  int index = 0;
  T min_distance = helper::array_distance(point, cluster_center[0]);
  for (int i = 1; i < (int)CLUSTER_SIZE; ++i) {
    T temp_distance = helper::array_distance(point, cluster_center[i]);
    if (temp_distance < min_distance) {
      min_distance = temp_distance;
      index = i;
    }
  }
  return index;
}

//! Merges cluster stat.
template <typename T, size_t SIZE, size_t CLUSTER_SIZE>
void merge_cluster_stat(
    const std::array<std::pair<int, std::array<T, SIZE>>, CLUSTER_SIZE>& left,
    std::array<std::pair<int, std::array<T, SIZE>>, CLUSTER_SIZE>* right) {
  auto left_iter = left.cbegin();
  auto right_iter = right->begin();
  while (left_iter != left.cend()) {
    right_iter->first += left_iter->first;
    helper::array_add(left_iter->second, &right_iter->second);
    ++left_iter;
    ++right_iter;
  }
}

//! Transforms stats to centers.
template <typename T, size_t SIZE, size_t CLUSTER_SIZE>
void transform_stat_to_center(
    const std::array<std::pair<int, std::array<T, SIZE>>, CLUSTER_SIZE>& left,
    std::array<std::array<T, SIZE>, CLUSTER_SIZE>* right) {
  auto left_iter = left.cbegin();
  auto right_iter = right->begin();
  while (left_iter != left.cend()) {
    if (left_iter->first != 0) {
      helper::array_mul(left_iter->second, 1. / left_iter->first,
                        &(*right_iter));
    }
    ++left_iter;
    ++right_iter;
  }
}

}  // namespace

namespace canary {

class KmeanLoopApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    rate_limiter_.Initialize(FLAG_app_rates);
    typedef std::array<double, DIMENSION> Point;
    typedef std::vector<Point> PointVector;
    typedef std::array<std::pair<int, Point>, NUM_CLUSTER> ClusterStat;
    typedef std::array<Point, NUM_CLUSTER> ClusterCenter;

    // Declares variables.
    auto d_point = DeclareVariable<PointVector>(FLAG_app_partitions);
    auto d_local_stat = DeclareVariable<ClusterStat>();
    auto d_local_center = DeclareVariable<ClusterCenter>();
    auto d_global_stat = DeclareVariable<ClusterStat>(1);
    auto d_global_center = DeclareVariable<ClusterCenter>();

    WriteAccess(d_point);
    Transform([=](CanaryTaskContext* task_context) {
      auto points = task_context->WriteVariable(d_point);
      points->resize(FLAG_app_samples / FLAG_app_partitions);
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::uniform_real_distribution<> dis(-1, 1);
      auto generator = [&dis, &gen] { return dis(gen); };
      for (auto& point : *points) {
        std::generate_n(point.begin(), point.size(), generator);
      }
    });

    WriteAccess(d_global_center);
    Transform([=](CanaryTaskContext* task_context) {
      auto global_center = task_context->WriteVariable(d_global_center);
      std::random_device rd;
      std::default_random_engine gen(rd());
      std::uniform_real_distribution<> dis(-1, 1);
      auto generator = [&dis, &gen] { return dis(gen); };
      for (auto& center : *global_center) {
        std::generate_n(center.begin(), center.size(), generator);
      }
    });

    Loop(FLAG_app_iterations);

    TrackNeeded();
    ReadAccess(d_global_center);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Broadcast(task_context->ReadVariable(d_global_center));
    });

    WriteAccess(d_local_center);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(1);
      task_context->GatherSingle(task_context->WriteVariable(d_local_center));
      return 0;
    });

    ReadAccess(d_point);
    ReadAccess(d_local_center);
    WriteAccess(d_local_stat);
    Transform([=](CanaryTaskContext* task_context) {
      this->rate_limiter_.Join();
      const auto& points = task_context->ReadVariable(d_point);
      const auto& local_center = task_context->ReadVariable(d_local_center);
      auto local_stat = task_context->WriteVariable(d_local_stat);
      for (auto& cluster_stat : *local_stat) {
        cluster_stat.first = 0;
        std::fill(cluster_stat.second.begin(), cluster_stat.second.end(), 0);
      }
      for (const auto& point : points) {
        const int cluster_id = find_closest_cluster(point, local_center);
        ++(*local_stat)[cluster_id].first;
        helper::array_add(point, &(*local_stat)[cluster_id].second);
      }
    });

    ReadAccess(d_local_stat);
    Scatter([=](CanaryTaskContext* task_context) {
      task_context->Scatter(0, task_context->ReadVariable(d_local_stat));
    });

    WriteAccess(d_global_stat);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto global_stat = task_context->WriteVariable(d_global_stat);
      for (auto& pair : *global_stat) {
        pair.first = 0;
        std::fill(pair.second.begin(), pair.second.end(), 0);
      }
      task_context->Reduce(global_stat,
                           merge_cluster_stat<double, DIMENSION, NUM_CLUSTER>);
      return 0;
    });

    ReadAccess(d_global_stat);
    WriteAccess(d_global_center);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_stat = task_context->ReadVariable(d_global_stat);
      auto global_center = task_context->WriteVariable(d_global_center);
      transform_stat_to_center(global_stat, global_center);
      return 0;
    });

    EndLoop();

    ReadAccess(d_global_center);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& global_center = task_context->ReadVariable(d_global_center);
      printf("c0=");
      for (auto point : global_center) {
        printf("%f ", point[0]);
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
      LoadFlag("rates", FLAG_app_rates, archive);
    }
  }

 private:
  RateLimiter rate_limiter_;
};

}  // namespace canary

REGISTER_APPLICATION(::canary::KmeanLoopApplication);
