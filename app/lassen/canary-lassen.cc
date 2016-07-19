#include <algorithm>
#include <array>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "canary/canary.h"

#include "simulation-mpi.h"

static int FLAG_app_partition_x = 1;  // Partitioning in x.
static int FLAG_app_partition_y = 1;  // Partitioning in y.
static int FLAG_app_partition_z = 1;  // Partitioning in z.
static int FLAG_app_cell_x = 10;      // Cell in x.
static int FLAG_app_cell_y = 10;      // Cell in y.
static int FLAG_app_cell_z = 10;      // Cell in z.
static int FLAG_app_iterations = 10;  // Number of iterations.

using ::Lassen::SimulationMPI;

struct GlobalState {
  std::array<double, 4> extreme_buffer;
  double dt = 0;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(extreme_buffer, dt);
  }
};

namespace canary {
class LassenApplication : public CanaryApplication {
 public:
  typedef std::array<double, 4> FourDoubleArray;
  static void array_pointwise_min(const FourDoubleArray& input,
                                  FourDoubleArray* output) {
    for (int i = 0; i < (int)input.size(); ++i) {
      (*output)[i] = std::min(input[i], (*output)[i]);
    }
  }

  //! The program.
  void Program() override {
    const auto NUM_PARTITION =
        FLAG_app_partition_x * FLAG_app_partition_y * FLAG_app_partition_z;
    auto d_partition = DeclareVariable<SimulationMPI>(NUM_PARTITION);
    auto d_global = DeclareVariable<GlobalState>(1);

    // Statement 0.
    // Initializes simulation object.
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      sim->metadata.initialize(NUM_PARTITION, task_context->GetPartitionId(),
                               FLAG_app_partition_x, FLAG_app_partition_y,
                               FLAG_app_partition_z, FLAG_app_cell_x,
                               FLAG_app_cell_y, FLAG_app_cell_z);
      CHECK(sim->metadata.checkConsistency());
      sim->Setup();
      sim->functionLocalInitialize();
    });

    // Reduction. For the global extreme.
    // Statement 1
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      FourDoubleArray buffer;
      sim.functionSendSyncExtreme(buffer.data());
      task_context->Scatter(0, buffer);
    });

    // Statement 2.
    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto global = task_context->WriteVariable(d_global);
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto& reduce_result = global->extreme_buffer;
      std::fill(reduce_result.begin(), reduce_result.end(),
                std::numeric_limits<double>::max());
      task_context->Reduce(&reduce_result, array_pointwise_min);
      return 0;
    });

    // Broadcast. For the extreme.
    // Statement 3.
    ReadAccess(d_global);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& global = task_context->ReadVariable(d_global);
      task_context->Broadcast(global.extreme_buffer);
    });

    // Statement 4.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      EXPECT_GATHER_SIZE(1);
      FourDoubleArray buffer;
      task_context->GatherSingle(&buffer);
      sim->functionRecvSyncExtreme(buffer.data());
      return 0;
    });

    // Statement 5.
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      sim->functionInitializeBoundaryNodes();
      sim->functionInitializeCommunicatingNodes();
    });

    // Neighboring.
    // Statement 6.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      std::vector<long> send_buffer;
      sim.functionSendInitializeNodeCommunication(send_buffer);
      for (int index : sim.domain->neighborDomains) {
        task_context->OrderedScatter(index, send_buffer);
      }
    });

    // Statement 7.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->domain->neighborDomains;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer = task_context->OrderedGather<std::vector<long> >();
      std::vector<std::vector<long> > sort_buffer;
      for (auto index : index_array) {
        sort_buffer.emplace_back(std::move(recv_buffer.at(index)));
      }
      sim->functionRecvInitializeNodeCommunication(sort_buffer);
      return 0;
    });

    // N-to-N.
    // Statement 8.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      Lassen::BoundingBox boundingBox;
      sim.functionSendFirstInitializeFacetCommunication(boundingBox);
      for (int i = 0; i < task_context->GetGatherParallelism(); ++i) {
        task_context->OrderedScatter(i, boundingBox);
      }
    });

    // Statement 9.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      auto recv_buffer = task_context->OrderedGather<Lassen::BoundingBox>();
      sim->allBoundingBox.resize(recv_buffer.size());
      for (auto& pair : recv_buffer) {
        sim->allBoundingBox.at(pair.first) = std::move(pair.second);
      }
      return 0;
    });

    // Neighboring.
    // Statement 10.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      std::vector<Lassen::Point> send_buffer(sim.boundaryNodes.size());
      sim.functionSendSecondInitializeFacetCommunication(send_buffer);
      for (int index : sim.overlapDomains) {
        task_context->OrderedScatter(index, send_buffer);
      }
    });

    // Statement 11.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->overlapDomains;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer =
          task_context->OrderedGather<std::vector<Lassen::Point> >();
      std::vector<std::vector<Lassen::Point> > sort_buffer;
      for (auto index : index_array) {
        sort_buffer.emplace_back(std::move(recv_buffer.at(index)));
      }
      sim->functionRecvSecondInitializeFacetCommunication(sort_buffer);
      return 0;
    });

    // Neighboring.
    // Statement 12.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      const auto& index_array = sim.overlapDomains;
      sim.functionSendThirdInitializeFacetCommunication();
      for (size_t i = 0; i < index_array.size(); ++i) {
        task_context->OrderedScatter(index_array[i], sim.isFacetNeighbor[i]);
      }
    });

    // Statement 13.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->overlapDomains;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer = task_context->OrderedGather<int>();
      std::vector<int> recvIsFacetNeighbor;
      for (auto index : index_array) {
        recvIsFacetNeighbor.emplace_back(recv_buffer.at(index));
      }
      sim->functionRecvThirdInitializeFacetCommunication(recvIsFacetNeighbor);
      return 0;
    });

    // Statement 14.
    Loop(FLAG_app_iterations);

    // Statement 15.
    TrackNeeded();
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      sim->functionPrepareStep();
      sim->functionUpdateSources();
      sim->functionUpdateNarrowband();
      sim->functionUpdateNearbyNodes();
      sim->functionConstructFront();
    });

    // Neighboring.
    // Statement 16.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      const auto& index_array = sim.facetNeighbors;
      std::vector<std::vector<double> > send_buffer;
      sim.functionSendCommunicateFront(send_buffer);
      for (size_t index = 0; index < index_array.size(); ++index) {
        task_context->OrderedScatter(index_array[index], send_buffer[index]);
      }
    });

    // Statement 17.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->facetNeighbors;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer = task_context->OrderedGather<std::vector<double> >();
      std::vector<std::vector<double> > sort_buffer;
      for (auto index : index_array) {
        sort_buffer.emplace_back(std::move(recv_buffer.at(index)));
      }
      sim->functionRecvCommunicateFront(sort_buffer);
      return 0;
    });

    // Statment 18.
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      sim->functionConstructDistancesToFront();
      sim->functionConvertNearbyNodesToNarrowBandNodes();
      sim->functionComputeNextTimeStep();
    });

    // Reduce.
    // Statement 19.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      task_context->Scatter(0, sim.dt);
    });

    // Statement 20.
    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto global = task_context->WriteVariable(d_global);
      EXPECT_GATHER_SIZE(NUM_PARTITION);
      global->dt = task_context->Reduce(
          std::numeric_limits<double>::max(),
          [](double x, double y) { return std::min(x, y); });
      return 0;
    });

    // Broadcast.
    // Statement 21.
    ReadAccess(d_global);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& global = task_context->ReadVariable(d_global);
      task_context->Broadcast(global.dt);
    });

    // Statement 22.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      EXPECT_GATHER_SIZE(1);
      task_context->GatherSingle(&sim->dt);
      return 0;
    });

    // Statement 23.
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      sim->functionComputeTimeIntegral();
    });

    // Neighboring.
    // Statement 24.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      const auto& index_array = sim.facetNeighbors;
      std::vector<std::vector<double> > send_buffer;
      sim.functionSendCommunicateFront(send_buffer);
      for (size_t index = 0; index < index_array.size(); ++index) {
        task_context->OrderedScatter(index_array[index], send_buffer[index]);
      }
    });

    // Statement 25.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->facetNeighbors;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer = task_context->OrderedGather<std::vector<double> >();
      std::vector<std::vector<double> > sort_buffer;
      for (auto index : index_array) {
        sort_buffer.emplace_back(std::move(recv_buffer.at(index)));
      }
      sim->functionRecvCommunicateFront(sort_buffer);
      return 0;
    });

    // Statement 26.
    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& sim = task_context->ReadVariable(d_partition);
      const auto& index_array = sim.domain->neighborDomains;
      std::vector<std::vector<Lassen::NodeData> > send_buffer;
      sim.functionSendSynchronizeNodeData(send_buffer);
      CHECK_EQ(send_buffer.size(), index_array.size());
      for (size_t index = 0; index < index_array.size(); ++index) {
        task_context->OrderedScatter(index_array[index], send_buffer[index]);
      }
    });

    // Statement 27.
    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto sim = task_context->WriteVariable(d_partition);
      auto& index_array = sim->domain->neighborDomains;
      EXPECT_GATHER_SIZE(index_array.size());
      auto recv_buffer =
          task_context->OrderedGather<std::vector<Lassen::NodeData> >();
      std::vector<std::vector<Lassen::NodeData> > sort_buffer;
      for (auto index : index_array) {
        sort_buffer.emplace_back(std::move(recv_buffer.at(index)));
      }
      sim->functionRecvSynchronizeNodeData(sort_buffer);
      return 0;
    });

    // Statement 28.
    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      int localDone = 0;
      sim->functionSendDoneCondition(localDone);
      // The local done is discarded.
    });

    EndLoop();

    // Statement 29.
    WriteAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      auto sim = task_context->WriteVariable(d_partition);
      FourDoubleArray buffer;
      sim->functionSendComputeError(&buffer[0], &buffer[1], &buffer[2],
                                    &buffer[3]);
      task_context->Scatter(0, buffer);
    });

    // Statement 30.
    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      std::array<double, 4> result;
      std::fill(result.begin(), result.end(), 0);
      std::vector<FourDoubleArray> recv_buffer =
          task_context->Gather<FourDoubleArray>();
      for (auto& element : recv_buffer) {
        result[0] += element[0];
        result[1] += element[1];
        result[2] = std::max(result[2], element[2]);
        result[3] += element[3];
      }
      result[0] /= result[3];
      result[1] /= result[3];
      std::cout << "==========================================================="
                   "=====================\n";
      std::cout << "nodeCount  = " << std::round(result[3]) << "\n";
      std::cout << "L1   error = " << result[0] << "\n";
      std::cout << "L2   error = " << result[1] << "\n";
      std::cout << "LInf error = " << result[2] << "\n";
      std::cout
          << "===========================================================";
      return 0;
    });
  }

  // Loads parameter.
  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::XMLInputArchive archive(ss);
      LoadFlag("partition_x", FLAG_app_partition_x, archive);
      LoadFlag("partition_y", FLAG_app_partition_y, archive);
      LoadFlag("partition_z", FLAG_app_partition_z, archive);
      LoadFlag("cell_x", FLAG_app_cell_x, archive);
      LoadFlag("cell_y", FLAG_app_cell_y, archive);
      LoadFlag("cell_z", FLAG_app_cell_z, archive);
      LoadFlag("iterations", FLAG_app_iterations, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::LassenApplication);
