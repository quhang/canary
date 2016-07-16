#include <list>
#include <map>
#include <utility>
#include <vector>

#include <cereal/archives/xml.hpp>
#include "canary/canary.h"

#include "../grid_helper.h"

static int FLAG_app_partition_x = 1;  // Partitioning in x.
static int FLAG_app_partition_y = 1;  // Partitioning in y.
static int FLAG_app_partition_z = 1;  // Partitioning in z.
static int FLAG_app_cell_x = 10;      // Cell in x.
static int FLAG_app_cell_y = 10;      // Cell in y.
static int FLAG_app_cell_z = 10;      // Cell in z.
static int FLAG_app_iterations = 10;  // Number of iterations.
static int FLAG_app_ghost = 1;        // Ghost bandwidth.

namespace canary {
class StencilApplication : public CanaryApplication {
 public:
  //! The program.
  void Program() override {
    const auto NUM_PARTITION =
        FLAG_app_partition_x * FLAG_app_partition_y * FLAG_app_partition_z;
    auto d_ghost_grid = DeclareVariable<helper::Grid>(NUM_PARTITION);
    auto d_ghost_data = DeclareVariable<std::vector<double>>();

    auto d_grid = DeclareVariable<helper::Grid>();
    auto d_data = DeclareVariable<std::vector<double>>();

    typedef std::list<std::pair<int, int>> GlobalRankAndLocalRank;
    auto d_metadata = DeclareVariable<std::map<int, GlobalRankAndLocalRank>>();

    auto d_global = DeclareVariable<int>(1);

    WriteAccess(d_grid);
    WriteAccess(d_ghost_grid);
    Transform([=](CanaryTaskContext* task_context) {
      auto grid = task_context->WriteVariable(d_grid);
      auto ghost_grid = task_context->WriteVariable(d_ghost_grid);
      ghost_grid->Initialize(
          {{0, 0, 0}, {1, 1, 1}},
          {FLAG_app_cell_x, FLAG_app_cell_y, FLAG_app_cell_z},
          {FLAG_app_partition_x, FLAG_app_partition_y, FLAG_app_partition_z},
          task_context->GetPartitionId(), FLAG_app_ghost);
      grid->Initialize(
          {{0, 0, 0}, {1, 1, 1}},
          {FLAG_app_cell_x, FLAG_app_cell_y, FLAG_app_cell_z},
          {FLAG_app_partition_x, FLAG_app_partition_y, FLAG_app_partition_z},
          task_context->GetPartitionId());
    });

    ReadAccess(d_grid);
    ReadAccess(d_ghost_grid);
    WriteAccess(d_data);
    WriteAccess(d_ghost_data);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto data = task_context->WriteVariable(d_data);
      auto ghost_data = task_context->WriteVariable(d_ghost_data);
      data->resize(ghost_grid.get_count());
      ghost_data->resize(ghost_grid.get_count());
      for (int iz = grid.get_sz(); iz < grid.get_ez(); ++iz)
        for (int iy = grid.get_sy(); iy < grid.get_ey(); ++iy)
          for (int ix = grid.get_sx(); ix < grid.get_ex(); ++ix) {
            const int index = ghost_grid.GetLocalCellRank(ix, iy, iz);
            data->at(index) = ix + iy + iz;
          }
    });

    ReadAccess(d_ghost_grid);
    ReadAccess(d_grid);
    WriteAccess(d_metadata);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto metadata = task_context->WriteVariable(d_metadata);
      for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj)
          for (int dk = -1; dk <= 1; ++dk) {
            if (di == 0 && dj == 0 && dk == 0) {
              continue;
            }
            helper::Grid neighbor_ghost_grid;
            if (!ghost_grid.GetNeighborSubgrid(di, dj, dk, FLAG_app_ghost,
                                               &neighbor_ghost_grid)) {
              continue;
            }
            auto& indices = (*metadata)[neighbor_ghost_grid.GetSubgridRank()];
            for (int iz = grid.get_sz(); iz < grid.get_ez(); ++iz)
              for (int iy = grid.get_sy(); iy < grid.get_ey(); ++iy)
                for (int ix = grid.get_sx(); ix < grid.get_ex(); ++ix) {
                  if (neighbor_ghost_grid.Contain(ix, iy, iz)) {
                    const int global_index =
                        ghost_grid.GetGlobalCellRank(ix, iy, iz);
                    const int index = ghost_grid.GetLocalCellRank(ix, iy, iz);
                    indices.emplace_back(global_index, index);
                  }
                }
          }
    });

    Loop(FLAG_app_iterations);

    ReadAccess(d_data);
    ReadAccess(d_metadata);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& data = task_context->ReadVariable(d_data);
      const auto& metadata = task_context->ReadVariable(d_metadata);
      for (const auto& pair : metadata) {
        // Rank => List(global_index, local_index)
        struct evbuffer* buffer = evbuffer_new();
        CanaryOutputArchive archive(buffer);
        archive(pair.second.size());
        for (const auto& inpair : pair.second) {
          archive(inpair.first, data[inpair.second]);
        }
        task_context->Scatter(pair.first, RawEvbuffer(buffer));
      }
    });

    ReadAccess(d_ghost_grid);
    ReadAccess(d_metadata);
    WriteAccess(d_data);
    Gather([=](CanaryTaskContext* task_context) -> int {
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      const auto& metadata = task_context->ReadVariable(d_metadata);
      auto data = task_context->WriteVariable(d_data);
      EXPECT_GATHER_SIZE(metadata.size());
      auto recv_buffer = task_context->Gather<RawEvbuffer>();
      for (auto& raw_buffer : recv_buffer) {
        CanaryInputArchive archive(raw_buffer.buffer);
        size_t ghost_cell_size;
        archive(ghost_cell_size);
        for (size_t i = 0; i < ghost_cell_size; ++i) {
          int global_index;
          double value;
          archive(global_index, value);
          (*data)[ghost_grid.GlobalCellRankToLocal(global_index)] = value;
        }
        evbuffer_free(raw_buffer.buffer);
      }
      return 0;
    });

    // Averaging.
    ReadAccess(d_ghost_grid);
    ReadAccess(d_grid);
    WriteAccess(d_data);
    WriteAccess(d_ghost_data);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto data = task_context->WriteVariable(d_data);
      auto ghost_data = task_context->WriteVariable(d_ghost_data);
      std::swap(*data, *ghost_data);
      for (int iz = grid.get_sz(); iz < grid.get_ez(); ++iz)
        for (int iy = grid.get_sy(); iy < grid.get_ey(); ++iy)
          for (int ix = grid.get_sx(); ix < grid.get_ex(); ++ix) {
            const int index = ghost_grid.GetLocalCellRank(ix, iy, iz);
            double temp = 0;
            int neighbors = 0;
            for (int di = -FLAG_app_ghost; di <= FLAG_app_ghost; ++di)
              for (int dj = -FLAG_app_ghost; dj <= FLAG_app_ghost; ++dj)
                for (int dk = -FLAG_app_ghost; dk <= FLAG_app_ghost; ++dk) {
                  if (!ghost_grid.Contain(ix + di, iy + dj, iz + dk)) {
                    continue;
                  }
                  const int temp_index =
                      ghost_grid.GetLocalCellRank(ix + di, iy + dj, iz + dk);
                  temp += (*ghost_data)[temp_index];
                  ++neighbors;
                }
            data->at(index) = temp / neighbors;
          }
    });

    EndLoop();

    ReadAccess(d_data);
    ReadAccess(d_grid);
    ReadAccess(d_ghost_grid);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      const auto& data = task_context->ReadVariable(d_data);
      double partial_sum = 0;
      for (int iz = grid.get_sz(); iz < grid.get_ez(); ++iz)
        for (int iy = grid.get_sy(); iy < grid.get_ey(); ++iy)
          for (int ix = grid.get_sx(); ix < grid.get_ex(); ++ix) {
            const int index = ghost_grid.GetLocalCellRank(ix, iy, iz);
            partial_sum += data[index];
          }
      task_context->Scatter(0, partial_sum);
    });

    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      double sum = 0;
      sum = task_context->Reduce(sum, std::plus<double>());
      printf("Sum: %f Reference: %d\n", sum,
             FLAG_app_cell_x * FLAG_app_cell_y * FLAG_app_cell_z *
                 (FLAG_app_cell_x + FLAG_app_cell_y + FLAG_app_cell_z - 3) / 2);
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
      LoadFlag("cell_x", FLAG_app_cell_z, archive);
      LoadFlag("cell_y", FLAG_app_cell_z, archive);
      LoadFlag("cell_z", FLAG_app_cell_z, archive);
      LoadFlag("iterations", FLAG_app_iterations, archive);
      LoadFlag("ghost", FLAG_app_ghost, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::StencilApplication);
