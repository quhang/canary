#include <list>
#include <iutility>
#include <vector>

#include <cereal/archives/xml.hpp>
#include "canary/canary.h"

#include "../grid_helper.h"

static int FLAG_app_partition_x = 1;        // Partitioning in x.
static int FLAG_app_partition_y = 1;        // Partitioning in y.
static int FLAG_app_partition_z = 1;        // Partitioning in z.
static int FLAG_app_cell_x = 1;             // Cell in x.
static int FLAG_app_cell_y = 1;             // Cell in y.
static int FLAG_app_cell_z = 1;             // Cell in z.
static int FLAG_app_iterations = 10;             // Number of iterations.
static int FLAG_app_ghost = 1;             // Ghost bandwidth.

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

    typedef std::list<std::pair<int, int>> GobalIndexAndLocalIndex;
    auto d_metadata = DeclareVariable<std::map<int, GobalIndexAndLocalIndex>();

    WriteAccess(d_grid);
    WriteAccess(d_ghost_grid);
    Transform([=](CanaryTaskContext* task_context) {
      auto grid = task_context->WriteVariable(d_grid);
      auto ghost_grid = task_context->WriteVariable(d_ghost_grid);
      ghost_grid->Initialize(
          {{0, 0, 0}, {1, 1, 1}},
          {FLAG_app_cell_x, FLAG_app_cell_y, FLAG_app_cell_z},
          task_context->GetPartitionId(),
          FLAG_app_ghost);
      grid->Initialize(
          {{0, 0, 0}, {1, 1, 1}},
          {FLAG_app_cell_x, FLAG_app_cell_y, FLAG_app_cell_z},
          task_context->GetPartitionId());
    });

    ReadAccess(d_grid);
    ReadAccess(d_ghost_grid);
    WriteAccess(d_data);
    WriteAccess(d_ghost_data);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_ghost_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto data = task_context->WriteVariable(d_data);
      auto ghost_data = task_context->WriteVariable(d_ghost_data);
      data->resize(ghost_grid_->get_count());
      ghost_data->resize(ghost_grid_->get_count());
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
      const auto& grid = task_context->ReadVariable(d_ghost_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto metadata = task_context->WriteVariable(d_metadata);
      for (int di = -FLAG_ghost; di <= FLAG_ghost; ++di)
        for (int dj = -FLAG_ghost; dj <= FLAG_ghost; ++dj)
          for (int dk = -FLAG_ghost; dk <= FLAG_ghost; ++dk) {
            if (di == 0 && dj == 0 && dk == 0) {
              continue;
            }
            helper::Grid neighbor_ghost_grid;
            if (!ghost_grid_.GetNeighborSubgrid(di, dj, dk, FLAG_ghost,
                                                &neighbor_ghost_grid)) {
              continue;
            }
            auto& indices = (*metadata)[neighbor_ghost_grid.GetSubgridRank()];
            for (int iz = ghost_grid.get_sz(); iz < ghost_grid.get_ez(); ++iz)
              for (int iy = ghost_grid.get_sy(); iy < ghost_grid.get_ey(); ++iy)
                for (int ix = ghost_grid.get_sx(); ix < ghost_grid.get_ex();
                     ++ix) {
                  const int global_index =
                      ghost_grid_.GetGlobalCellRank(ix, iy, iz);
                  const int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
                  if (neighbor_ghost_grid.Contain(ix, iy, iz) &&
                      grid.Contain(ix, iy, iz)) {
                      indices.emplace_back(global_index, index);
                    }
                  }
                }
          }
    });

    Loop(FLAG_iterations);

    // Averaging.
    ReadAccess(d_ghost_grid);
    ReadAccess(d_grid);
    WriteAccess(d_data);
    WriteAccess(d_ghost_data);
    Transform([=](CanaryTaskContext* task_context) {
      const auto& grid = task_context->ReadVariable(d_ghost_grid);
      const auto& ghost_grid = task_context->ReadVariable(d_ghost_grid);
      auto data = task_context->WriteVariable(d_data);
      auto ghost_data = task_context->WriteVariable(d_ghost_data);
      std::swap(*data, *ghost_data);
      const int neighbors = FLAG_ghost * FLAG_ghost * FLAG_ghost;
      for (int iz = grid.get_sz(); iz < grid.get_ez(); ++iz)
        for (int iy = grid.get_sy(); iy < grid.get_ey(); ++iy)
          for (int ix = grid.get_sx(); ix < grid.get_ex(); ++ix) {
            const int index = ghost_grid.GetLocalCellRank(ix, iy, iz);
            double temp = 0;
            for (int di = -FLAG_ghost; di <= FLAG_ghost; ++di)
              for (int dj = -FLAG_ghost; dj <= FLAG_ghost; ++dj)
                for (int dk = -FLAG_ghost; dk <= FLAG_ghost; ++dk) {
                  const int temp_index =
                      ghost_grid.GetLocalCellRank(ix + di, iy + dj , iz + dk);
                  temp += (*ghost_data)[temp_index])
                }
            data->at(index) = temp / neighbors;
          }
    });

    EndLoop();
  }
}

}  // namespace canary
