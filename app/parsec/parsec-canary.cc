// Code written by Richard O. Lee and Christian Bienia
// Modified by Christian Fensch

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <utility>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <float.h>


#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include "canary/canary.h"

#include "fluid.h"
#include "cellpool.h"

DEFINE_int32(app_partition_x, 1, "Partitioning in the x dimension.");
DEFINE_int32(app_partition_y, 1, "Partitioning in the y dimension.");
DEFINE_int32(app_partition_z, 1, "Partitioning in the z dimension.");
DEFINE_int32(app_fold_x, 1, "Folding in the x dimension.");
DEFINE_int32(app_fold_y, 1, "Folding in the y dimension.");
DEFINE_int32(app_fold_z, 1, "Folding in the z dimension.");
DEFINE_int32(app_depth, 1, "Folding depth in the y dimension.");
DEFINE_int32(app_frames, 10, "Partitioning in the z dimension.");
DEFINE_string(app_filename, "", "Input file name.");

struct GlobalState {
  template <typename Archive> void serialize(Archive&) {}
};

/**
 * The grid that describes a region of cells.
 */
struct Grid {
  // The region.
  int sx, sy, sz, ex, ey, ez;
  // d[x] = e[x] - s[x].
  int dx, dy, dz;
  // The cell count.
  int count;
  void CalculateLength() {
    dx = ex - sx;
    dy = ey - sy;
    dz = ez - sz;
    count = dx * dy * dz;
  }
  void ExpandToGhostGrid(int lim_x, int lim_y, int lim_z) {
    if (sx > 0) {
      --sx;
    }
    if (sy > 0) {
      --sy;
    }
    if (sz > 0) {
      --sz;
    }
    if (ex < lim_x - 1) {
      ++ex;
    }
    if (ey < lim_y - 1) {
      ++ey;
    }
    if (ez < lim_z - 1) {
      ++ez;
    }
    CalculateLength();
  }
  bool Contain(int ix, int iy, int iz) const {
    return (ix >= sx) && (ix < ex) && (iy >= sy) && (iy < ey) && (iz >= sz) &&
           (iz < ez);
  }

  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(sx, sy, sz, ex, ey, ez, dx, dy, dz, count);
  }
};

/*
 * Data of a partition
 */
class PartitionData {
 public:
  //! Allocates the cell pool in the constructor.
  PartitionData();
  //! Deallocate the cell pool in the destructor.
  virtual ~PartitionData();

 private:
  cellpool local_pool_;

  float energy_sum_ = 0;

  // Simulation parameters.
  fptype restParticlesPerMeter_ = 0, h_ = 0, hSq_ = 0;
  fptype densityCoeff_ = 0, pressureCoeff_ = 0, viscosityCoeff_ = 0;

  // Number of cells in each each dimension.
  int nx_ = 0, ny_ = 0, nz_ = 0;

  // The size of a cell.
  Vec3 delta_;

  // Number of partitioning in each dimension.
  int split_x_ = 0, split_y_ = 0, split_z_ = 0, split_total_ = 0;
  // The index of the partition.
  int partition_x_ = 0, partition_y_ = 0, partition_z_ = 0, partition_rank_ = 0;

  std::vector<Cell> cells_, cells2_;
  std::vector<int> cnumPars_, cnumPars2_;
  std::vector<Cell*> last_cells_;
  // Rank -> Metadata.
  struct Metadata {
    // Global index/local index.
    std::list<std::pair<int, int>> send_to_owner_local_indices;
    std::list<std::pair<int, int>> send_to_ghost_local_indices;
    template <typename Archive>
    void serialize(Archive& archive) {  // NOLINT
      archive(send_to_owner_local_indices);
      archive(send_to_ghost_local_indices);
    }
  };
  // Caution: has to be ordered map.
  std::map<int, Metadata> exchange_metadata_;
  int total_neighbor_partitions_ = 0;

  Grid local_grid_, ghost_grid_;

  Vec3 domainMinFull, domainMaxFull;

 public:
  template <typename Archive>
  void save(Archive& archive) const {  // NOLINT
    archive(energy_sum_);
    archive(restParticlesPerMeter_, h_, hSq_);
    archive(densityCoeff_, pressureCoeff_, viscosityCoeff_);
    archive(nx_, ny_, nz_);
    archive(delta_);
    archive(split_x_, split_y_, split_z_, split_total_);
    archive(partition_x_, partition_y_, partition_z_, partition_rank_);

    archive(cells_.size());
    for (int index = 0; index < (int)cells_.size(); ++index) {
      SerializeFullCell(archive, index);
    }

    archive(exchange_metadata_);
    archive(total_neighbor_partitions_);
    archive(local_grid_, ghost_grid_);
    archive(domainMinFull, domainMaxFull);
  }

  template <typename Archive>
  void load(Archive& archive) {  // NOLINT
    archive(energy_sum_);
    archive(restParticlesPerMeter_, h_, hSq_);
    archive(densityCoeff_, pressureCoeff_, viscosityCoeff_);
    archive(nx_, ny_, nz_);
    archive(delta_);
    archive(split_x_, split_y_, split_z_, split_total_);
    archive(partition_x_, partition_y_, partition_z_, partition_rank_);

    size_t size_cells;
    archive(size_cells);
    cells_.resize(size_cells);
    cnumPars_.resize(size_cells, 0);
    last_cells_.resize(size_cells, nullptr);
    for (int index = 0; index < (int)cells_.size(); ++index) {
      last_cells_[index] = &cells_[index];
      DeserializeFullCell(archive, index);
    }

    cells2_.resize(size_cells);
    cnumPars2_.resize(size_cells, 0);

    archive(exchange_metadata_);
    archive(total_neighbor_partitions_);
    archive(local_grid_, ghost_grid_);
    archive(domainMinFull, domainMaxFull);
  }
  int GetNumNeighbors() { return exchange_metadata_.size(); }

 protected:
  int GetLocalIndex(int global_i, int global_j, int global_k) const {
    CHECK(ghost_grid_.Contain(global_i, global_j, global_k));
    return (global_i - ghost_grid_.sx) +
           ((global_j - ghost_grid_.sy) +
            (global_k - ghost_grid_.sz) * ghost_grid_.dy) *
               ghost_grid_.dx;
  }
  int GetGlobalIndex(int global_i, int global_j, int global_k) const {
    return global_i + (global_j + global_k * ny_) * nx_;
  }
  int GetLocalIndexFromGlobalIndex(int global_index) const {
    const int nxy = nx_ * ny_;
    const int global_k = global_index / nxy;
    global_index -= global_k * nxy;
    const int global_j = global_index / nx_;
    global_index -= global_j * nx_;
    return GetLocalIndex(global_index, global_j, global_k);
  }
  int SampleInterval(int num_interval, int num_length, int index) const {
    return index * (num_length / num_interval) +
           std::min(index, num_length % num_interval);
  }
  template <typename Archive>
  void SerializeFullCell(Archive& archive, int index) const;
  template <typename Archive>
  void DeserializeFullCell(Archive& archive, int index);
  template <typename Archive>
  void SerializeDensity(Archive& archive, int index) const;
  template <typename Archive>
  void DeserializeDensity(Archive& archive, int index);

 public:
  void InitSim(char const* fileName, int split_x, int split_y, int split_z,
               int partition_rank);
  void ClearParticlesMT();
  void RebuildGridMT();
  int InitNeighCellList(int ci, int cj, int ck, int* neighCells) const;
  void ComputeDensitiesMT();
  void ComputeDensities2MT();
  void ComputeForcesMT();
  void ProcessCollisionsMT();
  void AdvanceParticlesMT();
  void ProcessCollisions2MT();
  void DataSwap() {
    std::swap(cells_, cells2_);
    std::swap(cnumPars_, cnumPars2_);
  }
  float ComputeEnergy();

  void SendExchangeGhostCells(std::vector<int>* neighbors,
                              std::vector<std::vector<char>>* send_buffer);
  void RecvExchangeGhostCells(
      const std::vector<std::vector<char>>& recv_buffer);
  void SendDensityGhost(std::vector<int>* neighbors,
                        std::vector<std::vector<char>>* send_buffer);
  void RecvDensityGhost(const std::vector<std::vector<char>>& recv_buffer);
};

void PartitionData::InitSim(
    char const* fileName, int split_x, int split_y, int split_z,
    int partition_rank) {
  split_x_ = split_x;
  split_y_ = split_y;
  split_z_ = split_z;
  partition_rank_ = partition_rank;
  split_total_ = split_x * split_y * split_z;
  CHECK_GE(split_x_, 1);
  CHECK_GE(split_y_, 1);
  CHECK_GE(split_z_, 1);
  CHECK_GE(partition_rank_, 0);
  CHECK_GT(split_total_, partition_rank);

  // Load input particles
  std::ifstream file(fileName, std::ios::binary);
  CHECK(file) << "Error opening file.";

  // Read numParticles and resetParticlesPerMeter.
  float restParticlesPerMeter_le;
  int numParticles_le;
  int numParticles;
  file.read((char*)&restParticlesPerMeter_le, FILE_SIZE_FLOAT);
  file.read((char*)&numParticles_le, FILE_SIZE_INT);
  if (!isLittleEndian()) {
    restParticlesPerMeter_ = bswap_float(restParticlesPerMeter_le);
    numParticles = bswap_int32(numParticles_le);
  } else {
    restParticlesPerMeter_ = restParticlesPerMeter_le;
    numParticles = numParticles_le;
  }

  // Initialize the pool.
  // cellpool_init(&local_pool_, numParticles / split_total_);

  // Initialize simulation parameters.
  h_ = kernelRadiusMultiplier / restParticlesPerMeter_;
  hSq_ = h_ * h_;
  fptype coeff1 = 315.0 / (64.0 * pi * powf(h_, 9.0));
  fptype coeff2 = 15.0 / (pi * powf(h_, 6.0));
  fptype coeff3 = 45.0 / (pi * powf(h_, 6.0));
  fptype particleMass = 0.5 * doubleRestDensity /
                        (restParticlesPerMeter_ * restParticlesPerMeter_ *
                         restParticlesPerMeter_);
  densityCoeff_ = particleMass * coeff1;
  pressureCoeff_ = 3.0 * coeff2 * 0.50 * stiffnessPressure * particleMass;
  viscosityCoeff_ = viscosity * coeff3 * particleMass;

  // Initialize simulation sizes.
  domainMaxFull.x =
      FLAGS_app_fold_x * (domainMaxPart.x - domainMinPart.x) + domainMinPart.x;
  domainMaxFull.y =
      FLAGS_app_fold_y * (domainMaxPart.y - domainMinPart.y) + domainMinPart.y;
  domainMaxFull.z =
      FLAGS_app_fold_z * (domainMaxPart.z - domainMinPart.z) + domainMinPart.z;
  domainMinFull = domainMinPart;
  Vec3 range = domainMaxFull - domainMinFull;
  nx_ = (int)(range.x / h_);
  ny_ = (int)(range.y / h_);
  nz_ = (int)(range.z / h_);
  CHECK_GE(nx_, 1);
  CHECK_GE(ny_, 1);
  CHECK_GE(nz_, 1);
  delta_.x = range.x / nx_;
  delta_.y = range.y / ny_;
  delta_.z = range.z / nz_;
  CHECK_GE(delta_.x, h_);
  CHECK_GE(delta_.y, h_);
  CHECK_GE(delta_.z, h_);
  CHECK_GE(nx_, split_x_);
  CHECK_GE(ny_, split_y_);
  CHECK_GE(nz_, split_z_);

  int temp_partition_rank = partition_rank;
  partition_z_ = temp_partition_rank / (split_x_ * split_y_);
  temp_partition_rank -= partition_z_ * (split_x_ * split_y_);
  partition_y_ = temp_partition_rank / split_x_;
  temp_partition_rank -= partition_y_ * split_x_;
  partition_x_ = temp_partition_rank;
  local_grid_.sx = SampleInterval(split_x_, nx_, partition_x_);
  local_grid_.ex = SampleInterval(split_x_, nx_, partition_x_ + 1);
  local_grid_.sy = SampleInterval(split_y_, ny_, partition_y_);
  local_grid_.ey = SampleInterval(split_y_, ny_, partition_y_ + 1);
  local_grid_.sz = SampleInterval(split_z_, nz_, partition_z_);
  local_grid_.ez = SampleInterval(split_z_, nz_, partition_z_ + 1);
  local_grid_.CalculateLength();
  ghost_grid_ = local_grid_;
  ghost_grid_.ExpandToGhostGrid(nx_, ny_, nz_);

  // printf("Partition#%d: [%d-%d,%d-%d,%d-%d] out of %d*%d*%d\n",
  //        partition_rank_,
  //        local_grid_.sx, local_grid_.ex,
  //        local_grid_.sy, local_grid_.ey,
  //        local_grid_.sz, local_grid_.ez,
  //        nx_, ny_, nz_);

  total_neighbor_partitions_ = 0;
  {
    ++total_neighbor_partitions_;
    auto& metadata = exchange_metadata_[partition_rank_];
    for (int iz = ghost_grid_.sz; iz < ghost_grid_.ez; ++iz)
      for (int iy = ghost_grid_.sy; iy < ghost_grid_.ey; ++iy)
        for (int ix = ghost_grid_.sx; ix < ghost_grid_.ex; ++ix) {
          if (!local_grid_.Contain(ix, iy, iz)) {
            int global_index = GetGlobalIndex(ix, iy, iz);
            int index = GetLocalIndex(ix, iy, iz);
            metadata.send_to_ghost_local_indices.emplace_back(global_index,
                                                              index);
          }
        }
  }

  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        if (di == 0 && dj == 0 && dk == 0) {
          continue;
        }
        int neighbor_x = partition_x_ + di;
        int neighbor_y = partition_y_ + dj;
        int neighbor_z = partition_z_ + dk;
        if (neighbor_x < 0 || neighbor_x >= split_x_ || neighbor_y < 0 ||
            neighbor_y >= split_y_ || neighbor_z < 0 ||
            neighbor_z >= split_z_) {
          continue;
        }
        ++total_neighbor_partitions_;
        Grid neighbor_grid;
        neighbor_grid.sx = SampleInterval(split_x_, nx_, neighbor_x);
        neighbor_grid.ex = SampleInterval(split_x_, nx_, neighbor_x + 1);
        neighbor_grid.sy = SampleInterval(split_y_, ny_, neighbor_y);
        neighbor_grid.ey = SampleInterval(split_y_, ny_, neighbor_y + 1);
        neighbor_grid.sz = SampleInterval(split_z_, nz_, neighbor_z);
        neighbor_grid.ez = SampleInterval(split_z_, nz_, neighbor_z + 1);
        neighbor_grid.CalculateLength();
        Grid neighbor_ghost_grid = neighbor_grid;
        neighbor_ghost_grid.ExpandToGhostGrid(nx_, ny_, nz_);
        int neighbor_rank =
            neighbor_x + (neighbor_y + neighbor_z * split_y_) * split_x_;
        auto& metadata = exchange_metadata_[neighbor_rank];
        for (int iz = ghost_grid_.sz; iz < ghost_grid_.ez; ++iz)
          for (int iy = ghost_grid_.sy; iy < ghost_grid_.ey; ++iy)
            for (int ix = ghost_grid_.sx; ix < ghost_grid_.ex; ++ix) {
              int global_index = GetGlobalIndex(ix, iy, iz);
              int index = GetLocalIndex(ix, iy, iz);
              if (neighbor_ghost_grid.Contain(ix, iy, iz)) {
                if (local_grid_.Contain(ix, iy, iz)) {
                  metadata.send_to_owner_local_indices.emplace_back(
                      global_index, index);
                } else {
                  metadata.send_to_ghost_local_indices.emplace_back(
                      global_index, index);
                }
              }
            }
      }

  // make sure Cell structure is multiple of estiamted cache line size
  static_assert(sizeof(Cell) % CACHELINE_SIZE == 0, "wrong padding");
  // make sure helper Cell structure is in sync with real Cell structure
  static_assert(
      offsetof(struct Cell_aux, padding) == offsetof(struct Cell, padding),
      "wrong padding");

  cells_.resize(ghost_grid_.count);
  cells2_.resize(ghost_grid_.count);
  cnumPars_.resize(ghost_grid_.count, 0);
  cnumPars2_.resize(ghost_grid_.count, 0);
  last_cells_.resize(ghost_grid_.count, nullptr);

  // Always use single precision float variables b/c file format uses single
  // precision float
  float unfold_px, unfold_py, unfold_pz, hvx, hvy, hvz, vx, vy, vz;
  float sum = 0;
  for (int i = 0; i < numParticles; ++i) {
    file.read((char*)&unfold_px, FILE_SIZE_FLOAT);
    file.read((char*)&unfold_py, FILE_SIZE_FLOAT);
    file.read((char*)&unfold_pz, FILE_SIZE_FLOAT);
    file.read((char*)&hvx, FILE_SIZE_FLOAT);
    file.read((char*)&hvy, FILE_SIZE_FLOAT);
    file.read((char*)&hvz, FILE_SIZE_FLOAT);
    file.read((char*)&vx, FILE_SIZE_FLOAT);
    file.read((char*)&vy, FILE_SIZE_FLOAT);
    file.read((char*)&vz, FILE_SIZE_FLOAT);
    if (!isLittleEndian()) {
      unfold_px = bswap_float(unfold_px);
      unfold_py = bswap_float(unfold_py);
      unfold_pz = bswap_float(unfold_pz);
      hvx = bswap_float(hvx);
      hvy = bswap_float(hvy);
      hvz = bswap_float(hvz);
      vx = bswap_float(vx);
      vy = bswap_float(vy);
      vz = bswap_float(vz);
    }

    for (int fold_x = 0; fold_x < FLAGS_app_fold_x; ++fold_x)
      for (int fold_y = 0; fold_y < FLAGS_app_depth; ++fold_y)
        for (int fold_z = 0; fold_z < FLAGS_app_fold_z; ++fold_z) {
          float px = unfold_px + fold_x * (domainMaxPart.x - domainMinPart.x);
          float py = unfold_py + fold_y * (domainMaxPart.y - domainMinPart.y);
          float pz = unfold_pz + fold_z * (domainMaxPart.z - domainMinPart.z);
          int ci = (int)((px - domainMinFull.x) / delta_.x);
          int cj = (int)((py - domainMinFull.y) / delta_.y);
          int ck = (int)((pz - domainMinFull.z) / delta_.z);

          if (ci < 0)
            ci = 0;
          else if (ci > (nx_ - 1))
            ci = nx_ - 1;
          if (cj < 0)
            cj = 0;
          else if (cj > (ny_ - 1))
            cj = ny_ - 1;
          if (ck < 0)
            ck = 0;
          else if (ck > (nz_ - 1))
            ck = nz_ - 1;

          if (!local_grid_.Contain(ci, cj, ck)) {
            continue;
          }
          int index = GetLocalIndex(ci, cj, ck);
          Cell* cell = &cells_[index];

          // go to last cell structure in list
          int np = cnumPars_[index];
          while (np > PARTICLES_PER_CELL) {
            cell = cell->next;
            np = np - PARTICLES_PER_CELL;
          }
          // add another cell structure if everything full
          if ((np % PARTICLES_PER_CELL == 0) && (cnumPars_[index] != 0)) {
            // Get cells from pools in round-robin fashion to balance load
            // during
            // parallel phase
            cell->next = cellpool_getcell(&local_pool_);
            cell = cell->next;
            np = np - PARTICLES_PER_CELL;
          }

          cell->p[np].x = px;
          cell->p[np].y = py;
          cell->p[np].z = pz;
          cell->hv[np].x = hvx;
          cell->hv[np].y = hvy;
          cell->hv[np].z = hvz;
          cell->v[np].x = vx;
          cell->v[np].y = vy;
          cell->v[np].z = vz;
          sum += vx * vx + vy * vy + vz * vz;
          ++cnumPars_[index];
        }
  }
}

////////////////////////////////////////////////////////////////////////////////
PartitionData::PartitionData() {
  cellpool_init(&local_pool_, 1000);
}

PartitionData::~PartitionData() {
  // first return extended cells to cell pools
  for (auto& cell : cells_) {
    while (cell.next) {
      Cell* temp = cell.next;
      cell.next = temp->next;
      cellpool_returncell(&local_pool_, temp);
    }
  }
  cellpool_destroy(&local_pool_);
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Clear the first array. The first array should be empty.
 */
void PartitionData::ClearParticlesMT() {
  for (int iz = ghost_grid_.sz; iz < ghost_grid_.ez; ++iz)
    for (int iy = ghost_grid_.sy; iy < ghost_grid_.ey; ++iy)
      for (int ix = ghost_grid_.sx; ix < ghost_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        cnumPars_[index] = 0;
        cells_[index].next = NULL;
        last_cells_[index] = &cells_[index];
      }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Move particles from second array to first array.
 */
void PartitionData::RebuildGridMT() {
  // Note, in parallel versions the below swaps occure outside RebuildGrid()
  // swap src and dest arrays with particles std::swap(cells, cells2); swap src
  // and dest arrays with counts of particles std::swap(cnumPars, cnumPars2);

  // iterate through source cell lists
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index2 = GetLocalIndex(ix, iy, iz);
        Cell* cell2 = &cells2_[index2];
        int np2 = cnumPars2_[index2];
        // iterate through source particles
        for (int j = 0; j < np2; ++j) {
          // get destination for source particle
          int ci =
              (int)((cell2->p[j % PARTICLES_PER_CELL].x - domainMinFull.x) /
                    delta_.x);
          int cj =
              (int)((cell2->p[j % PARTICLES_PER_CELL].y - domainMinFull.y) /
                    delta_.y);
          int ck =
              (int)((cell2->p[j % PARTICLES_PER_CELL].z - domainMinFull.z) /
                    delta_.z);

          if (ci < ghost_grid_.sx)
            ci = ghost_grid_.sx;
          else if (ci > (ghost_grid_.ex - 1))
            ci = ghost_grid_.ex - 1;
          if (cj < ghost_grid_.sy)
            cj = ghost_grid_.sy;
          else if (cj > (ghost_grid_.ey - 1))
            cj = ghost_grid_.ey - 1;
          if (ck < ghost_grid_.sz)
            ck = ghost_grid_.sz;
          else if (ck > (ghost_grid_.ez - 1))
            ck = ghost_grid_.ez - 1;

          // this assumes that particles cannot travel more than one grid cell
          // per time step
          // Failed.
          CHECK(ghost_grid_.Contain(ci, cj, ck));
          int index = GetLocalIndex(ci, cj, ck);

          Cell* cell = last_cells_[index];
          int np = cnumPars_[index];

          // add another cell structure if everything full
          if ((np % PARTICLES_PER_CELL == 0) && (cnumPars_[index] != 0)) {
            cell->next = cellpool_getcell(&local_pool_);
            cell = cell->next;
            last_cells_[index] = cell;
          }
          ++cnumPars_[index];

          // copy source to destination particle

          cell->p[np % PARTICLES_PER_CELL] = cell2->p[j % PARTICLES_PER_CELL];
          cell->hv[np % PARTICLES_PER_CELL] = cell2->hv[j % PARTICLES_PER_CELL];
          cell->v[np % PARTICLES_PER_CELL] = cell2->v[j % PARTICLES_PER_CELL];
          cell->density[np % PARTICLES_PER_CELL] = 0.0;
          cell->a[np % PARTICLES_PER_CELL] = externalAcceleration;
          // move pointer to next source cell in list if end of array is reached
          if (j % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            Cell* temp = cell2;
            cell2 = cell2->next;
            // return cells to pool that are not statically allocated head of
            // lists
            if (temp != &cells2_[index2]) {
              cellpool_returncell(&local_pool_, temp);
            }
          }
        }  // for(int j = 0; j < np2; ++j)
        // return cells to pool that are not statically allocated head of lists
        if ((cell2 != NULL) && (cell2 != &cells2_[index2])) {
          cellpool_returncell(&local_pool_, cell2);
        }
      }
  for (int iz = ghost_grid_.sz; iz < ghost_grid_.ez; ++iz)
    for (int iy = ghost_grid_.sy; iy < ghost_grid_.ey; ++iy)
      for (int ix = ghost_grid_.sx; ix < ghost_grid_.ex; ++ix) {
        if (!local_grid_.Contain(ix, iy, iz)) {
          int index2 = GetLocalIndex(ix, iy, iz);
          Cell* cell2 = &cells2_[index2];
          while (cell2->next) {
            Cell* temp = cell2->next;
            cell2->next = temp->next;
            cellpool_returncell(&local_pool_, temp);
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Calculate the geometrically neighboring cells.
 */
int PartitionData::InitNeighCellList(int ci, int cj, int ck,
                                     int* neighCells) const {
  int numNeighCells = 0;

  // have the nearest particles first -> help branch prediction
  int my_index = GetLocalIndex(ci, cj, ck);
  neighCells[numNeighCells] = my_index;
  ++numNeighCells;

  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        int ii = ci + di;
        int jj = cj + dj;
        int kk = ck + dk;
        if (ii >= 0 && ii < nx_ && jj >= 0 && jj < ny_ && kk >= 0 && kk < nz_) {
          int index = GetLocalIndex(ii, jj, kk);
          if ((cnumPars_[index] != 0) &&
              (!local_grid_.Contain(ii, jj, kk) || index < my_index)) {
            neighCells[numNeighCells] = index;
            ++numNeighCells;
          }
        }
      }
  return numNeighCells;
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Calculate densities as an average of densities of neighboring cells.
 */
void PartitionData::ComputeDensitiesMT() {
  int neighCells[3 * 3 * 3];

  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        int np = cnumPars_[index];
        if (np == 0) continue;

        int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

        Cell* cell = &cells_[index];
        for (int ipar = 0; ipar < np; ++ipar) {
          for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell* neigh = &cells_[indexNeigh];
            int numNeighPars = cnumPars_[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh) {
              if (index != indexNeigh ||
                  &neigh->p[iparNeigh % PARTICLES_PER_CELL] <
                      &cell->p[ipar % PARTICLES_PER_CELL]) {
                fptype distSq =
                    (cell->p[ipar % PARTICLES_PER_CELL] -
                     neigh->p[iparNeigh % PARTICLES_PER_CELL]).GetLengthSq();
                if (distSq < hSq_) {
                  fptype t = hSq_ - distSq;
                  fptype tc = t * t * t;
                  cell->density[ipar % PARTICLES_PER_CELL] += tc;
                  neigh->density[iparNeigh % PARTICLES_PER_CELL] += tc;
                }
              }
              // move pointer to next cell in list if end of array is reached
              if (iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
                neigh = neigh->next;
              }
            }
          }
          // move pointer to next cell in list if end of array is reached
          if (ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Update densities locally.
 */
void PartitionData::ComputeDensities2MT() {
  const fptype tc = hSq_ * hSq_ * hSq_;
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          cell->density[j % PARTICLES_PER_CELL] += tc;
          cell->density[j % PARTICLES_PER_CELL] *= densityCoeff_;
          // move pointer to next cell in list if end of array is reached
          if (j % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Update accelerations by examining neighboring cells.
 */
void PartitionData::ComputeForcesMT() {
  int neighCells[3 * 3 * 3];
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        int np = cnumPars_[index];
        if (np == 0) continue;

        int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

        Cell* cell = &cells_[index];
        for (int ipar = 0; ipar < np; ++ipar) {
          for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell* neigh = &cells_[indexNeigh];
            int numNeighPars = cnumPars_[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh) {
              if (index != indexNeigh ||
                  &neigh->p[iparNeigh % PARTICLES_PER_CELL] <
                      &cell->p[ipar % PARTICLES_PER_CELL]) {
                Vec3 disp = cell->p[ipar % PARTICLES_PER_CELL] -
                            neigh->p[iparNeigh % PARTICLES_PER_CELL];
                fptype distSq = disp.GetLengthSq();
                if (distSq < hSq_) {
                  fptype dist = sqrtf(std::max(distSq, (fptype)1e-12));
                  fptype hmr = h_ - dist;

                  Vec3 acc = disp * pressureCoeff_ * (hmr * hmr / dist) *
                             (cell->density[ipar % PARTICLES_PER_CELL] +
                              neigh->density[iparNeigh % PARTICLES_PER_CELL] -
                              doubleRestDensity);
                  acc += (neigh->v[iparNeigh % PARTICLES_PER_CELL] -
                          cell->v[ipar % PARTICLES_PER_CELL]) *
                         viscosityCoeff_ * hmr;
                  acc /= cell->density[ipar % PARTICLES_PER_CELL] *
                         neigh->density[iparNeigh % PARTICLES_PER_CELL];

                  cell->a[ipar % PARTICLES_PER_CELL] += acc;

                  neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc;
                }
              }
              // move pointer to next cell in list if end of array is reached
              if (iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
                neigh = neigh->next;
              }
            }
          }
          // move pointer to next cell in list if end of array is reached
          if (ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

// ProcessCollisions() with container walls
// Under the assumptions that
// a) a particle will not penetrate a wall
// b) a particle will not migrate further than once cell
// c) the parSize is smaller than a cell
// then only the particles at the perimiters may be influenced by the walls
/**
 * Update accelerations locally.
 */
void PartitionData::ProcessCollisionsMT() {
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        if (!((ix == 0) || (iy == 0) || (iz == 0) || (ix == (nx_ - 1)) ||
              (iy == (ny_ - 1)) == (iz == (nz_ - 1))))
          continue;  // not on domain wall
        int index = GetLocalIndex(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          int ji = j % PARTICLES_PER_CELL;
          Vec3 pos = cell->p[ji] + cell->hv[ji] * timeStep;

          if (ix == 0) {
            fptype diff = parSize - (pos.x - domainMinFull.x);
            if (diff > epsilon)
              cell->a[ji].x +=
                  stiffnessCollisions * diff - damping * cell->v[ji].x;
          }
          if (ix == (nx_ - 1)) {
            fptype diff = parSize - (domainMaxFull.x - pos.x);
            if (diff > epsilon)
              cell->a[ji].x -=
                  stiffnessCollisions * diff + damping * cell->v[ji].x;
          }
          if (iy == 0) {
            fptype diff = parSize - (pos.y - domainMinFull.y);
            if (diff > epsilon)
              cell->a[ji].y +=
                  stiffnessCollisions * diff - damping * cell->v[ji].y;
          }
          if (iy == (ny_ - 1)) {
            fptype diff = parSize - (domainMaxFull.y - pos.y);
            if (diff > epsilon)
              cell->a[ji].y -=
                  stiffnessCollisions * diff + damping * cell->v[ji].y;
          }
          if (iz == 0) {
            fptype diff = parSize - (pos.z - domainMinFull.z);
            if (diff > epsilon)
              cell->a[ji].z +=
                  stiffnessCollisions * diff - damping * cell->v[ji].z;
          }
          if (iz == (nz_ - 1)) {
            fptype diff = parSize - (domainMaxFull.z - pos.z);
            if (diff > epsilon)
              cell->a[ji].z -=
                  stiffnessCollisions * diff + damping * cell->v[ji].z;
          }
          // move pointer to next cell in list if end of array is reached
          if (ji == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

/**
 * Process boundary cells locally.
 */
void PartitionData::ProcessCollisions2MT() {
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          int ji = j % PARTICLES_PER_CELL;
          Vec3 pos = cell->p[ji];

          if (ix == 0) {
            fptype diff = pos.x - domainMinFull.x;
            if (diff < Zero) {
              cell->p[ji].x = domainMinFull.x - diff;
              cell->v[ji].x = -cell->v[ji].x;
              cell->hv[ji].x = -cell->hv[ji].x;
            }
          }
          if (ix == (nx_ - 1)) {
            fptype diff = domainMaxFull.x - pos.x;
            if (diff < Zero) {
              cell->p[ji].x = domainMaxFull.x + diff;
              cell->v[ji].x = -cell->v[ji].x;
              cell->hv[ji].x = -cell->hv[ji].x;
            }
          }
          if (iy == 0) {
            fptype diff = pos.y - domainMinFull.y;
            if (diff < Zero) {
              cell->p[ji].y = domainMinFull.y - diff;
              cell->v[ji].y = -cell->v[ji].y;
              cell->hv[ji].y = -cell->hv[ji].y;
            }
          }
          if (iy == (ny_ - 1)) {
            fptype diff = domainMaxFull.y - pos.y;
            if (diff < Zero) {
              cell->p[ji].y = domainMaxFull.y + diff;
              cell->v[ji].y = -cell->v[ji].y;
              cell->hv[ji].y = -cell->hv[ji].y;
            }
          }
          if (iz == 0) {
            fptype diff = pos.z - domainMinFull.z;
            if (diff < Zero) {
              cell->p[ji].z = domainMinFull.z - diff;
              cell->v[ji].z = -cell->v[ji].z;
              cell->hv[ji].z = -cell->hv[ji].z;
            }
          }
          if (iz == (nz_ - 1)) {
            fptype diff = domainMaxFull.z - pos.z;
            if (diff < Zero) {
              cell->p[ji].z = domainMaxFull.z + diff;
              cell->v[ji].z = -cell->v[ji].z;
              cell->hv[ji].z = -cell->hv[ji].z;
            }
          }
          // move pointer to next cell in list if end of array is reached
          if (ji == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Update positions based on accelerations locally.
 */
void PartitionData::AdvanceParticlesMT() {
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          Vec3 v_half = cell->hv[j % PARTICLES_PER_CELL] +
                        cell->a[j % PARTICLES_PER_CELL] * timeStep;
          // N.B. The integration of the position can place the particle
          // outside the domain. Although we could place a test in this loop
          // we would be unnecessarily testing particles on interior cells.
          // Therefore, to reduce the amount of computations we make a later
          // pass on the perimiter cells to account for particle migration
          // beyond domain
          cell->p[j % PARTICLES_PER_CELL] += v_half * timeStep;
          cell->v[j % PARTICLES_PER_CELL] =
              cell->hv[j % PARTICLES_PER_CELL] + v_half;
          cell->v[j % PARTICLES_PER_CELL] *= 0.5;
          cell->hv[j % PARTICLES_PER_CELL] = v_half;

          // move pointer to next cell in list if end of array is reached
          if (j % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
}

float PartitionData::ComputeEnergy() {
  energy_sum_ = 0;
  for (int iz = local_grid_.sz; iz < local_grid_.ez; ++iz)
    for (int iy = local_grid_.sy; iy < local_grid_.ey; ++iy)
      for (int ix = local_grid_.sx; ix < local_grid_.ex; ++ix) {
        int index = GetLocalIndex(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          energy_sum_ += cell->v[j % PARTICLES_PER_CELL].GetLengthSq();

          // move pointer to next cell in list if end of array is reached
          if (j % PARTICLES_PER_CELL == PARTICLES_PER_CELL - 1) {
            cell = cell->next;
          }
        }
      }
  return energy_sum_;
}

/*
 * Serialization/deserialization related.
 */

//! Serializes the data of a cell. INDEX is the local index.
template <typename Archive>
void PartitionData::SerializeFullCell(Archive& archive, int index) const {
  const Cell* cell = &cells_[index];
  const int np = cnumPars_[index];
  archive(np);
  for (int i = 0; i < np; ++i) {
    const int incell_index = i % PARTICLES_PER_CELL;
    archive(cell->p[incell_index], cell->hv[incell_index],
            cell->v[incell_index], cell->a[incell_index],
            cell->density[incell_index]);
    if (incell_index == PARTICLES_PER_CELL - 1) {
      cell = cell->next;
    }
  }
}

void PartitionData::SendExchangeGhostCells(
    std::vector<int>* neighbors, std::vector<std::vector<char>>* send_buffer) {
  neighbors->clear();
  neighbors->reserve(exchange_metadata_.size());
  send_buffer->clear();
  send_buffer->reserve(exchange_metadata_.size());
  for (const auto& pair : exchange_metadata_) {
    neighbors->push_back(pair.first);
    auto const& metadata = pair.second;
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive oarchive(ss);
      oarchive(metadata.send_to_owner_local_indices.size());
      for (const auto& in_pair : metadata.send_to_owner_local_indices) {
        oarchive(in_pair.first);
        SerializeFullCell(oarchive, in_pair.second);
      }
      oarchive(metadata.send_to_ghost_local_indices.size());
      for (const auto& in_pair : metadata.send_to_ghost_local_indices) {
        oarchive(in_pair.first);
        SerializeFullCell(oarchive, in_pair.second);
      }
    }  // Serialization done.
    send_buffer->emplace_back(ss.tellp());
    memcpy(&send_buffer->back()[0], ss.str().c_str(), ss.tellp());
  }
}

//! Deserialzes the data of a cell. INDEX is the local index.
template <typename Archive>
void PartitionData::DeserializeFullCell(Archive& archive, int index) {
  int added_particle;
  archive(added_particle);
  Cell* cell = last_cells_[index];
  for (int i = cnumPars_[index]; i < cnumPars_[index] + added_particle; ++i) {
    const int incell_index = i % PARTICLES_PER_CELL;
    if (incell_index == 0 && i != 0) {
      cell->next = cellpool_getcell(&local_pool_);
      cell = cell->next;
    }
    archive(cell->p[incell_index], cell->hv[incell_index],
            cell->v[incell_index], cell->a[incell_index],
            cell->density[incell_index]);
  }
  last_cells_[index] = cell;
  cnumPars_[index] += added_particle;
}

void PartitionData::RecvExchangeGhostCells(
    const std::vector<std::vector<char>>& recv_buffer) {
  for (auto& element :
       exchange_metadata_[partition_rank_].send_to_ghost_local_indices) {
    int local_index = element.second;
    Cell* cell = &cells_[local_index];
    while (cell->next) {
      Cell* temp = cell->next;
      cell->next = temp->next;
      cellpool_returncell(&local_pool_, temp);
    }
    cnumPars_[local_index] = 0;
    last_cells_[local_index] = cell;
  }

  const size_t max_size = recv_buffer.size();
  std::vector<std::unique_ptr<std::stringstream>> ss_vector(max_size);
  std::vector<std::unique_ptr<cereal::BinaryInputArchive>> iarchive_vector(
      max_size);
  for (size_t index = 0; index < max_size; ++index) {
    ss_vector[index].reset(new std::stringstream(
        std::string(recv_buffer[index].data(), recv_buffer[index].size())));
    iarchive_vector[index].reset(
        new cereal::BinaryInputArchive(*ss_vector[index]));
  }
  // Caution: the following code makes sure that the ghost cells are merged
  // exactly the same way.
  for (size_t index = 0; index < max_size; ++index) {
    auto& iarchive = *iarchive_vector[index];
    size_t owner_cells = 0;
    iarchive(owner_cells);
    while (owner_cells-- > 0) {
      int global_index;
      iarchive(global_index);
      int index = GetLocalIndexFromGlobalIndex(global_index);
      DeserializeFullCell(iarchive, index);
    }
  }
  for (size_t index = 0; index < max_size; ++index) {
    auto& iarchive = *iarchive_vector[index];
    size_t ghost_cells = 0;
    iarchive(ghost_cells);
    while (ghost_cells-- > 0) {
      int global_index;
      iarchive(global_index);
      int index = GetLocalIndexFromGlobalIndex(global_index);
      DeserializeFullCell(iarchive, index);
    }
  }
}

template <typename Archive>
void PartitionData::SerializeDensity(Archive& archive, int index) const {
  const Cell* cell = &cells_[index];
  int np = cnumPars_[index];
  archive(np);
  for (int j = 0; j < np; ++j) {
    if ((j % PARTICLES_PER_CELL == 0) && (j != 0)) {
      cell = cell->next;
    }
    // archive(cell->density[j % PARTICLES_PER_CELL], cell->v[j %
    // PARTICLES_PER_CELL]);
    archive(cell->density[j % PARTICLES_PER_CELL]);
  }
}

void PartitionData::SendDensityGhost(
    std::vector<int>* neighbors, std::vector<std::vector<char>>* send_buffer) {
  neighbors->clear();
  neighbors->reserve(exchange_metadata_.size());
  send_buffer->clear();
  send_buffer->reserve(exchange_metadata_.size());
  for (auto element : exchange_metadata_) {
    neighbors->push_back(element.first);
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive oarchive(ss);
      oarchive(element.second.send_to_owner_local_indices.size());
      for (auto& metadata : element.second.send_to_owner_local_indices) {
        oarchive(metadata.first);
        SerializeDensity(oarchive, metadata.second);
      }
    }  // Serialization done.
    send_buffer->emplace_back(ss.tellp());
    memcpy(&send_buffer->back()[0], ss.str().c_str(), ss.tellp());
  }
}

template <typename Archive>
void PartitionData::DeserializeDensity(Archive& archive, int index) {
  int num_particle;
  archive(num_particle);
  Cell* cell = &cells_[index];
  CHECK_EQ(num_particle, cnumPars_[index]);
  for (int count = 0; count < num_particle; ++count) {
    if ((count % PARTICLES_PER_CELL == 0) && (count != 0)) {
      cell = cell->next;
    }
    // Vec3 v;
    // archive(cell->density[count % PARTICLES_PER_CELL], v);
    archive(cell->density[count % PARTICLES_PER_CELL]);
    // CHECK(v == cell->v[count % PARTICLES_PER_CELL])
    //     << " " << v.x << " " << v.y << " " << v.z
    //     << " " << cell->v[count % PARTICLES_PER_CELL].x
    //     << " " << cell->v[count % PARTICLES_PER_CELL].y
    //     << " " << cell->v[count % PARTICLES_PER_CELL].z;
  }
  CHECK_EQ(cell, last_cells_[index]);
}

void PartitionData::RecvDensityGhost(
    const std::vector<std::vector<char>>& recv_buffer) {
  for (auto& recv_data : recv_buffer) {
    std::stringstream ss(std::string(recv_data.data(), recv_data.size()));
    cereal::BinaryInputArchive iarchive(ss);
    size_t density_cells = 0;
    iarchive(density_cells);
    while (density_cells-- > 0) {
      int global_index;
      iarchive(global_index);
      int index = GetLocalIndexFromGlobalIndex(global_index);
      DeserializeDensity(iarchive, index);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace canary {

class ParsecApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    CHECK_LE(FLAGS_app_depth, FLAGS_app_fold_y);

    const auto NUM_PARTITION =
        FLAGS_app_partition_x * FLAGS_app_partition_y * FLAGS_app_partition_z;
    auto d_partition = DeclareVariable<::PartitionData>(NUM_PARTITION);
    auto d_global = DeclareVariable<::GlobalState>(1);

    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      partition->InitSim(FLAGS_app_filename.c_str(), FLAGS_app_partition_x,
                         FLAGS_app_partition_y, FLAGS_app_partition_z,
                         task_context->GetPartitionId());
    });

    WriteAccess(d_global);
    Transform([=](CanaryTaskContext* task_context) {
      auto global = task_context->WriteVariable(d_global);
      CHECK_NOTNULL(global);
    });

    Loop(FLAGS_app_frames);

    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      // Swap src and dest arrays with particles.
      partition->DataSwap();
      // Clear the first array.
      partition->ClearParticlesMT();
      // Move particles from second array to first array.
      partition->RebuildGridMT();
    });

    WriteAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      std::vector<int> neighbors;
      std::vector<std::vector<char>> send_buffer;
      partition->SendExchangeGhostCells(&neighbors, &send_buffer);
      for (size_t index = 0; index < neighbors.size(); ++index) {
        task_context->OrderedScatter(neighbors[index], send_buffer[index]);
      }
    });

    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto partition = task_context->WriteVariable(d_partition);
      const int num_neighbor = partition->GetNumNeighbors();
      EXPECT_GATHER_SIZE(num_neighbor);
      auto recv_buffer = task_context->OrderedGather<std::vector<char>>();
      std::vector<std::vector<char>> sort_buffer;
      sort_buffer.reserve(recv_buffer.size());
      for (auto& pair : recv_buffer) {
        sort_buffer.emplace_back(std::move(pair.second));
      }
      partition->RecvExchangeGhostCells(sort_buffer);
      return 0;
    });

    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      // Update densities by examining neighboring cells.
      partition->ComputeDensitiesMT();
      // Update densities locally.
      partition->ComputeDensities2MT();
    });

    WriteAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      std::vector<int> neighbors;
      std::vector<std::vector<char>> send_buffer;
      partition->SendDensityGhost(&neighbors, &send_buffer);
      for (size_t index = 0; index < neighbors.size(); ++index) {
        task_context->OrderedScatter(neighbors[index], send_buffer[index]);
      }
    });

    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto partition = task_context->WriteVariable(d_partition);
      const int num_neighbor = partition->GetNumNeighbors();
      EXPECT_GATHER_SIZE(num_neighbor);
      auto recv_buffer = task_context->OrderedGather<std::vector<char>>();
      std::vector<std::vector<char>> sort_buffer;
      sort_buffer.reserve(recv_buffer.size());
      for (auto& pair : recv_buffer) {
        sort_buffer.emplace_back(std::move(pair.second));
      }
      partition->RecvExchangeGhostCells(sort_buffer);
      return 0;
    });


    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      // Update accelerations by examining neighboring cells.
      partition->ComputeForcesMT();
      // Update accelerations locally.
      partition->ProcessCollisionsMT();
      // Update positions based on accelerations locally.
      partition->AdvanceParticlesMT();
      // Process near-wall cells locally.
      partition->ProcessCollisions2MT();
    });

    EndLoop();

    ReadAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      const auto& partition = task_context->WriteVariable(d_partition);
      task_context->Scatter(0, partition->ComputeEnergy());
    });

    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      float sum = 0;
      sum = task_context->Reduce(sum, std::plus<float>());
      printf("Total energy: %f\n", sum);
    });
  }

  // Loads parameter.
  void LoadParameter(const std::string& parameter) override {
    std::stringstream ss;
    ss << parameter;
    {
      cereal::XMLInputArchive archive(ss);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::ParsecApplication);
