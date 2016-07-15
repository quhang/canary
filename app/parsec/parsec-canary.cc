// Code written by Richard O. Lee and Christian Bienia
// Modified by Christian Fensch

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <assert.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <pthread.h>
#include <utility>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include "canary/canary.h"

#include "fluid.h"
#include "cellpool.h"
#include "../grid_helper.h"

static int FLAG_app_partition_x = 1;        // Partitioning in x.
static int FLAG_app_partition_y = 1;        // Partitioning in y.
static int FLAG_app_partition_z = 1;        // Partitioning in z.
static int FLAG_app_fold_x = 1;             // Fold in x.
static int FLAG_app_fold_y = 1;             // Fold in y.
static int FLAG_app_fold_z = 1;             // Fold in z.
static int FLAG_app_fold_depth_y = 1;       // Fold depth in y.
static int FLAG_app_frames = 10;            // Frames.
static std::string FLAG_app_filename =
    "../app/parsec/in_15K.fluid";  // Input file name.

struct GlobalState {
  template <typename Archive>
  void serialize(Archive&) {}
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
  //! Memory pool.
  cellpool local_pool_;
  //! Energy sum.
  float energy_sum_ = 0;
  //! Simulation parameters.
  fptype restParticlesPerMeter_ = 0, h_ = 0, hSq_ = 0;
  fptype densityCoeff_ = 0, pressureCoeff_ = 0, viscosityCoeff_ = 0;
  //! Particles.
  std::vector<Cell> cells_, cells2_;
  std::vector<int> cnumPars_, cnumPars2_;
  //! The pointer corresponds to CELLS and CNUMPARS.
  std::vector<Cell*> last_cells_;
  struct Metadata {
    // Global index/local index.
    std::list<std::pair<int, int>> local_to_ghost_indices;
    std::list<std::pair<int, int>> ghost_to_ghost_indices;
    template <typename Archive>
    void serialize(Archive& archive) {  // NOLINT
      archive(local_to_ghost_indices);
      archive(ghost_to_ghost_indices);
    }
  };
  //! Tracks what cell data need to be sent to which partition.
  std::map<int, Metadata> exchange_metadata_;
  //! Grid.
  helper::Grid local_grid_, ghost_grid_;

 public:
  template <typename Archive>
  void save(Archive& archive) const {  // NOLINT
    archive(energy_sum_);
    archive(restParticlesPerMeter_, h_, hSq_);
    archive(densityCoeff_, pressureCoeff_, viscosityCoeff_);
    archive(cells_.size());
    for (int index = 0; index < (int)cells_.size(); ++index) {
      SerializeFullCell(archive, index);
    }
    archive(exchange_metadata_);
    archive(local_grid_, ghost_grid_);
  }

  template <typename Archive>
  void load(Archive& archive) {  // NOLINT
    archive(energy_sum_);
    archive(restParticlesPerMeter_, h_, hSq_);
    archive(densityCoeff_, pressureCoeff_, viscosityCoeff_);
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
    archive(local_grid_, ghost_grid_);
  }
  int GetNumNeighbors() { return exchange_metadata_.size(); }

 private:
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
               int subgrid_rank);
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
                              std::vector<struct evbuffer*>* send_buffer);
  void RecvExchangeGhostCells(std::vector<struct evbuffer*>* recv_buffer);
  void SendDensityGhost(std::vector<int>* neighbors,
                        std::vector<struct evbuffer*>* send_buffer);
  void RecvDensityGhost(std::vector<struct evbuffer*>* recv_buffer);
};

PartitionData::PartitionData() { cellpool_init(&local_pool_, 1000); }

PartitionData::~PartitionData() {
  cellpool_destroy(&local_pool_);
}

void PartitionData::InitSim(char const* filename, int split_x, int split_y,
                            int split_z, int subgrid_rank) {
  // Loads input particles
  std::ifstream file(filename, std::ios::binary);
  CHECK(file) << "Error opening file: " << filename;
  // Reads numParticles and resetParticlesPerMeter.
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
  // Sets up grid.
  const helper::Point domain_min{domainMinPart.x, domainMinPart.y,
                                 domainMinPart.z};
  const helper::Point domain_size{
      FLAG_app_fold_x * (domainMaxPart.x - domainMinPart.x),
      FLAG_app_fold_y * (domainMaxPart.y - domainMinPart.y),
      FLAG_app_fold_z * (domainMaxPart.z - domainMinPart.z)};
  const helper::Point domain_max{domain_size.x + domainMinPart.x,
                                 domain_size.y + domainMinPart.y,
                                 domain_size.z + domainMinPart.z};
  local_grid_.Initialize(
      {domain_min, domain_max},                   // Domain size.
      helper::cast_point<int>(domain_size / h_),  // Grid size.
      {split_x, split_y, split_z}, subgrid_rank);
  ghost_grid_.Initialize(
      {domain_min, domain_max},                   // Domain size.
      helper::cast_point<int>(domain_size / h_),  // Grid size.
      {split_x, split_y, split_z}, subgrid_rank, 1);
  // Sets up neighboring data exchanging metadata.
  {
    auto& metadata = exchange_metadata_[subgrid_rank];
    for (int iz = ghost_grid_.get_sz(); iz < ghost_grid_.get_ez(); ++iz)
      for (int iy = ghost_grid_.get_sy(); iy < ghost_grid_.get_ey(); ++iy)
        for (int ix = ghost_grid_.get_sx(); ix < ghost_grid_.get_ex(); ++ix) {
          if (!local_grid_.Contain(ix, iy, iz)) {
            const int global_index = ghost_grid_.GetGlobalCellRank(ix, iy, iz);
            const int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
            metadata.ghost_to_ghost_indices.emplace_back(global_index, index);
          }
        }
  }
  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        if (di == 0 && dj == 0 && dk == 0) {
          continue;
        }
        helper::Grid neighbor_ghost_grid;
        if (!ghost_grid_.GetNeighborSubgrid(di, dj, dk, 1,
                                            &neighbor_ghost_grid)) {
          continue;
        }
        auto& metadata =
            exchange_metadata_[neighbor_ghost_grid.GetSubgridRank()];
        for (int iz = ghost_grid_.get_sz(); iz < ghost_grid_.get_ez(); ++iz)
          for (int iy = ghost_grid_.get_sy(); iy < ghost_grid_.get_ey(); ++iy)
            for (int ix = ghost_grid_.get_sx(); ix < ghost_grid_.get_ex();
                 ++ix) {
              const int global_index =
                  ghost_grid_.GetGlobalCellRank(ix, iy, iz);
              const int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
              if (neighbor_ghost_grid.Contain(ix, iy, iz)) {
                if (local_grid_.Contain(ix, iy, iz)) {
                  metadata.local_to_ghost_indices.emplace_back(global_index,
                                                               index);
                } else {
                  metadata.ghost_to_ghost_indices.emplace_back(global_index,
                                                               index);
                }
              }
            }
      }
  // Makes sure Cell structure is multiple of estiamted cache line size.
  static_assert(sizeof(Cell) % CACHELINE_SIZE == 0, "wrong padding");
  // Makes sure helper Cell structure is in sync with real Cell structure.
  static_assert(
      offsetof(struct Cell_aux, padding) == offsetof(struct Cell, padding),
      "wrong padding");
  // Initializes all cells.
  cells_.resize(ghost_grid_.get_count());
  cells2_.resize(ghost_grid_.get_count());
  cnumPars_.resize(ghost_grid_.get_count(), 0);
  cnumPars2_.resize(ghost_grid_.get_count(), 0);
  last_cells_.resize(ghost_grid_.get_count(), nullptr);
  for (int iz = ghost_grid_.get_sz(); iz < ghost_grid_.get_ez(); ++iz)
    for (int iy = ghost_grid_.get_sy(); iy < ghost_grid_.get_ey(); ++iy)
      for (int ix = ghost_grid_.get_sx(); ix < ghost_grid_.get_ex(); ++ix) {
        const int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
        cnumPars_[index] = 0;
        cells_[index].next = NULL;
        last_cells_[index] = &cells_[index];
      }
  // Always uses single precision float variables b/c file format uses single
  // precision float.
  float unfold_px, unfold_py, unfold_pz, hvx, hvy, hvz, vx, vy, vz;
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

    for (int fold_x = 0; fold_x < FLAG_app_fold_x; ++fold_x)
      for (int fold_y = 0; fold_y < FLAG_app_fold_depth_y; ++fold_y)
        for (int fold_z = 0; fold_z < FLAG_app_fold_z; ++fold_z) {
          float px = unfold_px + fold_x * (domainMaxPart.x - domainMinPart.x);
          float py = unfold_py + fold_y * (domainMaxPart.y - domainMinPart.y);
          float pz = unfold_pz + fold_z * (domainMaxPart.z - domainMinPart.z);
          if (!local_grid_.ContainDomain(px, py, pz)) {
            continue;
          }
          const int index = ghost_grid_.GetLocalCellRankDomain(px, py, pz);
          Cell* cell = last_cells_[index];
          const int incell_index = cnumPars_[index] % PARTICLES_PER_CELL;
          if (incell_index == 0 && cnumPars_[index] != 0) {
            cell->next = cellpool_getcell(&local_pool_);
            cell = cell->next;
            last_cells_[index] = cell;
          }
          cell->p[incell_index].x = px;
          cell->p[incell_index].y = py;
          cell->p[incell_index].z = pz;
          cell->hv[incell_index].x = hvx;
          cell->hv[incell_index].y = hvy;
          cell->hv[incell_index].z = hvz;
          cell->v[incell_index].x = vx;
          cell->v[incell_index].y = vy;
          cell->v[incell_index].z = vz;
          ++cnumPars_[index];
        }
  }
}

/*
 * Execution logic.
 */

//! Clears the first array. It assumes cells are already returned to the memory
// pool.
void PartitionData::ClearParticlesMT() {
  for (int iz = ghost_grid_.get_sz(); iz < ghost_grid_.get_ez(); ++iz)
    for (int iy = ghost_grid_.get_sy(); iy < ghost_grid_.get_ey(); ++iy)
      for (int ix = ghost_grid_.get_sx(); ix < ghost_grid_.get_ex(); ++ix) {
        const int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
        cnumPars_[index] = 0;
        cells_[index].next = NULL;
        last_cells_[index] = &cells_[index];
      }
}

//! Moves particles from second array to first array.
void PartitionData::RebuildGridMT() {
  // Iterates through source cell lists.
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        const int index2 = ghost_grid_.GetLocalCellRank(ix, iy, iz);
        Cell* cell2 = &cells2_[index2];
        // Iterates through source particles.
        for (int j = 0; j < cnumPars2_[index2]; ++j) {
          const int incell_index2 = j % PARTICLES_PER_CELL;
          // This assumes that particles cannot travel more than one grid cell
          // per time step.
          int index = ghost_grid_.GetLocalCellRankDomain(
              cell2->p[incell_index2].x, cell2->p[incell_index2].y,
              cell2->p[incell_index2].z);
          Cell* cell = last_cells_[index];
          const int incell_index = cnumPars_[index] % PARTICLES_PER_CELL;
          // add another cell structure if everything full
          if ((incell_index == 0) && (cnumPars_[index] != 0)) {
            cell->next = cellpool_getcell(&local_pool_);
            cell = cell->next;
            last_cells_[index] = cell;
          }
          ++cnumPars_[index];
          cell->p[incell_index] = cell2->p[incell_index2];
          cell->hv[incell_index] = cell2->hv[incell_index2];
          cell->v[incell_index] = cell2->v[incell_index2];
          cell->density[incell_index] = 0.0;
          cell->a[incell_index] = externalAcceleration;
          // move pointer to next source cell in list if end of array is reached
          if (incell_index2 == PARTICLES_PER_CELL - 1) {
            Cell* temp = cell2;
            cell2 = cell2->next;
            // Returns cells to pool that are not statically allocated head of
            // lists
            if (temp != &cells2_[index2]) {
              cellpool_returncell(&local_pool_, temp);
            }
          }
        }
        // Returns cells to pool that are not statically allocated head of lists
        if ((cell2 != NULL) && (cell2 != &cells2_[index2])) {
          cellpool_returncell(&local_pool_, cell2);
        }
      }
  //! Cleans cells in the ghost boundary.
  for (int iz = ghost_grid_.get_sz(); iz < ghost_grid_.get_ez(); ++iz)
    for (int iy = ghost_grid_.get_sy(); iy < ghost_grid_.get_ey(); ++iy)
      for (int ix = ghost_grid_.get_sx(); ix < ghost_grid_.get_ex(); ++ix) {
        if (!local_grid_.Contain(ix, iy, iz)) {
          int index2 = ghost_grid_.GetLocalCellRank(ix, iy, iz);
          Cell* cell2 = &cells2_[index2];
          while (cell2->next) {
            Cell* temp = cell2->next;
            cell2->next = temp->next;
            cellpool_returncell(&local_pool_, temp);
          }
        }
      }
}

//! Calculate the geometrically neighboring cells.
int PartitionData::InitNeighCellList(int ci, int cj, int ck,
                                     int* neighCells) const {
  int numNeighCells = 0;
  // have the nearest particles first -> help branch prediction
  int my_index = ghost_grid_.GetLocalCellRank(ci, cj, ck);
  neighCells[numNeighCells] = my_index;
  ++numNeighCells;
  for (int di = -1; di <= 1; ++di)
    for (int dj = -1; dj <= 1; ++dj)
      for (int dk = -1; dk <= 1; ++dk) {
        int ii = ci + di;
        int jj = cj + dj;
        int kk = ck + dk;
        if (ghost_grid_.Contain(ii, jj, kk)) {
          int index = ghost_grid_.GetLocalCellRank(ii, jj, kk);
          if ((cnumPars_[index] != 0) &&
              (!local_grid_.Contain(ii, jj, kk) || index < my_index)) {
            neighCells[numNeighCells] = index;
            ++numNeighCells;
          }
        }
      }
  return numNeighCells;
}

//! Calculate densities as an average of densities of neighboring cells.
void PartitionData::ComputeDensitiesMT() {
  int neighCells[3 * 3 * 3];
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
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

//! Update densities locally.
void PartitionData::ComputeDensities2MT() {
  const fptype tc = hSq_ * hSq_ * hSq_;
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
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

//! Update accelerations by examining neighboring cells.
void PartitionData::ComputeForcesMT() {
  int neighCells[3 * 3 * 3];
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
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
  helper::Point domain_min, domain_max;
  std::tie(domain_min, domain_max) = ghost_grid_.get_domain();
  helper::IntPoint grid_size = ghost_grid_.get_grid_size();
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        if (!((ix == 0) || (iy == 0) || (iz == 0) ||
              (ix == (grid_size.x - 1)) ||
              (iy == (grid_size.y - 1)) == (iz == (grid_size.z - 1))))
          continue;  // not on domain wall
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          int ji = j % PARTICLES_PER_CELL;
          Vec3 pos = cell->p[ji] + cell->hv[ji] * timeStep;

          if (ix == 0) {
            fptype diff = parSize - (pos.x - domain_min.x);
            if (diff > epsilon)
              cell->a[ji].x +=
                  stiffnessCollisions * diff - damping * cell->v[ji].x;
          }
          if (ix == (grid_size.x - 1)) {
            fptype diff = parSize - (domain_max.x - pos.x);
            if (diff > epsilon)
              cell->a[ji].x -=
                  stiffnessCollisions * diff + damping * cell->v[ji].x;
          }
          if (iy == 0) {
            fptype diff = parSize - (pos.y - domain_min.y);
            if (diff > epsilon)
              cell->a[ji].y +=
                  stiffnessCollisions * diff - damping * cell->v[ji].y;
          }
          if (iy == (grid_size.y - 1)) {
            fptype diff = parSize - (domain_max.y - pos.y);
            if (diff > epsilon)
              cell->a[ji].y -=
                  stiffnessCollisions * diff + damping * cell->v[ji].y;
          }
          if (iz == 0) {
            fptype diff = parSize - (pos.z - domain_min.z);
            if (diff > epsilon)
              cell->a[ji].z +=
                  stiffnessCollisions * diff - damping * cell->v[ji].z;
          }
          if (iz == (grid_size.z - 1)) {
            fptype diff = parSize - (domain_max.z - pos.z);
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
  helper::Point domain_min, domain_max;
  std::tie(domain_min, domain_max) = ghost_grid_.get_domain();
  helper::IntPoint grid_size = ghost_grid_.get_grid_size();
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
        Cell* cell = &cells_[index];
        int np = cnumPars_[index];
        for (int j = 0; j < np; ++j) {
          int ji = j % PARTICLES_PER_CELL;
          Vec3 pos = cell->p[ji];

          if (ix == 0) {
            fptype diff = pos.x - domain_min.x;
            if (diff < Zero) {
              cell->p[ji].x = domain_min.x - diff;
              cell->v[ji].x = -cell->v[ji].x;
              cell->hv[ji].x = -cell->hv[ji].x;
            }
          }
          if (ix == (grid_size.x - 1)) {
            fptype diff = domain_max.x - pos.x;
            if (diff < Zero) {
              cell->p[ji].x = domain_max.x + diff;
              cell->v[ji].x = -cell->v[ji].x;
              cell->hv[ji].x = -cell->hv[ji].x;
            }
          }
          if (iy == 0) {
            fptype diff = pos.y - domain_min.y;
            if (diff < Zero) {
              cell->p[ji].y = domain_min.y - diff;
              cell->v[ji].y = -cell->v[ji].y;
              cell->hv[ji].y = -cell->hv[ji].y;
            }
          }
          if (iy == (grid_size.y - 1)) {
            fptype diff = domain_max.y - pos.y;
            if (diff < Zero) {
              cell->p[ji].y = domain_max.y + diff;
              cell->v[ji].y = -cell->v[ji].y;
              cell->hv[ji].y = -cell->hv[ji].y;
            }
          }
          if (iz == 0) {
            fptype diff = pos.z - domain_min.z;
            if (diff < Zero) {
              cell->p[ji].z = domain_min.z - diff;
              cell->v[ji].z = -cell->v[ji].z;
              cell->hv[ji].z = -cell->hv[ji].z;
            }
          }
          if (iz == (grid_size.z - 1)) {
            fptype diff = domain_max.z - pos.z;
            if (diff < Zero) {
              cell->p[ji].z = domain_max.z + diff;
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
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
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
  for (int iz = local_grid_.get_sz(); iz < local_grid_.get_ez(); ++iz)
    for (int iy = local_grid_.get_sy(); iy < local_grid_.get_ey(); ++iy)
      for (int ix = local_grid_.get_sx(); ix < local_grid_.get_ex(); ++ix) {
        int index = ghost_grid_.GetLocalCellRank(ix, iy, iz);
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
    std::vector<int>* neighbors, std::vector<struct evbuffer*>* send_buffer) {
  neighbors->clear();
  neighbors->reserve(exchange_metadata_.size());
  send_buffer->clear();
  send_buffer->reserve(exchange_metadata_.size());
  for (const auto& pair : exchange_metadata_) {
    neighbors->push_back(pair.first);
    auto const& metadata = pair.second;
    struct evbuffer* buffer = evbuffer_new();
    send_buffer->push_back(buffer);
    {
      ::canary::CanaryOutputArchive oarchive(buffer);
      oarchive(metadata.local_to_ghost_indices.size());
      for (const auto& in_pair : metadata.local_to_ghost_indices) {
        oarchive(in_pair.first);
        SerializeFullCell(oarchive, in_pair.second);
      }
      oarchive(metadata.ghost_to_ghost_indices.size());
      for (const auto& in_pair : metadata.ghost_to_ghost_indices) {
        oarchive(in_pair.first);
        SerializeFullCell(oarchive, in_pair.second);
      }
    }  // Serialization done.
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
    std::vector<struct evbuffer*>* recv_buffer) {
  for (
      auto& element :
      exchange_metadata_[ghost_grid_.GetSubgridRank()].ghost_to_ghost_indices) {
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

  // Caution: the following code makes sure that the ghost cells are merged
  // exactly the same way.
  for (int i = 0; i < 2; ++i)
    for (auto buffer : *recv_buffer) {
      ::canary::CanaryInputArchive iarchive(buffer);
      size_t owner_cells = 0;
      iarchive(owner_cells);
      while (owner_cells-- > 0) {
        int global_index;
        iarchive(global_index);
        int local_index = ghost_grid_.GlobalCellRankToLocal(global_index);
        DeserializeFullCell(iarchive, local_index);
      }
    }
  for (auto buffer : *recv_buffer) {
    CHECK_EQ(evbuffer_get_length(buffer), 0u);
    evbuffer_free(buffer);
  }
}

template <typename Archive>
void PartitionData::SerializeDensity(Archive& archive, int index) const {
  const int np = cnumPars_[index];
  archive(np);
  const Cell* cell = &cells_[index];
  for (int j = 0; j < np; ++j) {
    const int incell_index = j % PARTICLES_PER_CELL;
    archive(cell->density[incell_index]);
    if (incell_index == PARTICLES_PER_CELL - 1) {
      cell = cell->next;
    }
  }
}

void PartitionData::SendDensityGhost(
    std::vector<int>* neighbors, std::vector<struct evbuffer*>* send_buffer) {
  neighbors->clear();
  neighbors->reserve(exchange_metadata_.size());
  send_buffer->clear();
  send_buffer->reserve(exchange_metadata_.size());
  for (auto pair : exchange_metadata_) {
    neighbors->push_back(pair.first);
    auto const& metadata = pair.second;
    struct evbuffer* buffer = evbuffer_new();
    send_buffer->push_back(buffer);
    {
      ::canary::CanaryOutputArchive oarchive(buffer);
      oarchive(metadata.local_to_ghost_indices.size());
      for (auto& inpair: metadata.local_to_ghost_indices) {
        oarchive(inpair.first);
        SerializeDensity(oarchive, inpair.second);
      }
    }  // Serialization done.
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
    archive(cell->density[count % PARTICLES_PER_CELL]);
  }
  CHECK_EQ(cell, last_cells_[index]);
}

void PartitionData::RecvDensityGhost(
    std::vector<struct evbuffer*>* recv_buffer) {
  // exactly the same way.
  for (auto buffer : *recv_buffer) {
    ::canary::CanaryInputArchive iarchive(buffer);
    size_t density_cells = 0;
    iarchive(density_cells);
    while (density_cells-- > 0) {
      int global_index;
      iarchive(global_index);
      int index = ghost_grid_.GlobalCellRankToLocal(global_index);
      DeserializeDensity(iarchive, index);
    }
  }
  for (auto buffer : *recv_buffer) {
    CHECK_EQ(evbuffer_get_length(buffer), 0u);
    evbuffer_free(buffer);
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace canary {

class ParsecApplication : public CanaryApplication {
 public:
  // The program.
  void Program() override {
    CHECK_LE(FLAG_app_fold_depth_y, FLAG_app_fold_y);

    const auto NUM_PARTITION =
        FLAG_app_partition_x * FLAG_app_partition_y * FLAG_app_partition_z;
    auto d_partition = DeclareVariable<::PartitionData>(NUM_PARTITION);
    auto d_global = DeclareVariable<::GlobalState>(1);

    WriteAccess(d_partition);
    Transform([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      partition->InitSim(FLAG_app_filename.c_str(), FLAG_app_partition_x,
                         FLAG_app_partition_y, FLAG_app_partition_z,
                         task_context->GetPartitionId());
    });

    WriteAccess(d_global);
    Transform([=](CanaryTaskContext* task_context) {
      auto global = task_context->WriteVariable(d_global);
      CHECK_NOTNULL(global);
    });

    Loop(FLAG_app_frames);

    TrackNeeded();
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
      std::vector<struct evbuffer*> send_buffer;
      partition->SendExchangeGhostCells(&neighbors, &send_buffer);
      for (size_t index = 0; index < neighbors.size(); ++index) {
        task_context->OrderedScatter(neighbors[index],
                                     RawEvbuffer(send_buffer[index]));
      }
    });

    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto partition = task_context->WriteVariable(d_partition);
      const int num_neighbor = partition->GetNumNeighbors();
      EXPECT_GATHER_SIZE(num_neighbor);
      auto recv_buffer = task_context->OrderedGather<RawEvbuffer>();
      std::vector<struct evbuffer*> sort_buffer;
      sort_buffer.reserve(recv_buffer.size());
      for (auto& pair : recv_buffer) {
        sort_buffer.push_back(pair.second.buffer);
      }
      partition->RecvExchangeGhostCells(&sort_buffer);
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
      std::vector<struct evbuffer*> send_buffer;
      partition->SendDensityGhost(&neighbors, &send_buffer);
      for (size_t index = 0; index < neighbors.size(); ++index) {
        task_context->OrderedScatter(neighbors[index],
                                     RawEvbuffer(send_buffer[index]));
      }
    });

    WriteAccess(d_partition);
    Gather([=](CanaryTaskContext* task_context) -> int {
      auto partition = task_context->WriteVariable(d_partition);
      const int num_neighbor = partition->GetNumNeighbors();
      EXPECT_GATHER_SIZE(num_neighbor);
      auto recv_buffer = task_context->OrderedGather<RawEvbuffer>();
      std::vector<struct evbuffer*> sort_buffer;
      sort_buffer.reserve(recv_buffer.size());
      for (auto& pair : recv_buffer) {
        sort_buffer.push_back(pair.second.buffer);
      }
      partition->RecvDensityGhost(&sort_buffer);
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

    WriteAccess(d_partition);
    Scatter([=](CanaryTaskContext* task_context) {
      auto partition = task_context->WriteVariable(d_partition);
      task_context->Scatter(0, partition->ComputeEnergy());
    });

    WriteAccess(d_global);
    Gather([=](CanaryTaskContext* task_context) -> int {
      EXPECT_GATHER_SIZE(task_context->GetScatterParallelism());
      float sum = 0;
      sum = task_context->Reduce(sum, std::plus<float>());
      printf("Total energy: %f\n", sum);
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
      LoadFlag("fold_x", FLAG_app_fold_z, archive);
      LoadFlag("fold_y", FLAG_app_fold_z, archive);
      LoadFlag("fold_z", FLAG_app_fold_z, archive);
      LoadFlag("fold_depth_y", FLAG_app_fold_depth_y, archive);
      LoadFlag("frames", FLAG_app_frames, archive);
      LoadFlag("filename", FLAG_app_filename, archive);
    }
  }
};

}  // namespace canary

REGISTER_APPLICATION(::canary::ParsecApplication);
