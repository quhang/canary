/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file app/grid_helper.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class GridHelper.
 */

#ifndef CANARY_APP_GRID_HELPER_H_
#define CANARY_APP_GRID_HELPER_H_

#include <algorithm>
#include <iostream>
#include <utility>

namespace helper {

namespace internal {
/**
 * A 3-dimension point, which supports dimension-wide operations.
 */
template <typename T>
struct PointBase {
 private:
  typedef PointBase<T> Self;

 public:
  //! Data members.
  T x, y, z;
  //! Constructor.
  PointBase() : PointBase(0, 0, 0) {}
  PointBase(T in_x, T in_y, T in_z) : x(in_x), y(in_y), z(in_z) {}
  //! Serialization funciton.
  template <class Archive>
  void serialize(Archive& archive) {
    archive(x, y, z);
  }
  //! In-place assignment operators.
  inline Self& operator+=(const Self& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  inline Self& operator+=(T s) {
    x += s;
    y += s;
    z += s;
    return *this;
  }
  inline Self& operator-=(const Self& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  inline Self& operator-=(T s) {
    x -= s;
    y -= s;
    z -= s;
    return *this;
  }
  inline Self& operator*=(const Self& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
  }
  inline Self& operator*=(T s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  inline Self& operator/=(const Self& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  inline Self& operator/=(T s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }
  //! Out-of-place operators.
  inline Self operator-() const { return Self(-x, -y, -z); }
  inline Self operator+(T s) const { return Self(x + s, y + s, z + s); }
  inline Self operator-(T s) const { return Self(x - s, y - s, z - s); }
  inline Self operator*(T s) const { return Self(x * s, y * s, z * s); }
  inline Self operator/(T s) const { return Self(x / s, y / s, z / s); }
  inline Self operator+(const Self& v) const {
    return Self(x + v.x, y + v.y, z + v.z);
  }
  inline Self operator-(const Self& v) const {
    return Self(x - v.x, y - v.y, z - v.z);
  }
  inline Self operator*(const Self& v) const {
    return Self(x * v.x, y * v.y, z * v.z);
  }
  inline Self operator/(const Self& v) const {
    return Self(x / v.x, y / v.y, z / v.z);
  }
  //! Statistics.
  inline T Product() const { return x * y * z; }
};
}  // namespace internal

//! Two specializations.
typedef internal::PointBase<float> Point;
typedef internal::PointBase<int> IntPoint;
//! Cast facility.
template <typename T, typename InputType>
internal::PointBase<T> cast_point(const InputType& input) {
  return internal::PointBase<T>(input.x, input.y, input.z);
}
//! Printing facility.
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const internal::PointBase<T>& obj) {
  os << obj.x << ' ' << obj.y << ' ' << obj.z;
  return os;
}
template <typename T>
inline std::istream& operator>>(std::istream& is, internal::PointBase<T>& obj) {
  is >> obj.x >> obj.y >> obj.z;
  return is;
}

/**
 * Stores the metadata of A grid.
 *
 * A grid is split into many subgrids. Each grid or subgrid is a cube of cells,
 * and reprensents a geometric domain.
 */
class Grid {
 public:
  void Initialize(const std::pair<Point, Point>& domain,
                  const IntPoint& grid_size, const IntPoint& split,
                  int subgrid_rank, int ghost_bandwidth = 0) {
    // Initializes global domain.
    domain_ = domain;
    domain_size_ = domain_.second - domain_.first;
    // Initializes global grid: the right boundary is exclusive.
    grid_size_ = grid_size;
    grid_ = std::make_pair(IntPoint{0, 0, 0}, grid_size);
    // Calculates a cell size.
    cell_size_ = domain_size_ / cast_point<float>(grid_size_);
    // Initializes the rank and index of the load grid.
    split_ = split;
    subgrid_rank_ = subgrid_rank;
    subgrid_index_ = rank_to_index(subgrid_rank, split);
    // Initializes the ghost bandwidth.
    ghost_bandwidth_ = ghost_bandwidth;
    // Initializes the local grid.
    InitializeLocalGrid();
    initialized_ = true;
  }
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(initialized_);
    archive(domain_, domain_size_, grid_, grid_size_);
    archive(cell_size_);
    archive(split_);
    archive(subgrid_rank_, subgrid_index_);
    archive(ghost_bandwidth_);
    archive(subdomain_, subdomain_size_, subgrid_, subgrid_size_);
  }
  //! For debugging.
  void Print() {
    auto& ss = std::cout;
    ss << "domain=(" << domain_.first << ")-(" << domain_.second << ")\n";
    ss << "domainsize=" << domain_size_ << "\n";
    ss << "grid=(" << grid_.first << ")-(" << grid_.second << ")\n";
    ss << "gridsize=" << grid_size_ << "\n";
    ss << "subdomain=(" << subdomain_.first << ")-(" << subdomain_.second
       << ")\n";
    ss << "subdomainsize=" << subdomain_size_ << "\n";
    ss << "subgridindex=" << subgrid_index_ << "\n";
    ss << "subgrid=(" << subgrid_.first << ")-(" << subgrid_.second << ")\n";
    ss << "subgridsize=" << subgrid_size_ << "\n";
  }
  //! Whether a cell is in the grid.
  inline bool ContainDomain(float cord_x, float cord_y, float cord_z) const {
    int global_x = (int)((cord_x - domain_.first.x) / cell_size_.x);
    int global_y = (int)((cord_y - domain_.first.y) / cell_size_.y);
    int global_z = (int)((cord_z - domain_.first.z) / cell_size_.z);
    global_x = std::max(std::min(global_x, grid_.second.x - 1), grid_.first.x);
    global_y = std::max(std::min(global_y, grid_.second.y - 1), grid_.first.y);
    global_z = std::max(std::min(global_z, grid_.second.z - 1), grid_.first.z);
    return Contain(global_x, global_y, global_z);
  }
  //! Whether a cell is in the grid.
  inline bool Contain(int global_x, int global_y, int global_z) const {
    return global_x >= subgrid_.first.x && global_y >= subgrid_.first.y &&
           global_z >= subgrid_.first.z && global_x < subgrid_.second.x &&
           global_y < subgrid_.second.y && global_z < subgrid_.second.z;
  }
  //! Gets the local cell rank from a global cell cord.
  inline int GetLocalCellRankDomain(float cord_x, float cord_y,
                                    float cord_z) const {
    const int x =
        std::max(std::min(subgrid_.second.x - 1,
                          (int)((cord_x - domain_.first.x) / cell_size_.x)),
                 subgrid_.first.x);
    const int y =
        std::max(std::min(subgrid_.second.x - 1,
                          (int)((cord_y - domain_.first.y) / cell_size_.y)),
                 subgrid_.first.y);
    const int z =
        std::max(std::min(subgrid_.second.z - 1,
                          (int)((cord_z - domain_.first.z) / cell_size_.z)),
                 subgrid_.first.z);
    return GetLocalCellRank(x, y, z);
  }
  //! Gets the local cell rank from a global cell index.
  inline int GetLocalCellRank(int global_x, int global_y, int global_z) const {
    return (global_x - subgrid_.first.x) +
           ((global_y - subgrid_.first.y) +
            (global_z - subgrid_.first.z) * subgrid_size_.y) *
               subgrid_size_.x;
  }
  //! Gets the global cell rank from a global cell index.
  inline int GetGlobalCellRank(int global_x, int global_y, int global_z) const {
    return global_x + (global_y + global_z * grid_size_.y) * grid_size_.x;
  }
  //! Transform a global cell rank to a local cell rank.
  inline int GlobalCellRankToLocal(int global_rank) const {
    const auto& global_index = rank_to_index(global_rank, grid_size_);
    return GetLocalCellRank(global_index.x, global_index.y, global_index.z);
  }
  //! Gets the neighbor subgrid, and returns true if success.
  bool GetNeighborSubgrid(int dx, int dy, int dz, int ghost_bandwidth,
                          Grid* grid) {
    IntPoint neighbor_index = subgrid_index_;
    neighbor_index.x += dx;
    neighbor_index.y += dy;
    neighbor_index.z += dz;
    if (neighbor_index.x < 0 || neighbor_index.y < 0 || neighbor_index.z < 0 ||
        neighbor_index.x >= split_.x || neighbor_index.y >= split_.y ||
        neighbor_index.z >= split_.z) {
      return false;
    }
    grid->Initialize(domain_, grid_size_, split_,
                     index_to_rank(neighbor_index, split_), ghost_bandwidth);
    return true;
  }
  //! Gets the rank of the subgrid.
  int GetSubgridRank() const { return index_to_rank(subgrid_index_, split_); }
  //! Helper functions for accesssing all the cells.
  inline int get_sx() const { return subgrid_.first.x; }
  inline int get_sy() const { return subgrid_.first.y; }
  inline int get_sz() const { return subgrid_.first.z; }
  inline int get_ex() const { return subgrid_.second.x; }
  inline int get_ey() const { return subgrid_.second.y; }
  inline int get_ez() const { return subgrid_.second.z; }
  inline int get_count() const { return subgrid_size_.Product(); }
  std::pair<Point, Point> get_domain() const { return domain_; }
  IntPoint get_grid_size() const { return grid_size_; }

 private:
  //! Transforms an index to a rank.
  inline int index_to_rank(const IntPoint& index, const IntPoint& split) const {
    return index.x + (index.y + index.z * split.y) * split.x;
  }
  //! Transforms a rank to an index.
  inline IntPoint rank_to_index(int rank, const IntPoint& split) const {
    const int x = rank % split.x;
    rank -= x;
    rank /= split.x;
    const int y = rank % split.y;
    rank -= y;
    rank /= split.y;
    const int z = rank;
    return {x, y, z};
  }
  //! Initializes local subgrid.
  void InitializeLocalGrid() {
    // Calculates subdomain.
    subdomain_size_ = domain_size_ / cast_point<float>(split_);
    subdomain_.first =
        domain_.first + subdomain_size_ * cast_point<float>(subgrid_index_);
    subdomain_.second = subdomain_.first + subdomain_size_;
    // Calculates subgrid.
    const auto full_subgrid_size = grid_size_ / split_;
    const auto remain_subgrid_size = grid_size_ - full_subgrid_size * split_;
    subgrid_.first = subgrid_index_ * full_subgrid_size;
    subgrid_.first.x += std::min(subgrid_index_.x, remain_subgrid_size.x);
    subgrid_.first.y += std::min(subgrid_index_.y, remain_subgrid_size.y);
    subgrid_.first.z += std::min(subgrid_index_.z, remain_subgrid_size.z);
    subgrid_.second = subgrid_.first + full_subgrid_size;
    if (subgrid_index_.x < remain_subgrid_size.x) {
      ++subgrid_.second.x;
    }
    if (subgrid_index_.y < remain_subgrid_size.y) {
      ++subgrid_.second.y;
    }
    if (subgrid_index_.z < remain_subgrid_size.z) {
      ++subgrid_.second.z;
    }
    subgrid_size_ = subgrid_.second - subgrid_.first;
    // Updates for ghost regions.
    if (ghost_bandwidth_ != 0) {
      subgrid_.first -= ghost_bandwidth_;
      subgrid_.second += ghost_bandwidth_;
      subgrid_size_ = subgrid_.second - subgrid_.first;
      subdomain_.first -= cell_size_ * ghost_bandwidth_;
      subdomain_.second += cell_size_ * ghost_bandwidth_;
      subdomain_size_ = subdomain_.second - subdomain_.first;
      TrimExternalBoundary();
    }
  }
  //! Trim out of region cells.
  void TrimExternalBoundary() {
    subgrid_.first.x = std::max(subgrid_.first.x, grid_.first.x);
    subgrid_.first.y = std::max(subgrid_.first.y, grid_.first.y);
    subgrid_.first.z = std::max(subgrid_.first.z, grid_.first.z);
    subgrid_.second.x = std::min(subgrid_.second.x, grid_.second.x);
    subgrid_.second.y = std::min(subgrid_.second.y, grid_.second.y);
    subgrid_.second.z = std::min(subgrid_.second.z, grid_.second.z);
    subdomain_.first.x = std::max(subdomain_.first.x, domain_.first.x);
    subdomain_.first.y = std::max(subdomain_.first.y, domain_.first.y);
    subdomain_.first.z = std::max(subdomain_.first.z, domain_.first.z);
    subdomain_.second.x = std::min(subdomain_.second.x, domain_.second.x);
    subdomain_.second.y = std::min(subdomain_.second.y, domain_.second.y);
    subdomain_.second.z = std::min(subdomain_.second.z, domain_.second.z);
  }
  //! Initialization flag.
  bool initialized_ = false;
  //! Global grid.
  std::pair<Point, Point> domain_;
  Point domain_size_;
  std::pair<IntPoint, IntPoint> grid_;
  IntPoint grid_size_;
  //! Size of a cell.
  Point cell_size_;
  //! Partiioning info.
  IntPoint split_;
  int subgrid_rank_;
  IntPoint subgrid_index_;
  //! Ghost bandwidth.
  int ghost_bandwidth_;
  //! Local subgrid.
  std::pair<Point, Point> subdomain_;
  Point subdomain_size_;
  std::pair<IntPoint, IntPoint> subgrid_;
  IntPoint subgrid_size_;
};

}  // namespace helper
#endif  // CANARY_APP_GRID_HELPER_H_
