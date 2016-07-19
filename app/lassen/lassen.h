#ifndef LASSEN_H
#define LASSEN_H

#include <vector>
#include <iostream>
#include "lassen-util.h"
#include <algorithm>

namespace Lassen {

////////////////////////////////////////////////////////////////////////////////
/**
   NodeState

 */

namespace NodeState {
enum Enum { REACHED = 1, NARROWBAND = 2, NEARBY = 3, UNREACHED = 4 };
}

////////////////////////////////////////////////////////////////////////////////
/**
   Point

   A struct to hold an x,y,z position
 */

class Point {
 public:
  Point() : x(0.0), y(0.0), z(0.0) {}

  Point(Real ix, Real iy, Real iz = 0.0) : x(ix), y(iy), z(iz) {}

  Point(const Point &a, const Point &b, Real ratio) {
    x = (1.0 - ratio) * a.x + ratio * b.x;
    y = (1.0 - ratio) * a.y + ratio * b.y;
    z = (1.0 - ratio) * a.z + ratio * b.z;
  }

  Real distance(const Point &b) const {
    return LassenUtil::distance(x, y, z, b.x, b.y, b.z);
  }
  Real distance2(const Point &b) const {
    return LassenUtil::distance2(x, y, z, b.x, b.y, b.z);
  }

  Real x;
  Real y;
  Real z;
  template<class Archive> void serialize(Archive& archive) {
    archive(x, y, z);
  }
};

std::ostream &operator<<(std::ostream &o, const Point &p);

////////////////////////////////////////////////////////////////////////////////
/**
   Vector

   A struct to hold an x,y,z direction
 */

class Vector {
 public:
  Vector() : x(0), y(0), z(0) {}
  Vector(const Point &a, const Point &b)
      : x(b.x - a.x), y(b.y - a.y), z(b.z - a.z) {}
  Vector(Real ix, Real iy, Real iz = 0.0) : x(ix), y(iy), z(iz) {}
  Real length() { return std::sqrt(length2()); }
  Real length2() { return x * x + y * y + z * z; }
  void normalize() {
    Real l = length();
    if (l > LassenUtil::pico) {
      x = x / l;
      y = y / l;
      z = z / l;
    }
  }
  Real dot(const Vector &b) const { return x * b.x + y * b.y + z * b.z; }

  Real x, y, z;
};

std::ostream &operator<<(std::ostream &o, const Vector &p);

////////////////////////////////////////////////////////////////////////////////
/**
   BoundingBox

*/

struct BoundingBox {
  Point min;
  Point max;
  template<class Archive> void serialize(Archive& archive) {
    archive(min, max);
  }
};

void computeBoundingBox(const std::vector<Point> &points, int npoints,
                        BoundingBox &bb);

/// FIXME- remove domainID\, replace with something less specific
/// FIXME- remove this function and use the single interact one instead
void findBoundingBoxIntersections(
    int domainID, const std::vector<BoundingBox> &allBoundingBox,
    Real overlapDistance, std::vector<int> &overlap);

bool boundingBoxIntersection(const BoundingBox &a, const BoundingBox &b,
                             Real overlapDistance);

////////////////////////////////////////////////////////////////////////////////
/**
   Spatial Grid

*/

struct SpatialGrid {
  SpatialGrid(const std::vector<Point> &points, Real boxSize);

  bool gridLocation(const Point &point, int &i, int &j, int &k) const {
    Real dx = point.x - bb.min.x;
    Real dy = point.y - bb.min.y;
    Real dz = point.z - bb.min.z;
    i = (int)(std::floor(dx / boxSize));
    j = (int)(std::floor(dy / boxSize));
    k = (int)(std::floor(dz / boxSize));
    return true;
  }

  const std::vector<int> &getIndicies(int i, int j, int k) const {
    if (i < 0 || i >= numI) return empty;
    if (j < 0 || j >= numJ) return empty;
    if (k < 0 || k >= numK) return empty;
    if (grid[i].empty()) return empty;
    if (grid[i][j].empty()) return empty;
    return grid[i][j][k];
  }

 protected:
  BoundingBox bb;
  int numI, numJ, numK;
  Real boxSize;
  std::vector<std::vector<std::vector<std::vector<int> > > > grid;
  std::vector<int> empty;
};

////////////////////////////////////////////////////////////////////////////////
/**
   PlanarFront

   A collection of facets
 */

class PlanarFront {
 public:
  PlanarFront() {
    nfacets = 0;
    facetToVertexOffset.push_back(0);
  }

  void addFacet(int zoneIndex, int npoints, const Point *points,
                const Point &center, Real velocity) {
    vertices.insert(vertices.end(), points, points + npoints);
    facetToVertexOffset.push_back(vertices.size());
    facetZone.push_back(zoneIndex);
    facetCenter.push_back(center);
    facetVelocity.push_back(velocity);
    nfacets++;
  }

  // Copy a single facet from the other front to this front
  void addFacet(const PlanarFront *other, int index) {
    int vertexBegin = other->facetToVertexOffset[index];
    int vertexEnd = other->facetToVertexOffset[index + 1];
    int nVertex = vertexEnd - vertexBegin;
    this->addFacet(other->facetZone[index], nVertex,
                   &(other->vertices[vertexBegin]), other->facetCenter[index],
                   other->facetVelocity[index]);
  }

  // Copy all the facets from the other front to this front
  // Fixme: this can be done more efficiently by doing it all at once
  void addFacet(const PlanarFront *other) {
    for (int index = 0; index < other->nfacets; ++index) {
      this->addFacet(other, index);
    }
  }

  Point imagePointOnFacet2D(int facet, const Point &point) const;
  Point imagePointOnFacet3D(int facet, const Point &point) const;

  std::vector<Point> vertices;
  std::vector<int> facetToVertexOffset;
  std::vector<int> facetZone;
  std::vector<Point> facetCenter;  // FIXME -- see if we can get rid of this?
  std::vector<Real> facetVelocity;
  int nfacets;

  template<class Archive> void serialize(Archive& archive) {
    archive(vertices, facetToVertexOffset, facetZone, facetCenter,
            facetVelocity, nfacets);
  }

 private:
  // fixme - consider combining image point on path2d/3d for simplicity
  Point imagePointOnPath2D(const Point &a, const Point &b,
                           const Point &point) const;
  Point imagePointOnPath3D(const Point &a, const Point &b,
                           const Point &point) const;
  Point imagePointOnTriangle(const Point &a, const Point &b, const Point &c,
                             const Point &point) const;
};

////////////////////////////////////////////////////////////////////////////////
/**
   Mesh

 */

class Mesh {
 public:
  // dimension: 2 or 3
  int dim;

  // nodeToZoneOffset[i] is an offset into the nodeToZone array
  // number of zones per node:  nodeToZoneOffset[i+1] - nodeToZoneOffset[i]
  // nodeToZone[j] is the index of a zone
  int nnodes;
  std::vector<int> nodeToZoneOffset;
  std::vector<int> nodeToZone;
  std::vector<Point> nodePoint;

  /// Zone Data
  int nzones;
  int zoneToNodeCount;
  std::vector<int> zoneToNode;

  /// For distributed meshes (FIXME: move to domain?)
  std::vector<GlobalID> nodeGlobalID;
  int nLocalZones;
  int nLocalNodes;

  template<class Archive> void serialize(Archive& archive) {
    archive(dim, nnodes, nodeToZoneOffset, nodeToZone, nodePoint);
    archive(nzones, zoneToNodeCount, zoneToNode, nodeGlobalID, nLocalZones,
            nLocalNodes);
  }
};

void computeZoneCenters(const Mesh *mesh, std::vector<Point> &centers);

////////////////////////////////////////////////////////////////////////////////
/**
   Domain

 */

class Domain {
 public:
  Domain() : mesh(new Mesh) {}
  ~Domain() { delete mesh; }

  Mesh *mesh;

  // A number from [0, numDomains) for this domain.
  int domainID;

  // number of domains in the simulation
  int numDomains;

  // neighbor domains
  std::vector<int> neighborDomains;

  std::vector<std::pair<GlobalID, int> > globalToLocalNode;

  /// FIXME: Find a better spot for this.  Maybe on domain?
  void initializeGlobalToLocal() {
    globalToLocalNode.resize(mesh->nnodes);
    for (int i = 0; i < mesh->nnodes; ++i) {
      globalToLocalNode[i].first = mesh->nodeGlobalID[i];
      globalToLocalNode[i].second = i;
    }
    std::sort(globalToLocalNode.begin(), globalToLocalNode.end());
  }

  inline int getNodeIndex(const GlobalID &gid) const {
    std::pair<GlobalID, int> key(gid, -1);
    std::vector<std::pair<GlobalID, int> >::const_iterator it =
        std::lower_bound(globalToLocalNode.begin(), globalToLocalNode.end(),
                         key);
    if (it == globalToLocalNode.end()) return -1;
    if (it->first == gid) return it->second;
    return -1;
  }

  template<class Archive> void serialize(Archive& archive) {
    archive(*mesh);
    archive(domainID, numDomains, neighborDomains, globalToLocalNode);
  }
};

class CoordinateFrame {
 public:
  // local coordinate system
  Point origin;
  Vector dir[3];

  void setCoord2D(const Point &point, const Vector &normal) {
    // fixme - asert the normal is normalized.
    origin = point;
    dir[0] = normal;
    dir[1] = Vector(normal.y, -normal.x);
  }
};

class LocalNormalFront {
 public:
  CoordinateFrame frame;
  Real coef[6];

  void setCoef2D(Real c0, Real c1, Real c2) {
    coef[0] = c0;
    coef[1] = c1;
    coef[2] = c2;
  }
};
}

#endif
