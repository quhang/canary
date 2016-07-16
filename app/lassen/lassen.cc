#define DEBUG 0

#include <iostream>
#include "lassen.h"

namespace Lassen {

////////////////////////////////////////////////////////////////////////////////
// Point related functions
////////////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &o, const Point &p) {
  o << "(" << p.x << ", " << p.y << ", " << p.z << ")";
  return o;
}

////////////////////////////////////////////////////////////////////////////////
// Vector related functions
////////////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &o, const Vector &p) {
  o << "(" << p.x << ", " << p.y << ", " << p.z << ")";
  return o;
}

////////////////////////////////////////////////////////////////////////////////
// Bounding Box related functions
////////////////////////////////////////////////////////////////////////////////

void computeBoundingBox(const std::vector<Point> &points, int npoints,
                        BoundingBox &boundingBox) {
  boundingBox.min =
      Point(LassenUtil::maxReal, LassenUtil::maxReal, LassenUtil::maxReal);
  boundingBox.max =
      Point(LassenUtil::minReal, LassenUtil::minReal, LassenUtil::minReal);

  for (int i = 0; i < npoints; ++i) {
    const Point &p = points[i];

    boundingBox.min.x = std::min(boundingBox.min.x, p.x);
    boundingBox.min.y = std::min(boundingBox.min.y, p.y);
    boundingBox.min.z = std::min(boundingBox.min.z, p.z);

    boundingBox.max.x = std::max(boundingBox.max.x, p.x);
    boundingBox.max.y = std::max(boundingBox.max.y, p.y);
    boundingBox.max.z = std::max(boundingBox.max.z, p.z);
  }
}

bool boundingBoxIntersection(const BoundingBox &a, const BoundingBox &b,
                             Real overlapDistance) {
  if (a.min.x - overlapDistance > b.max.x ||
      a.max.x + overlapDistance < b.min.x) {
    return false;
  }
  if (a.min.y - overlapDistance > b.max.y ||
      a.max.y + overlapDistance < b.min.y) {
    return false;
  }
  if (a.min.z - overlapDistance > b.max.z ||
      a.max.z + overlapDistance < b.min.z) {
    return false;
  }
  return true;
}

void findBoundingBoxIntersections(
    int domainID, const std::vector<BoundingBox> &allBoundingBox,
    Real overlapDistance, std::vector<int> &overlap) {
  const BoundingBox &localBoundingBox = allBoundingBox[domainID];
  int numBoundingBox = allBoundingBox.size();
  for (int i = 0; i < numBoundingBox; ++i) {
    if (i == domainID) continue;
    if (boundingBoxIntersection(allBoundingBox[i], localBoundingBox,
                                overlapDistance)) {
      overlap.push_back(i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Spatial grid related functions
////////////////////////////////////////////////////////////////////////////////

SpatialGrid::SpatialGrid(const std::vector<Point> &points, Real boxSize)
    : boxSize(boxSize) {
  computeBoundingBox(points, points.size(), bb);
  numI = std::ceil((bb.max.x - bb.min.x) / boxSize);
  numJ = std::ceil((bb.max.y - bb.min.y) / boxSize);
  numK = std::ceil((bb.max.z - bb.min.z) / boxSize);
  numI = std::max(numI, 1);
  numJ = std::max(numJ, 1);
  numK = std::max(numK, 1);

  // create the grid
  grid.resize(numI);

  // populate the grid.
  for (size_t i = 0; i < points.size(); ++i) {
    const Point &point = points[i];
    int ii, jj, kk;
    gridLocation(point, ii, jj, kk);
    if (grid[ii].empty()) {
      grid[ii].resize(numJ);
    }
    if (grid[ii][jj].empty()) {
      grid[ii][jj].resize(numK);
    }

    grid[ii][jj][kk].push_back(i);
  }
}

////////////////////////////////////////////////////////////////////////////////
// PlanarFront related functions
////////////////////////////////////////////////////////////////////////////////

Point PlanarFront::imagePointOnFacet2D(int facet, const Point &point) const {
  int startPoint = facetToVertexOffset[facet];
  int endPoint = facetToVertexOffset[facet + 1];
  int numPoint = endPoint - startPoint;

  if (numPoint == 2) {
    return imagePointOnPath2D(vertices[startPoint], vertices[startPoint + 1],
                              point);
  } else if (numPoint == 4) {
    // A rare case

    Point imageMin = imagePointOnPath2D(vertices[startPoint],
                                        vertices[startPoint + 1], point);
    Real heightMin = imageMin.distance2(point);

    for (int i = 1; i < 4; ++i) {
      int j = (i + 1) % 4;
      Point image = imagePointOnPath2D(vertices[startPoint + i],
                                       vertices[startPoint + j], point);
      Real height = image.distance2(point);
      if (height < heightMin) {
        imageMin = image;
        heightMin = height;
      }
    }
    return imageMin;

  } else {
    assert(0);
  }

  return Point(LassenUtil::maxReal, LassenUtil::maxReal, LassenUtil::maxReal);
}

Point PlanarFront::imagePointOnPath2D(const Point &a, const Point &b,
                                      const Point &point) const {
  Real udX = point.x - a.x;
  Real udY = point.y - a.y;

  Real vdX = b.x - a.x;
  Real vdY = b.y - a.y;

  Real length = LassenUtil::dot(udX, udY, vdX, vdY);
  Real lambda = length / LassenUtil::dot(vdX, vdY, vdX, vdY);

  if (lambda < 0.0) {
    return a;
  } else if (lambda < 1.0) {
    return Point(a.x + lambda * vdX, a.y + lambda * vdY);
  } else {
    return b;
  }
}

Point PlanarFront::imagePointOnPath3D(const Point &a, const Point &b,
                                      const Point &point) const {
  Vector u(point, a);
  Vector v(a, b);

  Real length = u.dot(v);
  Real lambda = length / v.length2();

  if (lambda < 0.0) {
    return a;
  } else if (lambda < 1.0) {
    return Point(a.x + lambda * v.x, a.y + lambda * v.y, a.z + lambda * v.z);
  } else {
    return b;
  }
}

Point PlanarFront::imagePointOnFacet3D(int facet, const Point &point) const {
  int startPoint = facetToVertexOffset[facet];
  int endPoint = facetToVertexOffset[facet + 1];
  int numPoint = endPoint - startPoint;

  Point ip;

  if (numPoint == 3) {
    ip = imagePointOnTriangle(vertices[startPoint], vertices[startPoint + 1],
                              vertices[startPoint + 2], point);
    return ip;
  } else {
    const Point &center = facetCenter[facet];

    Point imageMin = imagePointOnTriangle(center, vertices[startPoint + 0],
                                          vertices[startPoint + 1], point);
    Real heightMin = imageMin.distance2(point);

    for (int i = 1; i < numPoint; ++i) {
      int j = (i + 1) % numPoint;
      Point image = imagePointOnTriangle(center, vertices[startPoint + i],
                                         vertices[startPoint + j], point);
      Real height = image.distance2(point);
      if (height < heightMin) {
        imageMin = image;
        heightMin = height;
      }
    }

    return imageMin;
  }
}

Point PlanarFront::imagePointOnTriangle(const Point &a, const Point &b,
                                        const Point &c,
                                        const Point &point) const {
  // project point into the plane of the triangle
  Vector r0(point, a);
  Vector r1(b, a);
  Vector r2(c, a);

  // q1 is the coordinates of the project point along r1
  Real q1 = r1.dot(r0) / r1.length2();

  // q2 is the coordinates of the project point along r2
  Real q2 = r2.dot(r0) / r2.length2();

  // Is the point in the triange
  if (q1 > 0 && q2 > 0 && (q1 + q2) < 1) {
    // a + q1*r1 + q2*r2
    return Point(a.x + q1 * r1.x + q2 * r2.x, a.y + q1 * r1.y + q2 * r2.y,
                 a.z + q1 * r1.z + q2 * r2.z);
  }

  // go over the edges
  // IMPROVE:  there are better algorithms for this.

  if (q1 < 0) {
    if (q2 < 0) {
      return a;
    } else if (q2 > 1) {
      return c;
    } else {
      // a + q2 *r2
      return Point(a.x + q2 * r2.x, a.y + q2 * r2.y, a.z + q2 * r2.z);
    }
  } else if (q2 < 0) {
    if (q1 > 1) {
      return b;
    } else {
      // a + q1 * r1
      return Point(a.x + q1 * r1.x, a.y + q1 * r1.y, a.z + q1 * r1.z);
    }
  } else {
    // should be on point b c
    Point e1 = imagePointOnPath3D(b, c, point);
    return e1;
  }
  return Point(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Mesh related functions
////////////////////////////////////////////////////////////////////////////////

void computeZoneCenters(const Mesh *mesh, std::vector<Point> &centers) {
  int zoneToNodeCount = mesh->zoneToNodeCount;
  centers.resize(mesh->nzones);
  for (int i = 0; i < mesh->nzones; ++i) {
    int startNode = i * zoneToNodeCount;
    Real x = 0.0;
    Real y = 0.0;
    Real z = 0.0;
    for (int j = 0; j < zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[startNode + j];
      const Point &nodePoint = mesh->nodePoint[nodeIndex];
      x += nodePoint.x;
      y += nodePoint.y;
      z += nodePoint.z;
    }
    x /= zoneToNodeCount;
    y /= zoneToNodeCount;
    z /= zoneToNodeCount;
    centers[i] = Point(x, y, z);
  }
}
};
