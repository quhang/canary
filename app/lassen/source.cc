#include "source.h"
#include "lassen-util.h"
#include "simulation.h"

namespace Lassen {

Source::~Source() {}

/**


 */

void PointSource::initialize(const Simulation *sim) {
  const Mesh *mesh = sim->getMesh();
  const std::vector<Real> &zoneVelocity = sim->getZoneVelocity();

  // reset the source to done, it will be marked as not done if a node is near.
  done = true;

  // compute the minVelocity of the point source by finding the node
  // that is closest to the center.  Since this is only done once, no
  // attempt is made to make this efficient.  We will just look at
  // every node.
  Real minr2 = LassenUtil::maxReal;
  Real minZoneVel = LassenUtil::maxReal;
  Real radius2 = LassenUtil::sqr(radius);

  // start with the highest possible value
  minVelocity = sim->getMaxVelocity();

  for (int i = 0; i < mesh->nnodes; ++i) {
    const Point &nodePoint = mesh->nodePoint[i];
    Real r2 = nodePoint.distance2(center);
    if (r2 > radius2) {
      continue;
    }
    if (r2 < minr2) {
      // The source is not done.
      done = false;

      // loop through zones of the node
      int zoneStart = mesh->nodeToZoneOffset[i];
      int zoneEnd = mesh->nodeToZoneOffset[i + 1];
      for (int j = zoneStart; j < zoneEnd; ++j) {
        int zoneIndex = mesh->nodeToZone[j];
        minZoneVel = std::min(minZoneVel, zoneVelocity[zoneIndex]);
      }
      minr2 = r2;
      minVelocity = minZoneVel;
    }
  }
}

bool PointSource::process(const Mesh *mesh, double timer,
                          double narrowBandWidth,
                          std::vector<NodeState::Enum> &nodeState,
                          std::vector<Real> &nodeLevel,
                          std::vector<Real> &nodeTimeReached,
                          std::vector<Point> &nodeImagePoint,
                          std::vector<Real> &nodeImageVelocity) {
  // If the soruce has already been processed, then do nothing
  if (done) {
    return false;
  }

  // Check the delay before continuing
  if (timer < delay + radius / minVelocity) {
    return false;
  }

  // Process the source now, so that next time we will be marked as done.
  done = true;

  // Loop over the mesh nodes to see which ones are in the radius of the source
  // BCM - this could be done with a spatial search to avoid the O(nnodes)
  // algorithm

  Real instantRadius = (timer - delay) * minVelocity;

  for (int i = 0; i < mesh->nnodes; ++i) {
    const Point &nodePoint = mesh->nodePoint[i];

    // Distance from the node to the source point
    Real r = nodePoint.distance(center);

    // Distance from the node to the radius
    Real level = r - instantRadius;

    // Check to see if this node is too far away from the center to matter.
    if (level > narrowBandWidth) {
      continue;
    }

    // If the new level is less than the current level, then reset
    // the image level and image point data.
    if (level < nodeLevel[i] && level < narrowBandWidth) {
      // Update the node level
      nodeLevel[i] = level;

      // Update the velocity at the image point
      nodeImageVelocity[i] = minVelocity;

      // Update the image point to be on the instant radius
      if (r ==
          0.0) {  /// BCM maybe not quite right put it doesn't really matter.
        nodeImagePoint[i] = Point(center.x + instantRadius, center.y, center.z);
      } else {
        Real ratio = instantRadius / r;
        nodeImagePoint[i] = Point(center, nodePoint, ratio);
      }
    }

    // If the node is within the narrowband
    if (std::abs(nodeLevel[i]) < narrowBandWidth) {
      nodeState[i] = NodeState::NARROWBAND;
    }

    // If the node is behind the narrowband, mark it as reached.
    // and compute the time it was reached.
    if (nodeLevel[i] < 0.0) {
      nodeTimeReached[i] = timer + nodeLevel[i] / minVelocity;
      // it is possible that nodeTimeReached can be slightly negative due to
      // floating point
      // fixme: for now, set to 0.0 in this case
      nodeTimeReached[i] = std::max(0.0, nodeTimeReached[i]);
      if (nodeLevel[i] < -narrowBandWidth) {
        nodeState[i] = NodeState::REACHED;
      }
    }
  }

  return true;
}
};
