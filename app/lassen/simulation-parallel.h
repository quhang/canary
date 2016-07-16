#ifndef SIMULATION_PARALLEL_H
#define SIMULATION_PARALLEL_H

#include "simulation.h"
#include "lassen.h"

#include "shared/internal.h"

namespace Lassen {

// forward declare
struct NodeData;

class SimulationParallel : public Simulation {
 public:
  SimulationParallel();

  ~SimulationParallel() {}

  bool isNodeOwner(int nodeIndex) const;

  /// protected:
  //// FIXME these should be protected.  need to find better way for the Charm++
  /// version to inherit from this.

  // initialization functions
  void initializeBoundaryNodes(std::vector<int> &boundaryNodes) const;
  void initializeCommunicatingNodes(const std::vector<int> &boundaryNodes,
                                    std::vector<int> &communicatingNodes) const;

  // determine which zones are near to any node of another domain,
  // such that communication will be required to send.
  // return true if any of the zones needs to send it's facet to the neighbor

  bool findFacetDomainOverlap(
      const SpatialGrid &grid, const std::vector<Point> &zoneCenter,
      const int overlapDomainID,
      const std::vector<Point> &neighborBoundaryPoints,
      const BoundingBox &neighborBoundingBox, const Real cutoffDistance,
      std::vector<std::pair<int, int> > &facetToDomainPair);

  void constructFacetToDomain(
      std::vector<std::pair<int, int> > &facetToDomainPair);

 protected:
  void nodeCommunicationCreateMsg(const std::vector<int> &communicatingNodes,
                                  std::vector<GlobalID> &sendBuffer);
  void nodeCommunicationProcessMsg(
      int neighborIndex, const std::vector<GlobalID> &buffer,
      std::vector<std::pair<int, int> > &nodeToDomainPair);
  void nodeCommunicationComplete(
      std::vector<std::pair<int, int> > &nodeToDomainPair);

  void gatherNodeData(std::vector<std::vector<NodeData> > &sendBuffer);
  void accumulateNodeData(const std::vector<NodeData> &recvBuffer);

 public:
  // nodeToDomain
  std::vector<int> nodeToDomainOffset;
  std::vector<int> nodeToDomain;  // the values are indices into the
                                  // domain->domainNeighbors array

  // facetToDomain
  std::vector<int> facetToDomainOffset;
  std::vector<int> facetToDomain;  // the values are domain ID's

  // an array to map from facetToDomain numbers to real domain indicies.
  std::vector<int> facetNeighbors;
};

// Used in the synchronizeNodeData algorithm
struct NodeData {
  NodeData() {}
  NodeData(GlobalID globalID, Real level, Real timeReached, Real velocity,
           const Point &imagePoint)
      : globalID(globalID),
        level(level),
        timeReached(timeReached),
        velocity(velocity),
        imagePoint(imagePoint) {}

  GlobalID globalID;
  Real level;
  Real timeReached;
  Real velocity;
  Point imagePoint;

  template<class Archive> void serialize(Archive& archive) {
    archive(globalID, level, timeReached, velocity, imagePoint);
  }
};
}

#endif
