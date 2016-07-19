#ifndef SIMULATION_MPI
#define SIMULATION_MPI

// #include "mpi.h"
#include "simulation-parallel.h"
#include "lassen.h"

namespace Lassen {

class SimulationMPI : public SimulationParallel {
 public:
  SimulationMPI();
  ~SimulationMPI() {}

 protected:
  void packFacet(const PlanarFront& planarFront, int facetIndex,
                 std::vector<Real>& buffer) const;
  void unpackFacet(const std::vector<Real>& buffer, PlanarFront& planarFront);

 public:
  // Canary function functions.
  void functionLocalInitialize() { Simulation::initialize(); }
  void functionInitializeBoundaryNodes() {
    initializeBoundaryNodes(boundaryNodes);
  }
  void functionInitializeCommunicatingNodes() {
    initializeCommunicatingNodes(boundaryNodes, communicatingNodes);
  }
  void functionPrepareStep() {
    // These should be constant time operations.
    stepState.nodeTempArray.resize(mesh->nnodes);
    stepState.planarFront = PlanarFront();
  }
  void functionUpdateSources() { updateSources(); }
  void functionUpdateNarrowband() { updateNarrowband(); }
  void functionUpdateNearbyNodes() { updateNearbyNodes(); }
  void functionConstructFront() { constructFront(); }
  void functionConstructDistancesToFront() { constructDistancesToFront(); }
  void functionConvertNearbyNodesToNarrowBandNodes() {
    convertNearbyNodesToNarrowBandNodes();
  }
  void functionComputeNextTimeStep() { computeNextTimeStep(); }
  void functionComputeTimeIntegral() { computeTimeIntegral(); }

  void functionSendSyncExtreme(double input[4]) const {
    input[0] = minVelocity;
    input[1] = -maxVelocity;
    input[2] = minEdgeSize;
    input[3] = -narrowBandWidth;
  }
  void functionRecvSyncExtreme(double output[4]) {
    minVelocity = output[0];
    maxVelocity = -output[1];
    minEdgeSize = output[2];
    narrowBandWidth = -output[3];
  }

  void functionSendInitializeNodeCommunication(
      std::vector<GlobalID>& sendBuffer) const;
  void functionRecvInitializeNodeCommunication(
      const std::vector<std::vector<GlobalID> >& recvBuffer);

  void functionSendFirstInitializeFacetCommunication(BoundingBox& boundingBox)
      const;
  void functionRecvFirstInitializeFacetCommunication(
      const std::vector<BoundingBox> allBoundingBox);

  void functionSendSecondInitializeFacetCommunication(
      std::vector<Point>& sendBuffer) const;
  void functionRecvSecondInitializeFacetCommunication(
      const std::vector<std::vector<Point> >& recvBuffer);

  void functionSendThirdInitializeFacetCommunication() const;
  void functionRecvThirdInitializeFacetCommunication(
      const std::vector<int>& recvIsFacetNeighbor);

  void functionSendCommunicateFront(
      std::vector<std::vector<Real> >& sendBuffer) const;
  void functionRecvCommunicateFront(
      const std::vector<std::vector<Real> >& recvBuffer);

  void functionSendSynchronizeNodeData(
      std::vector<std::vector<NodeData> >& sendBuffer) const;
  void functionRecvSynchronizeNodeData(
      const std::vector<std::vector<NodeData> >& recvBuffer);

  void functionSendDoneCondition(int& localDone) {
    this->cycle++;
    this->time += this->dt;
    localDone = checkEndingCriteria();
  }
  void functionSendComputeError(double* a, double* b, double* c, double* d) {
    ComputeErrors& err = computeError;
    err.computeLocalErrors(this);
    *a = err.error1;
    *b = err.error2;
    *c = err.errorMax;
    *d = err.errorNodeCount;
  }

 public:
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(metadata);
    archive(*domain);
    archive(sources);
    archive(stepState);
    archive(boundaryNodes, communicatingNodes);
    archive(allBoundingBox, overlapDomains, isFacetNeighbor, facetToDomainPair);
    archive(narrowBandNodes, nodeState, nodeTimeReached, nodeLevel,
            nodeImagePoint, nodeImageVelocity, zoneVelocity);
    archive(minVelocity, maxVelocity, minEdgeSize, narrowBandWidth, dt, time,
            cycle, sourcesComplete, numNodesUnreached);
    archive(computeError);
    archive(nodeToDomainOffset, nodeToDomain, facetToDomainOffset, facetToDomain,
            facetNeighbors);
  }
};
};

#endif
