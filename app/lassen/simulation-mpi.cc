#include "simulation-mpi.h"
#include "source.h"

namespace Lassen {

SimulationMPI::SimulationMPI() : SimulationParallel() {}

void SimulationMPI::functionSendInitializeNodeCommunication(
    std::vector<GlobalID> &sendBuffer) const {
  sendBuffer.resize(communicatingNodes.size());
  nodeCommunicationCreateMsg(communicatingNodes, sendBuffer);
}

void SimulationMPI::functionRecvInitializeNodeCommunication(
    const std::vector<std::vector<GlobalID> > &recvBuffer) {
  // maps from local node index to neighbor domain.  This will be used to setup
  // the node communication
  std::vector<std::pair<int, int> > nodeToDomainPair;

  for (size_t i = 0; i < recvBuffer.size(); ++i) {
    int neighborIndex = static_cast<int>(i);
    nodeCommunicationProcessMsg(neighborIndex, recvBuffer[i], nodeToDomainPair);
  }

  nodeCommunicationComplete(nodeToDomainPair);
}

void SimulationMPI::functionSendFirstInitializeFacetCommunication(
    BoundingBox &boundingBox) const {
  computeBoundingBox(mesh->nodePoint, mesh->nLocalNodes, boundingBox);
}

void SimulationMPI::functionRecvFirstInitializeFacetCommunication(
    const std::vector<BoundingBox> inAllBoundingBox) {
  findBoundingBoxIntersections(domain->domainID, inAllBoundingBox,
                               this->getNarrowBandWidth(), overlapDomains);
}

void SimulationMPI::functionSendSecondInitializeFacetCommunication(
    std::vector<Point> &sendBuffer) const {
  for (size_t i = 0; i < boundaryNodes.size(); ++i) {
    int nodeIndex = boundaryNodes[i];
    sendBuffer[i] = mesh->nodePoint[nodeIndex];
  }
}

void SimulationMPI::functionRecvSecondInitializeFacetCommunication(
    const std::vector<std::vector<Point> > &recvBuffer) {
  std::vector<Point> zoneCenter(mesh->nzones);
  computeZoneCenters(mesh, zoneCenter);
  Real boxsize = this->getNarrowBandWidth();
  SpatialGrid grid(zoneCenter, boxsize);
  Real cutoffDistance = 2 * narrowBandWidth;
  facetNeighbors.clear();

  isFacetNeighbor.resize(overlapDomains.size(), 0);

  for (size_t i = 0; i < overlapDomains.size(); ++i) {
    const std::vector<Point> &boundaryPoint = recvBuffer[i];
    int neighborDomainID = overlapDomains[i];
    bool overlap = findFacetDomainOverlap(
        grid, zoneCenter, neighborDomainID, boundaryPoint,
        allBoundingBox[neighborDomainID], cutoffDistance, facetToDomainPair);
    if (overlap) {
      facetNeighbors.push_back(neighborDomainID);
      isFacetNeighbor[i] = 1;
    }
  }
}

void SimulationMPI::functionSendThirdInitializeFacetCommunication() const {}

void SimulationMPI::functionRecvThirdInitializeFacetCommunication(
    const std::vector<int> &recvIsFacetNeighbor) {
  for (int i = 0; i < static_cast<int>(overlapDomains.size()); ++i) {
    if (recvIsFacetNeighbor[i] == 1) {
      int neighborDomainID = overlapDomains[i];
      if (std::find(facetNeighbors.begin(), facetNeighbors.end(),
                    neighborDomainID) == facetNeighbors.end()) {
        facetNeighbors.push_back(neighborDomainID);
      }
    }
  }

  constructFacetToDomain(facetToDomainPair);
  // Canary: clear up data structures.
  allBoundingBox.clear();
  overlapDomains.clear();
  isFacetNeighbor.clear();
  facetToDomainPair.clear();
  boundaryNodes.clear();
  communicatingNodes.clear();
}

////////////////////////////////////////////////////////////////////////////////

void SimulationMPI::functionSendCommunicateFront(
    std::vector<std::vector<Real> > &sendBuffer) const {
  // send/recv the facets of the front.
  sendBuffer.resize(facetNeighbors.size());
  const PlanarFront &planarFront = stepState.planarFront;

  // determine which facets need to be sent, and to which domain.
  // FIXME: is it better to communicate the facet center, or to recompute from
  // the vertices?
  for (int i = 0; i < planarFront.nfacets; ++i) {
    int zoneIndex = planarFront.facetZone[i];
    int startDomain = facetToDomainOffset[zoneIndex];
    int endDomain = facetToDomainOffset[zoneIndex + 1];
    if (startDomain == endDomain) {
      continue;  // this is an entirely local facet
    }
    // Pack up the facet for communication:
    std::vector<Real> buffer;
    packFacet(planarFront, i, buffer);
    for (int j = startDomain; j < endDomain; ++j) {
      int receiverID = facetToDomain[j];
      sendBuffer[receiverID].insert(sendBuffer[receiverID].end(),
                                    buffer.begin(), buffer.end());
    }
  }
}

void SimulationMPI::functionRecvCommunicateFront(
    const std::vector<std::vector<Real> > &recvBuffer) {
  PlanarFront &planarFront = stepState.planarFront;
  for (size_t i = 0; i < recvBuffer.size(); ++i) {
    if (recvBuffer[i].size() == 0) {
      continue;
    }
    unpackFacet(recvBuffer[i], planarFront);
  }
}

void SimulationMPI::functionSendSynchronizeNodeData(
    std::vector<std::vector<NodeData> > &sendBuffer) const {
  sendBuffer.resize(domain->neighborDomains.size());
  gatherNodeData(sendBuffer);
}

void SimulationMPI::functionRecvSynchronizeNodeData(
    const std::vector<std::vector<NodeData> > &recvBuffer) {
  for (size_t i = 0; i < recvBuffer.size(); ++i) {
    accumulateNodeData(recvBuffer[i]);
  }
  // sort the narroband nodes
  std::sort(narrowBandNodes.begin(), narrowBandNodes.end());
}

////////////////////////////////////////////////////////////////////////////////
// pack facet
void SimulationMPI::packFacet(const PlanarFront &planarFront, int facetIndex,
                              std::vector<Real> &buffer) const {
  int vertexBegin = planarFront.facetToVertexOffset[facetIndex];
  int vertexEnd = planarFront.facetToVertexOffset[facetIndex + 1];
  int nVertex = vertexEnd - vertexBegin;
  buffer.push_back(static_cast<Real>(nVertex));
  for (int i = vertexBegin; i < vertexEnd; ++i) {
    const Point &point = planarFront.vertices[i];
    buffer.push_back(point.x);
    buffer.push_back(point.y);
    buffer.push_back(point.z);
  }
  const Point &center = planarFront.facetCenter[facetIndex];
  buffer.push_back(center.x);
  buffer.push_back(center.y);
  buffer.push_back(center.z);
  buffer.push_back(planarFront.facetVelocity[facetIndex]);
}

// unpack facet
void SimulationMPI::unpackFacet(const std::vector<Real> &buffer,
                                PlanarFront &planarFront) {
  size_t index = 0;

  while (index < buffer.size()) {
    int nVertex = buffer[index++];
    std::vector<Point> p(nVertex);
    for (int j = 0; j < nVertex; ++j) {
      Real x = buffer[index++];
      Real y = buffer[index++];
      Real z = buffer[index++];
      p[j] = Point(x, y, z);
    }
    Real x = buffer[index++];
    Real y = buffer[index++];
    Real z = buffer[index++];
    Point center(x, y, z);
    Real vel = buffer[index++];
    planarFront.addFacet(-1, nVertex, &p[0], center, vel);
  }
  assert(index == buffer.size());
}

};  // namespace
