// fixme - remove
#include <stdio.h>

#include "simulation-parallel.h"
#include "source.h"
#include <set>

namespace Lassen {

SimulationParallel::SimulationParallel() {}

////////////////////////////////////////////////////////////////////////////////
/**

 */

void SimulationParallel::initializeBoundaryNodes(
    std::vector<int> &boundaryNodes) const {
  const int zoneToNodeCount = mesh->zoneToNodeCount;

  // locate the boundary nodes:  intersection of local nodes and nodes on ghost
  // zones
  std::set<int> boundaryNodesSet;
  for (int i = mesh->nLocalZones; i < mesh->nzones; ++i) {
    int startNode = zoneToNodeCount * i;
    for (int j = 0; j < zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[j + startNode];
      if (nodeIndex < mesh->nLocalNodes) {
        boundaryNodesSet.insert(nodeIndex);
      }
    }
  }
  boundaryNodes.assign(boundaryNodesSet.begin(), boundaryNodesSet.end());
}

////////////////////////////////////////////////////////////////////////////////
/**

   initializeCommunicatingNodes

   This algorithm determines the collection of communicatingNodes.
   These are nodes that are shared by one or more neighbor domains.
   They may be local nodes, shared nodes, and ghost nodes.

 */

void SimulationParallel::initializeCommunicatingNodes(
    const std::vector<int> &boundaryNodes,
    std::vector<int> &communicatingNodes) const {
  const int zoneToNodeCount = mesh->zoneToNodeCount;

  // determine the nodes that need to be communicated with neighbors.
  // This includes the nodes that touch zones on either side of the
  // boundaryNodes
  std::set<int> communicatingNodesSet;
  for (size_t i = 0; i < boundaryNodes.size(); ++i) {
    int nodeIndex = boundaryNodes[i];
    int zoneBegin = mesh->nodeToZoneOffset[nodeIndex];
    int zoneEnd = mesh->nodeToZoneOffset[nodeIndex + 1];
    for (int j = zoneBegin; j < zoneEnd; ++j) {
      int zoneIndex = mesh->nodeToZone[j];
      int startCommNode = zoneToNodeCount * zoneIndex;
      for (int k = 0; k < zoneToNodeCount; ++k) {
        int commNodeIndex = mesh->zoneToNode[startCommNode + k];
        communicatingNodesSet.insert(commNodeIndex);
      }
    }
  }
  communicatingNodes.assign(communicatingNodesSet.begin(),
                            communicatingNodesSet.end());
}

////////////////////////////////////////////////////////////////////////////////

bool SimulationParallel::isNodeOwner(int nodeIndex) const {
  // PRECOND: assume the domains are sorted between domainStart and domainEnd
  int domainStart = nodeToDomainOffset[nodeIndex];
  int domainEnd = nodeToDomainOffset[nodeIndex + 1];
  if (domainStart == domainEnd) {
    return true;
  }
  int neighborIndex = nodeToDomain[domainStart];
  int neighborID = domain->neighborDomains[neighborIndex];
  if (domain->domainID < neighborID) {
    return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////

bool SimulationParallel::findFacetDomainOverlap(
    const SpatialGrid &grid, const std::vector<Point> &zoneCenter,
    const int overlapDomainID, const std::vector<Point> &neighborBoundaryPoints,
    const BoundingBox &neighborBoundingBox, const Real cutoffDistance,
    std::vector<std::pair<int, int> > &facetToDomainPair) {
  Real cutoffDistance2 = LassenUtil::sqr(cutoffDistance);

  int li, lj, lk;
  int hi, hj, hk;
  grid.gridLocation(neighborBoundingBox.min, li, lj, lk);
  grid.gridLocation(neighborBoundingBox.max, hi, hj, hk);
  bool toSend = false;
  for (int ii = li - 1; ii <= hi + 1; ++ii) {
    for (int jj = lj - 1; jj <= hj + 1; ++jj) {
      for (int kk = lk - 1; kk <= hk + 1; ++kk) {
        const std::vector<int> &indicies = grid.getIndicies(ii, jj, kk);
        for (size_t ll = 0; ll < indicies.size(); ++ll) {
          int zoneIndex = indicies[ll];
          for (size_t p = 0; p < neighborBoundaryPoints.size(); ++p) {
            const Point &nodePoint = neighborBoundaryPoints[p];
            if (zoneCenter[zoneIndex].distance2(nodePoint) < cutoffDistance2) {
              facetToDomainPair.push_back(
                  std::pair<int, int>(zoneIndex, overlapDomainID));
              toSend = true;
              break;
            }
          }
        }
      }
    }
  }
  return toSend;
}

////////////////////////////////////////////////////////////////////////////////
// FIXME: find a better name for this function.
void SimulationParallel::constructFacetToDomain(
    std::vector<std::pair<int, int> > &facetToDomainPair) {
  // temporary mapping from domainID to offset in the facetNeighbor index
  std::vector<int> facetNeighborMapping(domain->numDomains, -1);
  for (size_t i = 0; i < facetNeighbors.size(); ++i) {
    facetNeighborMapping[facetNeighbors[i]] = i;
  }

  // facetToDomainPair is sorted, such that all the zones are together.
  std::sort(facetToDomainPair.begin(), facetToDomainPair.end());

  // Copy the data from facetToDomainPair into a compressed storage
  facetToDomainOffset.resize(mesh->nzones + 1);
  facetToDomain.resize(facetToDomainPair.size());
  int index = 0;
  for (int i = 0; i < mesh->nzones; ++i) {
    facetToDomainOffset[i] = index;
    for (size_t j = index; j < facetToDomainPair.size(); ++j) {
      if (facetToDomainPair[j].first == i) {
        int domain = facetToDomainPair[j].second;
        facetToDomain[index] = facetNeighborMapping[domain];
        index++;
      } else {
        break;
      }
    }
  }
  facetToDomainOffset[mesh->nzones] = index;
}

////////////////////////////////////////////////////////////////////////////////
void SimulationParallel::nodeCommunicationCreateMsg(
    const std::vector<int> &communicatingNodes,
    std::vector<GlobalID> &sendBuffer) {
  sendBuffer.resize(communicatingNodes.size());
  for (size_t i = 0; i < communicatingNodes.size(); ++i) {
    int nodeIndex = communicatingNodes[i];
    sendBuffer[i] = mesh->nodeGlobalID[nodeIndex];
  }
}

////////////////////////////////////////////////////////////////////////////////

void SimulationParallel::nodeCommunicationProcessMsg(
    int neighborIndex, const std::vector<GlobalID> &buffer,
    std::vector<std::pair<int, int> > &nodeToDomainPair) {
  for (size_t i = 0; i < buffer.size(); ++i) {
    int nodeIndex = domain->getNodeIndex(buffer[i]);
    if (nodeIndex != -1) {
      nodeToDomainPair.push_back(std::pair<int, int>(nodeIndex, neighborIndex));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void SimulationParallel::nodeCommunicationComplete(
    std::vector<std::pair<int, int> > &nodeToDomainPair) {
  std::sort(nodeToDomainPair.begin(), nodeToDomainPair.end());

  // Copy the data from nodeToDomainPair into a compressed storage
  nodeToDomainOffset.resize(mesh->nnodes + 1);
  nodeToDomain.resize(nodeToDomainPair.size());
  int index = 0;
  for (int i = 0; i < mesh->nnodes; ++i) {
    nodeToDomainOffset[i] = index;
    for (size_t j = index; j < nodeToDomainPair.size(); ++j) {
      if (nodeToDomainPair[j].first == i) {
        nodeToDomain[index] = nodeToDomainPair[j].second;
        index++;
      } else {
        break;
      }
    }
  }
  nodeToDomainOffset[mesh->nnodes] = index;
}

////////////////////////////////////////////////////////////////////////////////
void SimulationParallel::gatherNodeData(
    std::vector<std::vector<NodeData> > &sendBuffer) {
  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    int startDomain = nodeToDomainOffset[nodeIndex];
    int endDomain = nodeToDomainOffset[nodeIndex + 1];
    if (startDomain == endDomain) {
      continue;  // this is an entirely local narrowband node
    }
    // Pack up the node for communication:
    NodeData nodeData(mesh->nodeGlobalID[nodeIndex], nodeLevel[nodeIndex],
                      nodeTimeReached[nodeIndex], nodeImageVelocity[nodeIndex],
                      nodeImagePoint[nodeIndex]);

    for (int j = startDomain; j < endDomain; ++j) {
      int receiverIndex = nodeToDomain[j];
      sendBuffer[receiverIndex].push_back(nodeData);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// FIXME - better name?
void SimulationParallel::accumulateNodeData(
    const std::vector<NodeData> &recvBuffer) {
  for (size_t i = 0; i < recvBuffer.size(); ++i) {
    const NodeData &nodeData = recvBuffer[i];

    int nodeIndex = domain->getNodeIndex(nodeData.globalID);
    assert(nodeIndex != -1);
    if (nodeState[nodeIndex] == NodeState::UNREACHED) {
      narrowBandNodes.push_back(nodeIndex);
      nodeState[nodeIndex] = NodeState::NARROWBAND;
      nodeLevel[nodeIndex] = nodeData.level;
      nodeTimeReached[nodeIndex] = nodeData.timeReached;
      nodeImagePoint[nodeIndex] = nodeData.imagePoint;
      nodeImageVelocity[nodeIndex] = nodeData.velocity;
    }
  }
}
};
