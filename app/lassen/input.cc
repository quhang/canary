#include "input.h"

using namespace Lassen;

static void CreateMap(std::vector<int> &indexMap, int nitems[3],
                      int startitem[3], int enditem[3]) {
  indexMap.resize(nitems[0] * nitems[1] * nitems[2]);
  std::vector<int> mark(indexMap.size(), 0);

  int index = 0;
  for (int i = startitem[0]; i < enditem[0]; ++i) {
    for (int j = startitem[1]; j < enditem[1]; ++j) {
      for (int k = startitem[2]; k < enditem[2]; ++k) {
        int currentIndex = i * nitems[1] * nitems[2] + j * nitems[2] + k;
        indexMap[index++] = currentIndex;
        mark[currentIndex] = 1;
      }
    }
  }
  for (size_t i = 0; i < indexMap.size(); ++i) {
    if (mark[i] == 0) {
      indexMap[index++] = i;
    }
  }
}

static void RemapMesh(const Mesh *sourceMesh, const std::vector<int> &nodeMap,
                      const std::vector<int> &zoneMap, Mesh *targetMesh) {
  targetMesh->dim = sourceMesh->dim;

  std::vector<int> invNodeMap(nodeMap.size());
  for (size_t i = 0; i < nodeMap.size(); ++i) {
    invNodeMap[nodeMap[i]] = i;
  }
  std::vector<int> invZoneMap(zoneMap.size());
  for (size_t i = 0; i < zoneMap.size(); ++i) {
    invZoneMap[zoneMap[i]] = i;
  }

  // nodes connectivity
  int nnodes = sourceMesh->nnodes;
  targetMesh->nnodes = nnodes;
  targetMesh->nodeToZoneOffset.resize(sourceMesh->nodeToZoneOffset.size());
  targetMesh->nodeToZone.reserve(sourceMesh->nodeToZone.size());

  for (int i = 0; i < nnodes; ++i) {
    int nodeIndex = nodeMap[i];
    int zoneStart = sourceMesh->nodeToZoneOffset[nodeIndex];
    int zoneEnd = sourceMesh->nodeToZoneOffset[nodeIndex + 1];

    targetMesh->nodeToZoneOffset[i] = targetMesh->nodeToZone.size();
    for (int j = zoneStart; j < zoneEnd; ++j) {
      int sourceZoneIndex = sourceMesh->nodeToZone[j];
      int targetZoneIndex = invZoneMap[sourceZoneIndex];
      targetMesh->nodeToZone.push_back(targetZoneIndex);
    }
  }
  targetMesh->nodeToZoneOffset[nnodes] = targetMesh->nodeToZone.size();

  // node field
  targetMesh->nodePoint.resize(nnodes);
  targetMesh->nodeGlobalID.resize(nnodes);
  for (int i = 0; i < nnodes; ++i) {
    int nodeIndex = nodeMap[i];
    targetMesh->nodePoint[i] = sourceMesh->nodePoint[nodeIndex];
    targetMesh->nodeGlobalID[i] = sourceMesh->nodeGlobalID[nodeIndex];
  }

  // zone
  int nzones = sourceMesh->nzones;
  int zoneToNodeCount = sourceMesh->zoneToNodeCount;
  targetMesh->nzones = nzones;
  targetMesh->zoneToNodeCount = zoneToNodeCount;
  targetMesh->zoneToNode.resize(sourceMesh->zoneToNode.size());
  for (int i = 0; i < nzones; ++i) {
    int zoneIndex = zoneMap[i];
    int sourceStartNode = zoneToNodeCount * zoneIndex;
    for (int j = 0; j < zoneToNodeCount; ++j) {
      int sourceNodeIndex = sourceMesh->zoneToNode[sourceStartNode + j];
      int targetNodeIndex = invNodeMap[sourceNodeIndex];
      targetMesh->zoneToNode[i * zoneToNodeCount + j] = targetNodeIndex;
    }
  }
}

void MeshConstructor::MakeRegularMesh(Mesh *mesh, int ndim, int *numGlobal,
                                      int *globalStart, int *numZones,
                                      Real *zoneSize) {
  mesh->dim = ndim;

  // Initialize Nodes
  int numNodeI = numZones[0] + 1;
  int numNodeJ = numZones[1] + 1;
  int numNodeK = (mesh->dim == 2) ? 1 : (numZones[2] + 1);

  int numGlobalNodeJ = numGlobal[1] + 1;
  int numGlobalNodeK = (mesh->dim == 2) ? 1 : (numGlobal[2] + 1);

  int numZoneI = numZones[0];
  int numZoneJ = numZones[1];
  int numZoneK = (mesh->dim == 2) ? 1 : numZones[2];

  int globalStartI = globalStart[0];
  int globalStartJ = globalStart[1];
  int globalStartK = (mesh->dim == 2) ? 0 : globalStart[2];

  Real dx = zoneSize[0];
  Real dy = zoneSize[1];
  Real dz = (mesh->dim == 2) ? 0 : zoneSize[2];

  Real startx = globalStartI * dx;
  Real starty = globalStartJ * dy;
  Real startz = globalStartK * dz;

  mesh->nnodes = numNodeI * numNodeJ * numNodeK;
  mesh->nodePoint.resize(mesh->nnodes);
  mesh->nodeToZoneOffset.resize(mesh->nnodes + 1);
  mesh->nodeGlobalID.resize(mesh->nnodes);

  int nodeToZoneConnect[8][3] = {
      {-1, -1, -1},
      {-1, -1, 0},
      {-1, 0, -1},
      {-1, 0, 0},
      {0, -1, -1},
      {0, -1, 0},
      {0, 0, -1},
      {0, 0, 0},
  };

  size_t nodeIndex = 0;

  for (int i = 0; i < numNodeI; ++i) {
    for (int j = 0; j < numNodeJ; ++j) {
      for (int k = 0; k < numNodeK; ++k, ++nodeIndex) {
        mesh->nodeGlobalID[nodeIndex] =
            (globalStartI + i) * numGlobalNodeJ * numGlobalNodeK +
            (globalStartJ + j) * numGlobalNodeK + (globalStartK + k);

        mesh->nodePoint[nodeIndex].x = startx + i * dx;
        mesh->nodePoint[nodeIndex].y = starty + j * dy;
        mesh->nodePoint[nodeIndex].z = startz + k * dz;

        mesh->nodeToZoneOffset[nodeIndex] = mesh->nodeToZone.size();

        for (size_t c = 0; c < 8; ++c) {
          int zonei = i + nodeToZoneConnect[c][0];
          int zonej = j + nodeToZoneConnect[c][1];
          int zonek = k + nodeToZoneConnect[c][2];
          if ((zonei >= 0 && zonei < numZoneI) &&
              (zonej >= 0 && zonej < numZoneJ) &&
              (zonek >= 0 && zonek < numZoneK)) {
            int zoneIndex =
                (zonei * numZoneJ * numZoneK) + (zonej * numZoneK) + (zonek);

            mesh->nodeToZone.push_back(zoneIndex);
          }
        }
      }
    }
  }
  mesh->nodeToZoneOffset[mesh->nnodes] = mesh->nodeToZone.size();

  // Initialize Zones
  mesh->nzones = numZoneI * numZoneJ * numZoneK;

  // FIXME - determine the zoneToNodeCount automatically
  mesh->zoneToNodeCount = (mesh->dim == 2) ? 4 : 8;
  mesh->zoneToNode.resize(mesh->zoneToNodeCount * mesh->nzones);

  size_t zoneIndex = 0;

  int zoneToNodeConnect[8][3] = {{0, 0, 0},
                                 {0, 1, 0},
                                 {1, 1, 0},
                                 {1, 0, 0},
                                 {0, 0, 1},
                                 {0, 1, 1},
                                 {1, 1, 1},
                                 {1, 0, 1}};

  for (int i = 0; i < numZoneI; ++i) {
    for (int j = 0; j < numZoneJ; ++j) {
      for (int k = 0; k < numZoneK; ++k, ++zoneIndex) {
        size_t zoneToNodeOffset = zoneIndex * mesh->zoneToNodeCount;
        for (int c = 0; c < mesh->zoneToNodeCount; ++c) {
          int nodei = i + zoneToNodeConnect[c][0];
          int nodej = j + zoneToNodeConnect[c][1];
          int nodek = k + zoneToNodeConnect[c][2];
          int zoneNodeIndex =
              nodei * (numNodeJ * numNodeK) + nodej * numNodeK + nodek;
          mesh->zoneToNode[zoneToNodeOffset + c] = zoneNodeIndex;
        }
      }
    }
  }
}

void MeshConstructor::MakeRegularDomain(Domain *domain, int ndim, int domainID,
                                        int *numDomain, int *numGlobal,
                                        Real *zoneSize) {
  // domain level data
  int numDomainI = numDomain[0];
  int numDomainJ = numDomain[1];
  int numDomainK = (ndim == 2) ? 1 : numDomain[2];

  int numGlobalZoneI = numGlobal[0];
  int numGlobalZoneJ = numGlobal[1];
  int numGlobalZoneK = (ndim == 2) ? 1 : numGlobal[2];

  assert(numGlobalZoneI % numDomainI == 0);
  assert(numGlobalZoneJ % numDomainJ == 0);
  assert(numGlobalZoneK % numDomainK == 0);

  int domainIndexI = domainID / (numDomainJ * numDomainK);
  int domainIndexJ = (domainID % (numDomainJ * numDomainK)) / numDomainK;
  int domainIndexK = domainID % numDomainK;

  domain->domainID = domainID;
  domain->numDomains = numDomainI * numDomainJ * numDomainK;

  // neighbor domains
  for (int i = -1; i <= 1; ++i) {
    int neighborDomainI = domainIndexI + i;
    if (neighborDomainI < 0 || neighborDomainI >= numDomainI) {
      continue;
    }
    for (int j = -1; j <= 1; ++j) {
      int neighborDomainJ = domainIndexJ + j;
      if (neighborDomainJ < 0 || neighborDomainJ >= numDomainJ) {
        continue;
      }
      for (int k = -1; k <= 1; ++k) {
        int neighborDomainK = domainIndexK + k;
        if (neighborDomainK < 0 || neighborDomainK >= numDomainK) {
          continue;
        }
        int neighborDomainID = neighborDomainI * (numDomainJ * numDomainK) +
                               neighborDomainJ * numDomainK + neighborDomainK;

        if (neighborDomainID != domainID) {
          domain->neighborDomains.push_back(neighborDomainID);
        }
      }
    }
  }

  int numLocalZones[3] = {numGlobalZoneI / numDomainI,
                          numGlobalZoneJ / numDomainJ,
                          numGlobalZoneK / numDomainK};

  int startOwnedI = domainIndexI * numLocalZones[0];
  int startOwnedJ = domainIndexJ * numLocalZones[1];
  int startOwnedK = domainIndexK * numLocalZones[2];

  Mesh *mesh = domain->mesh;

  int startDomainI = startOwnedI;
  int endDomainI = startOwnedI + numLocalZones[0];
  int startDomainJ = startOwnedJ;
  int endDomainJ = startOwnedJ + numLocalZones[1];
  int startDomainK = startOwnedK;
  int endDomainK = startOwnedK + numLocalZones[2];

  if (domainIndexI > 0) startDomainI -= 1;
  if (domainIndexI < numDomainI - 1) endDomainI += 1;
  if (domainIndexJ > 0) startDomainJ -= 1;
  if (domainIndexJ < numDomainJ - 1) endDomainJ += 1;
  if (domainIndexK > 0) startDomainK -= 1;
  if (domainIndexK < numDomainK - 1) endDomainK += 1;

  int globalStart[3] = {startDomainI, startDomainJ, startDomainK};

  int numLocalZonesWithGhosts[3] = {endDomainI - startDomainI,
                                    endDomainJ - startDomainJ,
                                    endDomainK - startDomainK};
  Mesh temp;

  MakeRegularMesh(&temp, ndim, numGlobal, globalStart, numLocalZonesWithGhosts,
                  zoneSize);

  // PrintMesh(temp);

  // Create the node map to assign ghosts to the end.

  std::vector<int> nodeMap;

  int numLocalNodes[3];
  numLocalNodes[0] = numLocalZones[0] + 1;
  numLocalNodes[1] = numLocalZones[1] + 1;
  numLocalNodes[2] = ndim == 2 ? 1 : numLocalZones[2] + 1;

  int numLocalNodesWithGhosts[3];
  numLocalNodesWithGhosts[0] = numLocalZonesWithGhosts[0] + 1;
  numLocalNodesWithGhosts[1] = numLocalZonesWithGhosts[1] + 1;
  numLocalNodesWithGhosts[2] = ndim == 2 ? 1 : numLocalZonesWithGhosts[2] + 1;

  int startnode[3] = {startOwnedI - startDomainI, startOwnedJ - startDomainJ,
                      startOwnedK - startDomainK};

  int endnode[3] = {startOwnedI - startDomainI + numLocalNodes[0],
                    startOwnedJ - startDomainJ + numLocalNodes[1],
                    startOwnedK - startDomainK + numLocalNodes[2]};

  CreateMap(nodeMap, numLocalNodesWithGhosts, startnode, endnode);

  // Create the zone map to assign ghosts to the end.

  std::vector<int> zoneMap;

  int startzone[3] = {startOwnedI - startDomainI, startOwnedJ - startDomainJ,
                      startOwnedK - startDomainK};

  int endzone[3] = {startOwnedI - startDomainI + numLocalZones[0],
                    startOwnedJ - startDomainJ + numLocalZones[1],
                    startOwnedK - startDomainK + numLocalZones[2]};

  CreateMap(zoneMap, numLocalZonesWithGhosts, startzone, endzone);

  RemapMesh(&temp, nodeMap, zoneMap, mesh);

  mesh->nLocalZones = (numLocalZones[0] * numLocalZones[1] * numLocalZones[2]);
  mesh->nLocalNodes = (numLocalNodes[0] * numLocalNodes[1] * numLocalNodes[2]);

  domain->initializeGlobalToLocal();

  // PrintMesh(domain->mesh);
}
