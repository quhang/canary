/**
 * A zone is a cell.
 */
#include <fstream>
#include <exception>
#include "source.h"

#include <stdio.h>
#include <set>
#include <algorithm>
#include "simulation.h"
#include "lassen.h"
#include "source.h"
#include "input.h"

namespace Lassen {

Simulation::Simulation() : dt(0), time(0), cycle(0), sourcesComplete(false) {
  domain = new Domain();
  mesh = domain->mesh;
}

Simulation::~Simulation() { delete domain; }

void Simulation::initialize() {
  numNodesUnreached = mesh->nnodes;

  initializeVelocity();
  initializeMeshBasedQuantities();

  // initialize the sources
  for (size_t i = 0; i < sources.size(); ++i) {
    sources[i].initialize(this);
  }

  // initialize state

  // - initialize there are 0 narrow band nodes
  narrowBandNodes.resize(0);

  // - all the nodes are marked as unreached
  nodeState.assign(mesh->nnodes, NodeState::UNREACHED);

  // - node level is not meaningful unless at the narrowband,
  // - initialize to the max value.
  nodeLevel.assign(mesh->nnodes, LassenUtil::maxReal);

  // - node timeReached is initialized to maxReal
  nodeTimeReached.assign(mesh->nnodes, LassenUtil::maxReal);

  // - node image point
  nodeImagePoint.resize(mesh->nnodes);

  // - node image velocity
  nodeImageVelocity.assign(mesh->nnodes, 0.0);

  // - global simulation data
  this->sourcesComplete = false;
  this->cycle = 0;
  this->time = 0;
  this->dt = 0;
}

void Simulation::initializeVelocity() {
  minVelocity = LassenUtil::maxReal;
  maxVelocity = LassenUtil::minReal;

  for (int i = 0; i < mesh->nzones; ++i) {
    if (zoneVelocity[i] != 0.0) {
      minVelocity = std::min(minVelocity, zoneVelocity[i]);
      maxVelocity = std::max(maxVelocity, zoneVelocity[i]);
    }
  }
}

void Simulation::initializeMeshBasedQuantities() {
  // Compute the narrow band width:  max edge length + max diagonal of any zone
  // also compute min edge size
  narrowBandWidth = LassenUtil::minReal;
  minEdgeSize = LassenUtil::maxReal;
  int zoneToNodeCount = mesh->zoneToNodeCount;

  for (int i = 0; i < mesh->nzones; ++i) {
    int startNode = i * zoneToNodeCount;

    // loop over edges
    Real zoneMinEdgeSize = LassenUtil::maxReal;
    Real zoneMaxEdgeSize = LassenUtil::minReal;
    Real zoneMaxDiagonalSize = LassenUtil::minReal;

    // node to node
    for (int j = 0; j < zoneToNodeCount; ++j) {
      for (int k = 0; k < j; ++k) {
        int nodeA = mesh->zoneToNode[startNode + j];
        int nodeB = mesh->zoneToNode[startNode + k];
        Real distance2 =
            mesh->nodePoint[nodeA].distance2(mesh->nodePoint[nodeB]);
        zoneMaxDiagonalSize = std::max(zoneMaxDiagonalSize, distance2);
      }
    }
    // edges
    if (mesh->dim == 2) {
      for (int j = 0; j < 4; ++j) {
        int n1 = j;
        int n2 = (j + 1) % 4;
        int nodeA = mesh->zoneToNode[startNode + n1];
        int nodeB = mesh->zoneToNode[startNode + n2];
        Real distance2 =
            mesh->nodePoint[nodeA].distance2(mesh->nodePoint[nodeB]);
        zoneMaxEdgeSize = std::max(zoneMaxEdgeSize, distance2);
        zoneMinEdgeSize = std::min(zoneMinEdgeSize, distance2);
      }
    } else {
      for (int j = 0; j < LassenUtil::HexNumEdges; ++j) {
        int n1 = LassenUtil::HexEdgeToNode[j][0];
        int n2 = LassenUtil::HexEdgeToNode[j][1];
        int nodeA = mesh->zoneToNode[startNode + n1];
        int nodeB = mesh->zoneToNode[startNode + n2];
        Real distance2 =
            mesh->nodePoint[nodeA].distance2(mesh->nodePoint[nodeB]);
        zoneMaxEdgeSize = std::max(zoneMaxEdgeSize, distance2);
        zoneMinEdgeSize = std::min(zoneMinEdgeSize, distance2);
      }
    }
    narrowBandWidth =
        std::max(narrowBandWidth, LassenUtil::sqrt(zoneMaxEdgeSize) +
                                      LassenUtil::sqrt(zoneMaxDiagonalSize));

    minEdgeSize = std::min(minEdgeSize, LassenUtil::sqrt(zoneMinEdgeSize));
  }
}

////////////////////////////////////////////////////////////////////////////////
/**
   updateSources

 */

void Simulation::updateSources() {
  updateSources(mesh, time, narrowBandWidth, sources, narrowBandNodes,
                nodeState, nodeLevel, nodeTimeReached, nodeImagePoint,
                nodeImageVelocity, sourcesComplete, numNodesUnreached);
}

void Simulation::updateSources(
    const Mesh *mesh, Real time, Real narrowBandWidth,
    std::vector<PointSource> &sources, std::vector<int> &narrowBandNodes,
    std::vector<NodeState::Enum> &nodeState, std::vector<Real> &nodeLevel,
    std::vector<Real> &nodeTimeReached, std::vector<Point> &nodeImagePoint,
    std::vector<Real> &nodeImageVelocity, bool &sourcesComplete,
    int &numNodesUnreached) {
  // Check for completion of sources, return early
  if (sourcesComplete) {
    return;
  }

  // Process the sources
  bool processed = false;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (sources[i].isComplete()) {
    }
    // Initialize node states based on the source, return whether a new source
    // is triggered. Node state can be NARROWBAND or REACHED. --quhang
    bool ret =
        sources[i].process(mesh, time, narrowBandWidth, nodeState, nodeLevel,
                           nodeTimeReached, nodeImagePoint, nodeImageVelocity);
    processed = processed || ret;
  }

  // Update the narrowband nodes if one or more of the sources was processed
  // this cycle
  if (processed) {
    // This is an order O(nnodes) routine, but it only happens once per source.
    // In general, all algorithm that take place in the step should be based on
    // the narrowband.
    narrowBandNodes.resize(0);
    for (int i = 0; i < mesh->nnodes; ++i) {
      if (nodeState[i] == NodeState::NARROWBAND) {
        narrowBandNodes.push_back(i);
      }
      if (nodeState[i] == NodeState::REACHED) {
        numNodesUnreached -= 1;
        assert(numNodesUnreached >= 0);
      }
    }
  }

  // Determine if the sources are complete for next cycle
  sourcesComplete = true;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i].isComplete()) {
      sourcesComplete = false;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/**
   updateNarrowband

   @param[in] mesh
   @param[in]
   @param[out] surfaceFrontZone
*/

void Simulation::updateNarrowband() {
  updateNarrowband(mesh, minEdgeSize, nodeLevel, zoneVelocity, narrowBandNodes,
                   nodeState, stepState.surfaceFrontZone,
                   stepState.surfaceFrontNode, stepState.nodeTempArray,
                   numNodesUnreached);
}

void Simulation::updateNarrowband(
    const Mesh *mesh, const Real minEdgeSize,
    const std::vector<Real> &nodeLevel, const std::vector<Real> &zoneVelocity,
    std::vector<int> &narrowBandNodes, std::vector<NodeState::Enum> &nodeState,
    std::vector<int> &surfaceFrontZone, std::vector<int> &surfaceFrontNode,
    std::vector<int> &nodeTempArray, int &numNodesUnreached)

{
  // Set the number of surfaceFrontZone to 0.
  surfaceFrontZone.resize(0);
  surfaceFrontNode.resize(0);

  // return early if there are no narrowband nodes.
  if (narrowBandNodes.size() == 0) {
    return;
  }

  // Find zones that touch a narrowBandNodes
  std::set<int> zones;

  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    int startZone = mesh->nodeToZoneOffset[nodeIndex];
    int endZone = mesh->nodeToZoneOffset[nodeIndex + 1];
    for (int j = startZone; j < endZone; ++j) {
      int zoneIndex = mesh->nodeToZone[j];
      if (zoneVelocity[zoneIndex] > 0.0) {
        zones.insert(zoneIndex);
      }
    }
  }

  // Determine which of these zones are half-reached This means that
  // one or mode node has a positive level, and one or more has a
  // negative level.
  // Surface front refers to the interface. --quhang
  int zoneToNodeCount = mesh->zoneToNodeCount;

  for (std::set<int>::iterator it = zones.begin(); it != zones.end(); ++it) {
    int zoneIndex = *it;
    int numNodesReached = 0;
    int startNode = zoneIndex * zoneToNodeCount;

    for (int i = 0; i < mesh->zoneToNodeCount; ++i) {
      int nodeIndex = mesh->zoneToNode[startNode + i];
      if (nodeLevel[nodeIndex] < 0.0) {
        numNodesReached++;
      }
    }

    if (numNodesReached > 0 && numNodesReached < zoneToNodeCount) {
      if (zoneVelocity[zoneIndex] > 0.0) {
        surfaceFrontZone.push_back(zoneIndex);
      }
    }
  }

  // Update the narrowband nodes:
  //   Keep nodes if they have a positive nodeLevel OR if they touch a half
  //   reached zone

  // Initialize the nodeTempArray over the current narrowband nodes.
  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    nodeTempArray[nodeIndex] = 0;
  }
  // Determine which narrowband nodes touch a half reached zone
  for (size_t i = 0; i < surfaceFrontZone.size(); ++i) {
    int zoneIndex = surfaceFrontZone[i];
    int startNode = zoneIndex * zoneToNodeCount;
    for (int j = 0; j < zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[startNode + j];
      if (nodeState[nodeIndex] == NodeState::NARROWBAND) {
        nodeTempArray[nodeIndex] = 1;
      } else {
        // fixme -- make sure this always will hold
        // assert(0);
      }
    }
  }

  std::vector<int> newNarrowBandNode;
  newNarrowBandNode.reserve(narrowBandNodes.size());
  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    // mark the surface node
    if (nodeTempArray[nodeIndex] == 1) {
      surfaceFrontNode.push_back(nodeIndex);
    }

    // mark the reached nodes
    if (nodeTempArray[nodeIndex] == 0 && nodeLevel[nodeIndex] < 0.0) {
      if (nodeState[nodeIndex] != NodeState::REACHED) {
        nodeState[nodeIndex] = NodeState::REACHED;
        numNodesUnreached -= 1;
        assert(numNodesUnreached >= 0);
      }
    } else {
      // update the new narrowband nodes.
      newNarrowBandNode.push_back(nodeIndex);
    }
  }
  narrowBandNodes.swap(newNarrowBandNode);
}

////////////////////////////////////////////////////////////////////////////////
/**
   updateNearbyNodes.


*/

void Simulation::updateNearbyNodes() {
  updateNearbyNodes(mesh, stepState.surfaceFrontZone, nodeLevel, zoneVelocity,
                    stepState.nearbyNodes, nodeState);
}

void Simulation::updateNearbyNodes(const Mesh *mesh,
                                   const std::vector<int> &surfaceFrontZone,
                                   const std::vector<Real> &nodeLevel,
                                   const std::vector<Real> &zoneVelocity,
                                   std::vector<int> &nearbyNodes,
                                   std::vector<NodeState::Enum> &nodeState) {
  // Locate the nearby nodes.
  //  looks at the layer of zones around the half reach zones.
  //  Convert UNREACHED nodes on these neighbor zones to NEARBY nodes.
  std::set<int> nearbySet;
  const int zoneToNodeCount = mesh->zoneToNodeCount;
  for (size_t i = 0; i < surfaceFrontZone.size(); ++i) {
    int zoneIndex = surfaceFrontZone[i];
    int startNode = zoneIndex * zoneToNodeCount;
    for (int j = 0; j < zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[startNode + j];
      int startZone = mesh->nodeToZoneOffset[nodeIndex];
      int endZone = mesh->nodeToZoneOffset[nodeIndex + 1];
      for (int k = startZone; k < endZone; ++k) {
        int neighborZoneIndex = mesh->nodeToZone[k];
        if (zoneVelocity[neighborZoneIndex] > 0.0) {
          int neighborStartNode = neighborZoneIndex * zoneToNodeCount;

          for (int l = 0; l < zoneToNodeCount; ++l) {
            int neighborNodeIndex = mesh->zoneToNode[neighborStartNode + l];
            // Only include nearby nodes that are local to this domain
            // Ghost nodes will be picked up by other domains
            if (neighborNodeIndex >= mesh->nLocalNodes) {
              continue;
            }
            // if (nodeState[neighborNodeIndex] == NodeState::UNREACHED) {
            //  nodeState[neighborNodeIndex] = NodeState::NEARBY;
            //  nearbySet.insert( neighborNodeIndex );
            //}
            if (nodeState[neighborNodeIndex] != NodeState::NARROWBAND &&
                nodeLevel[neighborNodeIndex] >= 0.0) {
              nodeState[neighborNodeIndex] = NodeState::NEARBY;
              nearbySet.insert(neighborNodeIndex);
            }
          }
        }
      }
    }
  }
  nearbyNodes.assign(nearbySet.begin(), nearbySet.end());
}

////////////////////////////////////////////////////////////////////////////////
/**
   constructFront


*/

void Simulation::constructFront() {
  constructFront(mesh, nodeLevel, zoneVelocity, minEdgeSize,
                 stepState.surfaceFrontZone, stepState.planarFront);
}

void Simulation::constructFront(const Mesh *mesh,
                                const std::vector<Real> &nodeLevel,
                                const std::vector<Real> &zoneVelocity,
                                const Real minEdgeSize,
                                const std::vector<int> &surfaceFrontZone,
                                PlanarFront &facets) {
  // Loop throught the surface zones (half-reached) to construct the planar
  // front
  // For each surfaceFrontZone pull out the node positions, and node levels.
  // From this information, the facet corresponding that zone can be
  // constructed.

  for (size_t i = 0; i < surfaceFrontZone.size(); ++i) {
    int zoneIndex = surfaceFrontZone[i];
    Point positions[8];
    Real levels[8];
    int zoneToNodeCount = mesh->zoneToNodeCount;
    int startNode = zoneToNodeCount * zoneIndex;

    for (int j = 0; j < zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[startNode + j];
      positions[j] = mesh->nodePoint[nodeIndex];
      levels[j] = nodeLevel[nodeIndex];

      // if a level is small, we give it a small negative value for the facet
      // IMPROVE:  perhaps use a local measure instead of minEdgeSize
      if (std::abs(levels[j]) < LassenUtil::pico * minEdgeSize) {
        // BCM FIXME
        levels[j] = -LassenUtil::pico * minEdgeSize;
      }
    }

    if (mesh->dim == 2) {
      constructFacet2D(zoneIndex, zoneVelocity[zoneIndex], positions, levels,
                       facets);
    } else {
      constructFacet3D(zoneIndex, zoneVelocity[zoneIndex], positions, levels,
                       facets);
    }
  }
}

/**
   nodeLevel of each node of the cell
   position of each node of the cell
   minEdgeSize of the mesh is used to check against levels too close to the
   front
     - perhaps use a local measure

 */

void Simulation::constructFacet2D(int zoneIndex, Real zoneVelocity,
                                  const Point positions[4],
                                  const Real levels[4],
                                  PlanarFront &planarFront) {
  int edgeMarker[4];
  int facetCount = 0;
  Point facetVertex[4];
  int facetIndex = 0;

  // Mark the edges
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    // Determine if vertex i and j straddle the front
    // (i.e. their level's have opppsite signs)
    edgeMarker[i] = 0;
    if (levels[i] < 0 && levels[j] > 0) {
      facetCount += 1;
      edgeMarker[i] = -1;
    } else if (levels[i] > 0 && levels[j] < 0) {
      facetCount += 1;
      edgeMarker[i] = 1;
    }
  }

  // the direction of the walk makes the REACHED region on the left.
  // find the starting facet vertex

  /// FIXME:  Not sure if this is entirely correct for the 4 facetCount case.

  for (int i = 0; i < 4; ++i) {
    if (edgeMarker[i] == -1) {  // starting facetVertex on this edge
      int j = (i + 1) % 4;
      Real ratio = levels[i] / (levels[i] - levels[j]);
      facetVertex[facetIndex] = Point(positions[i], positions[j], ratio);
      facetIndex++;
    }
  }

  for (int i = 0; i < 4; ++i) {
    if (edgeMarker[i] == 1) {  // ending facetVertex on this edge
      int j = (i + 1) % 4;
      Real ratio = levels[i] / (levels[i] - levels[j]);
      facetVertex[facetIndex] = Point(positions[i], positions[j], ratio);
      facetIndex++;
    }
  }

  // Sanity check
  assert(facetIndex == facetCount);

  // Compute the center of the facet.
  if (facetCount > 0) {
    Point center;
    for (int i = 0; i < facetCount; ++i) {
      center.x += facetVertex[i].x;
      center.y += facetVertex[i].y;
      center.z += facetVertex[i].z;
    }
    center.x /= facetCount;
    center.y /= facetCount;
    center.z /= facetCount;

    planarFront.addFacet(zoneIndex, facetCount, facetVertex, center,
                         zoneVelocity);
  }
}

/**
   nodeLevel of each node of the cell
   position of each node of the cell
   minEdgeSize of the mesh is used to check against levels too close to the
   front
     - perhaps use a local measure

 */

void Simulation::constructFacet3D(int zoneIndex, Real zoneVelocity,
                                  const Point positions[8],
                                  const Real levels[8],
                                  PlanarFront &planarFront) {
  int edgeMarker[LassenUtil::HexNumEdges];
  Point facetVertex[12];
  int numEdgesToVisit = 0;

  // Initialize the edgeMarker.
  //   0 = the edge is not split by the surface
  //   1 = the edge is split by the surface, but not yet visited;
  //   2 = the edge is split by the surface, and has been visited;
  for (int i = 0; i < LassenUtil::HexNumEdges; ++i) {
    int nodeA = LassenUtil::HexEdgeToNode[i][0];
    int nodeB = LassenUtil::HexEdgeToNode[i][1];

    // If the levels of the two nodes have different signs
    if (levels[nodeA] * levels[nodeB] < 0) {
      edgeMarker[i] = 1;  // yellow
      numEdgesToVisit++;
    } else {
      edgeMarker[i] = 0;  // black
    }
  }

  // Each iteration of this while loop will create one facet.
  // The common case is to create one facet per zone.
  while (numEdgesToVisit > 0) {
    // find the edge to start (the first marked edge
    int edgeToStart = -1;
    for (int i = 0; i < LassenUtil::HexNumEdges; ++i) {
      if (edgeMarker[i] == 1) {
        edgeToStart = i;
        break;
      }
    }

    // Gather the points for the facet.
    int nextEdge = -1;
    int currentEdge = edgeToStart;
    int facetIndex = 0;

    while (nextEdge != edgeToStart) {
      int startingNode = LassenUtil::HexEdgeToNode[currentEdge][0];

      // pick a face of this edge
      int face = -1;
      if (levels[startingNode] > 0) {
        face = LassenUtil::HexEdgeToFace[currentEdge][0];
      } else {
        face = LassenUtil::HexEdgeToFace[currentEdge][1];
      }

      // find the edge index within the face, call this the startEdge
      int startEdge = -1;
      for (int i = 0; i < 4; ++i) {
        if (currentEdge == LassenUtil::HexFaceToEdge[face][i]) {
          startEdge = i;
          break;
        }
      }

      // beginning at the startEdge, interate clockwise on this face, looking
      // for the next marked edge
      for (int i = 1; i < 4; ++i) {
        // "i" starts from 1 to exclude the startEdge.
        int j = (startEdge - i + 4) % 4;
        int edgeOnFace = LassenUtil::HexFaceToEdge[face][j];
        if (edgeMarker[edgeOnFace]) {
          nextEdge = edgeOnFace;
          break;
        }
      }

      currentEdge = nextEdge;
      edgeMarker[currentEdge] = 2;
      numEdgesToVisit -= 1;

      int nodeA = LassenUtil::HexEdgeToNode[currentEdge][0];
      int nodeB = LassenUtil::HexEdgeToNode[currentEdge][1];
      Real ratio = levels[nodeA] / (levels[nodeA] - levels[nodeB]);
      facetVertex[facetIndex] =
          Point(positions[nodeA], positions[nodeB], ratio);
      facetIndex++;
    }

    // Create the facet
    if (facetIndex > 0) {
      Point center;
      for (int i = 0; i < facetIndex; ++i) {
        center.x += facetVertex[i].x;
        center.y += facetVertex[i].y;
        center.z += facetVertex[i].z;
      }
      center.x /= facetIndex;
      center.y /= facetIndex;
      center.z /= facetIndex;

      planarFront.addFacet(zoneIndex, facetIndex, facetVertex, center,
                           zoneVelocity);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/**
   constructDistancesToFront

 */

void Simulation::constructDistancesToFront() {
  constructDistancesToFront(mesh, stepState.planarFront, stepState.nearbyNodes,
                            narrowBandWidth, nodeState, nodeLevel,
                            narrowBandNodes, nodeImagePoint, nodeImageVelocity);
}

void Simulation::constructDistancesToFront(
    const Mesh *mesh, const PlanarFront &planarFront,
    const std::vector<int> &nearbyNodes, Real narrowBandWidth,
    std::vector<NodeState::Enum> &nodeState, std::vector<Real> &nodeLevel,
    std::vector<int> &narrowBandNodes, std::vector<Point> &nodeImagePoint,
    std::vector<Real> &nodeImageVelocity)

{
  // early return
  if (nearbyNodes.size() == 0) {
    return;
  }

  ///  cuttoff2, the factor of 2 is due to the origal coding where a
  ///  search matrix was used.

  Real simpleCutoff2 = LassenUtil::sqr(2 * narrowBandWidth);

  // Create a spatial search grid for the nearby nodes
  std::vector<Point> nearbyNodePoint(nearbyNodes.size());
  for (size_t i = 0; i < nearbyNodes.size(); ++i) {
    int nodeIndex = nearbyNodes[i];
    nearbyNodePoint[i] = mesh->nodePoint[nodeIndex];
  }

  // IMPROVE: the narrowBandWidth must be >= narrowBandWidth, however
  // larger values may improve performance (to a point), This is one
  // knob to turn to improve performance.
  SpatialGrid grid(nearbyNodePoint, narrowBandWidth * 2);

  // find all the facets that are near the narrowband nodes.
  // Find the image point of the nearby node on the facet.
  for (int j = 0; j < planarFront.nfacets; ++j) {
    const Point &facetCenter = planarFront.facetCenter[j];
    int fi, fj, fk;
    grid.gridLocation(facetCenter, fi, fj, fk);
    for (int ii = fi - 1; ii <= fi + 1; ++ii) {
      for (int jj = fj - 1; jj <= fj + 1; ++jj) {
        for (int kk = fk - 1; kk <= fk + 1; ++kk) {
          const std::vector<int> &indicies = grid.getIndicies(ii, jj, kk);
          for (size_t ll = 0; ll < indicies.size(); ++ll) {
            int nearbyNodeIndex = indicies[ll];
            int nodeIndex = nearbyNodes[nearbyNodeIndex];
            assert(nodeState[nodeIndex] == NodeState::NEARBY);

            const Point &nodePoint = mesh->nodePoint[nodeIndex];
            Real distance2 = facetCenter.distance2(nodePoint);
            if (distance2 < simpleCutoff2) {
              Point imagePoint;
              if (mesh->dim == 2) {
                imagePoint = planarFront.imagePointOnFacet2D(j, nodePoint);
              } else {
                imagePoint = planarFront.imagePointOnFacet3D(j, nodePoint);
              }
              Real level = imagePoint.distance(nodePoint);

              if (level < nodeLevel[nodeIndex]) {
                nodeLevel[nodeIndex] = level;
                nodeImagePoint[nodeIndex] = imagePoint;
                nodeImageVelocity[nodeIndex] = planarFront.facetVelocity[j];
              }
            }
          }
        }
      }
    }
  }

  // fixme - we can chekc here that progress was made.  at least on
  // nearby node should have a closer level.
}

////////////////////////////////////////////////////////////////////////////////

void Simulation::convertNearbyNodesToNarrowBandNodes() {
  std::vector<int> &nearbyNodes = stepState.nearbyNodes;

  /// Nearby nodes become narrowband nodes.
  for (size_t i = 0; i < nearbyNodes.size(); ++i) {
    int nodeIndex = nearbyNodes[i];
    assert(nodeState[nodeIndex] == NodeState::NEARBY);
    nodeState[nodeIndex] = NodeState::NARROWBAND;
    narrowBandNodes.push_back(nodeIndex);
  }

  // Keep the narrowBandNodes vector sorted
  std::sort(narrowBandNodes.begin(), narrowBandNodes.end());
}

////////////////////////////////////////////////////////////////////////////////
/**
   computeNextTimeStep

   The purpose of this function is to compute the next timestep.  The
   timestep is based on how long it should take the front to reach the
   first narrowBandNodes.

   @param[in]
   @param[in]

*/

void Simulation::computeNextTimeStep() {
  computeNextTimeStep(mesh, narrowBandNodes, nodeLevel, nodeImageVelocity,
                      stepState.surfaceFrontZone, minEdgeSize, maxVelocity,
                      nodeState, dt, stepState.nodeTempArray);
}

void Simulation::computeNextTimeStep(
    const Mesh *mesh, const std::vector<int> &narrowBandNodes,
    const std::vector<Real> &nodeLevel,
    const std::vector<Real> &nodeImageVelocity,
    const std::vector<int> &surfaceFrontZone, const Real minEdgeSize,
    const Real maxVelocity, const std::vector<NodeState::Enum> &nodeState,
    Real &dt, std::vector<int> &nodeTempArray)

{
  Real timeArrivalMin = LassenUtil::maxReal;
  const Real TIMESTEP_LIMITER = 0.5;
  const Real minLevelToConsider = 0.25 * minEdgeSize;
  // const Real minLevelToConsider = 0;
  const Real maxIncreaseFactor = 0.05;
  const Real dtlast = dt;

  // Find the minimum arrival time of the front to a narrowband node.
  // Base the next timestep on this minimum arrival time.

  //  . exclude nodes that touch a surface zone . too close to the front to
  //  matter
  //  . exclude ghost nodes (since they could be surface nodes of another
  //  domain)
  //  . exclude nodes that are spatially too close to the front
  //  (minLevelToConsider)

  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    nodeTempArray[nodeIndex] = 1;
  }

  // exclude nodes that touch the surfaceFrontZones
  int zoneToNodeCount = mesh->zoneToNodeCount;
  for (size_t i = 0; i < surfaceFrontZone.size(); ++i) {
    int zoneIndex = surfaceFrontZone[i];
    int startNode = zoneIndex * zoneToNodeCount;
    for (int j = startNode; j < startNode + zoneToNodeCount; ++j) {
      int nodeIndex = mesh->zoneToNode[j];
      if (nodeState[nodeIndex] == NodeState::NARROWBAND) {
        nodeTempArray[nodeIndex] = 0;
      }
    }
  }

  int nNarrowBandNode = narrowBandNodes.size();
  for (int i = 0; i < nNarrowBandNode; ++i) {
    int nodeIndex = narrowBandNodes[i];
    if (nodeTempArray[nodeIndex] == 0) {
      continue;
    }

    if (nodeIndex >= mesh->nLocalNodes) {
      // It could be the case that a ghost node is a surface node
      // on another domain.  If this is the case, then including
      // that node's contribution on this domain is incorrect,
      // since it could control the timestep lower that it would
      // otherwise be.
      continue;
    }
    if (nodeLevel[nodeIndex] <= minLevelToConsider) {
      continue;
    }
    Real velocity = nodeImageVelocity[nodeIndex];

    Real timeArrival = nodeLevel[nodeIndex] / velocity;
    if (timeArrival < timeArrivalMin) {
      timeArrivalMin = timeArrival;
    }
  }

  if (timeArrivalMin == LassenUtil::maxReal) {
    // If timeArrivalMin was not set, this means there are no
    // narrowBandNodes.  This case occurs usually at the beginning of
    // the problem before the sources start.  In this case, construct a
    // dt based on the minEdgeSize and maxVelocity.

    // FIXME: An alternative is to simply advance the time to the point
    // when the first source is processed.
    dt = (1.0 - .000001) * minEdgeSize / maxVelocity;
  } else {
    // In the normal case, a limiter is applied to reduce the
    // timestep for stability.
    dt = TIMESTEP_LIMITER * timeArrivalMin;
    dt = std::min((1.0 + maxIncreaseFactor) * dtlast, dt);
  }
}

////////////////////////////////////////////////////////////////////////////////
/**
   computeTimeIntegral
*/

void Simulation::computeTimeIntegral() {
  computeTimeIntegral(mesh, time, dt, narrowBandNodes, nodeLevel,
                      nodeTimeReached, nodeImageVelocity, nodeImagePoint);
}

void Simulation::computeTimeIntegral(const Mesh *mesh, Real time, Real dt,
                                     const std::vector<int> &narrowBandNodes,
                                     std::vector<Real> &nodeLevel,
                                     std::vector<Real> &nodeTimeReached,
                                     std::vector<Real> &nodeImageVelocity,
                                     std::vector<Point> &nodeImagePoint) {
  // Loop over the trusted nodes
  // compute the travel distance
  // update nodeLevel
  // update Image point

  for (size_t i = 0; i < narrowBandNodes.size(); ++i) {
    int nodeIndex = narrowBandNodes[i];
    Real velocity = nodeImageVelocity[nodeIndex];
    Real travelDistance = velocity * dt;
    Real currentLevel = nodeLevel[nodeIndex];
    Real newLevel = currentLevel - travelDistance;
    nodeLevel[nodeIndex] = newLevel;

    const Point &currentImagePoint = nodeImagePoint[nodeIndex];
    const Point &nodePoint = mesh->nodePoint[nodeIndex];
    Vector dir(currentImagePoint, nodePoint);
    dir.normalize();
    int sign = LassenUtil::sign(currentLevel);
    Point newImagePoint(currentImagePoint.x + sign * dir.x * travelDistance,
                        currentImagePoint.y + sign * dir.y * travelDistance,
                        currentImagePoint.z + sign * dir.z * travelDistance);
    nodeImagePoint[nodeIndex] = newImagePoint;

    // Check to see if the node is done
    if (newLevel < 0.0) {
      Real newNodeTimeReached = time + currentLevel / velocity;
      nodeTimeReached[nodeIndex] = newNodeTimeReached;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

bool Simulation::checkEndingCriteria() const {
  if (time == 0.0) {
    return false;
  }

  if (!sourcesComplete) {
    return false;
  }

  if (narrowBandNodes.size() > 0) {
    return false;
  }

  return true;
}

void Simulation::printBanner() const {
  if (domain->domainID == 0) {
    std::cout << "============================================================="
                 "===================\n";
    std::cout << "mesh:\n";
    std::cout << "  ndim   = " << mesh->dim << "\n";
    std::cout << "  nzones = " << mesh->nzones << "\n";
    std::cout << "  nnodes = " << mesh->nnodes << "\n";
    std::cout << "simulation:\n";
    std::cout << "  minVelocity = " << minVelocity << "\n";
    std::cout << "  maxVelocity = " << maxVelocity << "\n";
    std::cout << "  minEdgeSize = " << minEdgeSize << "\n";
    std::cout << "  narrowBandWidth = " << narrowBandWidth << "\n";
    std::cout << "sources:\n";
    for (size_t i = 0; i < sources.size(); ++i) {
      std::cout << "  " << (sources[i]) << "\n";
    }
    std::cout << "============================================================="
                 "===================\n";
  }
}

void Simulation::printProgress() const {
  int freq = 1000;
  freq = 1;
  if (domain->domainID == 0) {
    if (cycle % freq == 0) {
      std::cout << "step: cycle = " << this->cycle << " time = " << this->time
                << " dt = " << this->dt
                << " narrowBandNodes = " << narrowBandNodes.size() << "\n";
      std::cout.flush();
    }
  }
}

void Simulation::printFinal() const {
  if (domain->domainID == 0) {
    std::cout << "step: cycle = " << this->cycle << " time  = " << this->time
              << " dt = " << this->dt
              << " narrowBandNodes = " << narrowBandNodes.size() << "\n";
    std::cout.flush();
  }
}

// fixme?  Better place for this?
/////////////////////////////////////////////////////////////////////////////////
/**
   computeLocalErrors
   This routine computes errors under the following conditions.
     - there is one point source
     - all nodes are within line of sight of the point source
     - front velocities are constant
   In parallel, these values need to be combined to get the global answer
*/

void Simulation::ComputeErrors::computeLocalErrors(const Simulation *sim) {
  const Mesh *mesh = sim->getMesh();
  //const Real velocity = sim->getMinVelocity();

  const std::vector<Real> &nodeTimeReached = sim->getNodeTimeReached();

  // get the source
  // const std::vector<PointSource> &sources = sim->getSources();
  // assert(sources.size() == 1);
  // const PointSource &source = sources[0];
  // const Point center = source.getCenter();
  // const Real sourceRadius = source.getRadius();
  // assert(source.getDelay() == 0.0);

  error1 = 0.0;
  error2 = 0.0;
  errorMax = 0.0;
  errorNodeCount = 0;

  // for (int i = 0; i < mesh->nnodes; ++i) {
  //   if (!(sim->isNodeOwner(i))) {
  //     continue;
  //   }
  //   const Point &point = mesh->nodePoint[i];
  //   Real radius = point.distance(center);
  //   if (radius > sourceRadius) {
  //     Real exactTime = radius / velocity;
  //     Real diff = std::abs(nodeTimeReached[i] - exactTime);
  //     error2 += LassenUtil::sqr(diff);
  //     error1 += diff;
  //     errorMax = std::max(errorMax, diff);
  //     errorNodeCount++;
  //   }
  // }
  for (int i = 0; i < mesh->nnodes; ++i) {
    if (!(sim->isNodeOwner(i))) {
      continue;
    }
    if (nodeTimeReached[i] != LassenUtil::maxReal) {
      error2 += LassenUtil::sqr(nodeTimeReached[i]);
      error1 += nodeTimeReached[i];
      errorMax = std::max(errorMax, nodeTimeReached[i]);
      errorNodeCount++;
    }
  }
}

void Simulation::ComputeErrors::completeErrors() {
  error1 /= errorNodeCount;
  error2 /= errorNodeCount;
}

void Simulation::ComputeErrors::reportErrors() {
  std::cout << "==============================================================="
               "=================\n";
  std::cout << "nodeCount  = " << errorNodeCount << "\n";
  std::cout << "L1   error = " << error1 << "\n";
  std::cout << "L2   error = " << error2 << "\n";
  std::cout << "LInf error = " << errorMax << "\n";
  std::cout << "==============================================================="
               "=================\n";
}

void Simulation::Setup() {
  Point problemsize(1.0, 1.0, 1.0);
  double zoneSizes[3] = {problemsize.x / metadata.numGlobalZones[0],
    problemsize.y / metadata.numGlobalZones[1],
    problemsize.z / metadata.numGlobalZones[2]};
  const int dim = 3;
  // Modify sim->domain.
  MeshConstructor::MakeRegularDomain(
      domain, dim, metadata.rank,
      metadata.numDomains, metadata.numGlobalZones, zoneSizes);
  double sourceRad =
      2.0 * std::max(std::max(zoneSizes[0], zoneSizes[1]), zoneSizes[2]);
  // Modify sim->sources.
  ProblemSetup(*this, sourceRad);
}

////////////////////////////////////////////////////////////////////////////////
// Problem setup related functions
////////////////////////////////////////////////////////////////////////////////

bool ProblemSetup(Simulation &sim, Real sourceRad) {
  Domain *domain = sim.getDomain();
  std::vector<Real> zoneVelocity(domain->mesh->nzones, 1.0);
  sim.setZoneVelocity(zoneVelocity);
  const int interval = 10;
  for (float source_x = 0; source_x <= 1; source_x += interval * sourceRad)
    for (float source_y = 0;
         source_y <= 1 - source_x;
         source_y += interval * sourceRad) {
      float source_z = 1- source_x - source_y;
      sim.addSource(PointSource(Point(source_x, source_y, source_z),
                                sourceRad, 0));
    }
  return true;
}

}
