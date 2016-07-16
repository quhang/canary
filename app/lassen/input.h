#ifndef INPUT_H
#define INPUT_H

#include "lassen.h"

namespace Lassen {

class MeshConstructor {
 public:
  static void MakeRegularMesh(
      Mesh *mesh, int ndim,
      int *numGlobal,    // total number of zones in each dimension
      int *globalStart,  // global start index in each dimension
      int *numZones,     // number of zone to create in each dimension
      Real *zoneSize);   // size of each zone in each dimension

  // Create a single domain, which is part of a larger mesh
  static void MakeRegularDomain(
      Domain *domain, int ndim,
      int domainId,         // domainID of the domain to create
      int *numDomain,       // number of domains in each dimension
      int *numGlobalZones,  // number of global zones in each dimension
      Real *zoneSize);      // size of each zone in each dimension
};
};
#endif
