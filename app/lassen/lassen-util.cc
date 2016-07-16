#include "lassen-util.h"
#include <limits>
#include <unistd.h>
#include <sys/time.h>

namespace Lassen {

const Real LassenUtil::pico = 1e-12;
const Real LassenUtil::maxReal = std::numeric_limits<Real>::max();
const Real LassenUtil::minReal = std::numeric_limits<Real>::min();

const int LassenUtil::HexNumEdges = 12;
const int LassenUtil::HexEdgeToNode[12][2] = {{0, 1},
                                              {1, 2},
                                              {2, 3},
                                              {3, 0},
                                              {0, 4},
                                              {1, 5},
                                              {2, 6},
                                              {3, 7},
                                              {4, 5},
                                              {5, 6},
                                              {6, 7},
                                              {7, 4}};

const int LassenUtil::HexEdgeToFace[12][2] = {{4, 5},
                                              {0, 5},
                                              {1, 5},
                                              {3, 5},
                                              {3, 4},
                                              {4, 0},
                                              {0, 1},
                                              {1, 3},
                                              {2, 4},
                                              {2, 0},
                                              {2, 1},
                                              {2, 3}};

const int LassenUtil::HexFaceToEdge[6][4] = {{1, 6, 9, 5},
                                             {7, 10, 6, 2},
                                             {8, 9, 10, 11},
                                             {4, 11, 7, 3},
                                             {0, 5, 8, 4},
                                             {3, 2, 1, 0}};

Real LassenUtil::timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((tv.tv_sec) + (tv.tv_usec * 0.000001));
}
}
