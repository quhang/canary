#ifndef LASSENUTIL_H_
#define LASSENUTIL_H_

#include <cmath>
#include <assert.h>

namespace Lassen {

typedef double Real;
typedef long GlobalID;

class LassenUtil {
 public:
  /**
     Return the square of the distance between the points (x1,y1,z1) and
     (x2,y2,z2)
  */

  static inline Real distance2(Real x1, Real y1, Real z1, Real x2, Real y2,
                               Real z2) {
    Real dx = x1 - x2;
    Real dy = y1 - y2;
    Real dz = z1 - z2;
    return dx * dx + dy * dy + dz * dz;
  }

  /**
     Return the distance between the points (x1,y1,z1) and (x2,y2,z2)
  */

  static Real distance(Real x1, Real y1, Real z1, Real x2, Real y2, Real z2) {
    return LassenUtil::sqrt(distance2(x1, y1, z1, x2, y2, z2));
  }

  /**
     Sqaure x


  */
  static Real sqr(Real x) { return x * x; }

  /**
     Sqrt
  */

  static Real sqrt(Real x) { return std::sqrt(x); }

  /**
     Dot
  */
  static Real dot(Real x1, Real y1, Real x2, Real y2) {
    return x1 * x2 + y1 * y2;
  }

  /**
     Sign
  */
  static int sign(Real x) { return (x >= 0.0) ? 1 : -1; }

  static const Real pico;
  static const Real maxReal;
  static const Real minReal;

  // Connectivity
  static const int HexNumEdges;
  static const int HexEdgeToNode[12][2];
  static const int HexEdgeToFace[12][2];
  static const int HexFaceToEdge[6][4];

  // timer
  static Real timer();
};
};

#endif
