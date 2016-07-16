#ifndef SOURCE_H__
#define SOURCE_H__

#include "lassen.h"

namespace Lassen {

// forward declare
class Simulation;

class Source {
 public:
  virtual ~Source();
  virtual bool process(const Mesh *mesh, double timer, double narrowBandWidth,
                       std::vector<NodeState::Enum> &nodeState,
                       std::vector<Real> &nodeLevel,
                       std::vector<Real> &nodeTimeReached,
                       std::vector<Point> &nodeImagePoint,
                       std::vector<Real> &nodeImageVelocity) = 0;

  virtual void initialize(const Simulation *sim) = 0;

  virtual bool isComplete() const = 0;

  virtual std::ostream &operator<<(std::ostream &o) const = 0;
};

class PointSource : public Source {
 public:
  PointSource() {}

  PointSource(const Point &center, Real radius, Real delay)
      : center(center),
        radius(radius),
        delay(delay),
        minVelocity(0.0),
        done(false) {}

  void initialize(const Simulation *sim);

  bool process(const Mesh *mesh, double timer, double narrowBandWidth,
               std::vector<NodeState::Enum> &nodeState,
               std::vector<Real> &nodeLevel, std::vector<Real> &nodeTimeReached,
               std::vector<Point> &nodeImagePoint,
               std::vector<Real> &nodeImageVelocity);

  bool isComplete() const { return done; }

  std::ostream &operator<<(std::ostream &o) const {
    o << "PointSource : center = " << center << " radius = " << radius
      << " delay = " << delay << " minVelocity = " << minVelocity;
    return o;
  }

  const Point &getCenter() const { return center; }
  const Real &getRadius() const { return radius; }
  const Real &getDelay() const { return delay; }
  const Real &getMinVelocity() const { return minVelocity; }

 protected:
  Point center;
  Real radius;
  Real delay;
  Real minVelocity;
  bool done;

 public:
  template<class Archive> void serialize(Archive& archive) {
    archive(center, radius, delay, minVelocity, done);
  }
};

inline std::ostream &operator<<(std::ostream &o, const Source &p) {
  p << o;
  return o;
}
};

#endif
