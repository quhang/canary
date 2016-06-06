#include <canary/canary.h>

#include <glog/logging.h>

extern "C" {
void start() {
  LOG(INFO) << "Start";
  done();
}
}
