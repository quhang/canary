#include "shared/canary_internal.h"

#include "shared/initialize.h"
#include "shared/resource_monitor.h"

int main(int argc, char* argv[]) {
  using namespace canary;
  InitializeCanaryWorker(&argc, &argv);
  ResourceMonitor monitor;
  monitor.Initialize();
  while (true) {
    sleep(6);
    LOG(INFO) << "All: " << monitor.get_all_cpu_usage_percentage() <<
        " Canary : " << monitor.get_canary_cpu_usage_percentage();
  }
  return 0;
}
