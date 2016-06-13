#include <shared/initialize.h>
int main(int argc, char* argv[]) {
  ::canary::InitializeCanaryController(&argc, &argv);
  return 0;
}
