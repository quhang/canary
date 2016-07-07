#include <cereal/archives/xml.hpp>
#include <sstream>
#include <string>

#include "shared/canary_internal.h"

#include "shared/initialize.h"
#include "shared/canary_application.h"
#include "message/launch_message.h"

DEFINE_string(launch_binary, "", "Binary location.");

int main(int argc, char** argv) {
  using namespace canary;  // NOLINT
  InitializeCanaryWorker(&argc, &argv);
  std::stringstream ss;
  {
    cereal::XMLOutputArchive archive(ss);
    for (int i = 1; i < argc; ++i) {
      std::string token(argv[i]);
      archive(token);
    }
  }
  message::LaunchApplication launch_application;
  launch_application.binary_location = FLAGS_launch_binary;
  launch_application.application_parameter = ss.str();

  void* handle = nullptr;
  auto application = CanaryApplication::LoadApplication(
      launch_application.binary_location,
      launch_application.application_parameter, &handle);
  LOG(INFO) << "Application:\n" << application->Print();
  CanaryApplication::UnloadApplication(handle, application);
  return 0;
}
