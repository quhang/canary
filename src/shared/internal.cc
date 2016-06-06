#include "shared/internal.h"

#include <event2/thread.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <sys/signal.h>

namespace {
void InitializeCanaryInternal(const std::string& help_message, int* argc,
                              char** argv[]) {
  ::gflags::SetUsageMessage(help_message);
  ::gflags::ParseCommandLineFlags(argc, argv, true);
  ::google::InitGoogleLogging((*argv)[0]);
  CHECK_EQ(evthread_use_pthreads(), 0);
  PCHECK(signal(SIGPIPE, SIG_IGN) != SIG_ERR);
}
}  // namespace

namespace canary {

void InitializeCanaryWorker(int* argc, char** argv[]) {
  InitializeCanaryInternal("Run a Canary worker process.", argc, argv);
}
void InitializeCanaryController(int* argc, char** argv[]) {
  InitializeCanaryInternal("Run a Canary controller process.", argc, argv);
}

}  // namespace
