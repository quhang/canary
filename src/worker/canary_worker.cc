#include <shared/internal.h>

#include <dlfcn.h>
#include <glog/logging.h>

#include <canary/canary.h>

DEFINE_string(app, "/", "The application library.");

void done() { LOG(INFO) << "Application is done."; }

int main(int argc, char* argv[]) {
  ::canary::InitializeCanaryWorker(&argc, &argv);

  dlerror();  // Clear error code.
  void* handle = dlopen(FLAGS_app.c_str(), RTLD_LAZY);
  const char* err = reinterpret_cast<const char*>(dlerror());
  if (err) {
    LOG(FATAL) << err;
  }
  LOG(INFO) << "After open";

  dlerror();  // Clear error code.
  typedef void (*ApplicationEntry)();
  ApplicationEntry app_entry =
      reinterpret_cast<ApplicationEntry>(dlsym(handle, "start"));
  err = reinterpret_cast<const char*>(dlerror());
  if (err) {
    LOG(FATAL) << err;
  }
  LOG(INFO) << "After load";

  app_entry();

  LOG(INFO) << "After run";

  dlclose(handle);

  LOG(INFO) << "Done.";
  return 0;
}
