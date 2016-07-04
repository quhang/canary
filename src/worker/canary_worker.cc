/*
 * Copyright 2015 Stanford University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither the name of the copyright holders nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file src/worker/canary_worker.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryWorker.
 */

#include "shared/canary_internal.h"

#include "shared/initialize.h"
#include "worker/worker_communication_manager.h"
#include "worker/worker_scheduler.h"

int main(int argc, char** argv) {
  using namespace canary;  // NOLINT
  InitializeCanaryWorker(&argc, &argv);

  network::EventMainThread event_main_thread;
  WorkerCommunicationManager manager;
  WorkerScheduler scheduler;

  manager.Initialize(&event_main_thread,
                     &scheduler,  // command receiver.
                     &scheduler,  // data receiver.
                     FLAGS_controller_host, FLAGS_controller_service,
                     FLAGS_worker_service);
  scheduler.Initialize(&manager,                  // command sender.
                       manager.GetDataRouter());  // data sender.

  // The main thread runs both the manager and the scheduler.
  event_main_thread.Run();

  return 0;
}
