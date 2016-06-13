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
 * @file src/shared/initialize.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class Initialize.
 */

#include "shared/initialize.h"

#include <event2/thread.h>
#include <sys/signal.h>

#include <string>

namespace {
void InitializeCanaryInternal(const std::string& help_message, int* argc,
                              char** argv[]) {
  ::gflags::SetUsageMessage(help_message);
  ::gflags::ParseCommandLineFlags(argc, argv, true);
  ::google::InitGoogleLogging((*argv)[0]);
  // Tells libevent to support threading.
  CHECK_EQ(evthread_use_pthreads(), 0);
  // Disables SIGPIPE signal, such that writing to a broken pipe can be handled
  // without requiring signal mechanism.
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

}  // namespace canary
