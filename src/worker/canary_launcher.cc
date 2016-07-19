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
 * @file src/worker/canary_launcher.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryLauncher.
 */

#include <algorithm>
#include <cereal/archives/xml.hpp>  // NOLINT
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "shared/canary_internal.h"

#include "message/launch_message.h"
#include "shared/initialize.h"
#include "shared/network.h"

DEFINE_string(launch_binary, "", "Application binary location.");
DEFINE_int32(launch_num_worker, -1, "Specify the number of worker.");
DEFINE_int32(launch_first_barrier, -1, "Specify the first barrier stage.");

namespace {
void SendLaunchMessage(struct evbuffer* buffer) {
  using namespace canary;  // NOLINT
  struct addrinfo hints;
  network::initialize_addrinfo(&hints, false);
  struct addrinfo* available_addresses = nullptr;
  const int errorcode =
      getaddrinfo(FLAGS_controller_host.c_str(), FLAGS_launch_service.c_str(),
                  &hints, &available_addresses);
  CHECK_EQ(errorcode, 0) << gai_strerror(errorcode);
  CHECK_NOTNULL(available_addresses);
  const int result_fd =
      socket(available_addresses->ai_family, available_addresses->ai_socktype,
             available_addresses->ai_protocol);
  PCHECK(result_fd >= 0);
  const int status = connect(result_fd, available_addresses->ai_addr,
                             available_addresses->ai_addrlen);
  freeaddrinfo(available_addresses);
  PCHECK(status == 0);
  auto length = evbuffer_get_length(buffer);
  ssize_t written_bytes =
      write(result_fd, evbuffer_pullup(buffer, length), length);
  PCHECK(written_bytes >= 0);
  CHECK_EQ(static_cast<size_t>(written_bytes), length)
      << "Networking buffer is too small.";
  evbuffer_free(buffer);
  network::close_socket(result_fd);
}
}  // namespace

int main(int argc, char** argv) {
  using namespace canary;  // NOLINT
  InitializeCanaryWorker(&argc, &argv);
  std::stringstream ss;
  {
    cereal::XMLOutputArchive archive(ss);
    for (int i = 1; i < argc; ++i) {
      std::string token(argv[i]);
      auto iter = std::find(token.begin(), token.end(), '=');
      CHECK(iter != token.end())
          << "Application argument needs to be specified as key=value";
      archive(cereal::make_nvp(std::string(token.begin(), iter),
                               std::string(std::next(iter), token.end())));
    }
  }
  message::LaunchApplication launch_application;
  launch_application.binary_location = FLAGS_launch_binary;
  launch_application.application_parameter = ss.str();
  launch_application.fix_num_worker = FLAGS_launch_num_worker;
  launch_application.first_barrier_stage = FLAGS_launch_first_barrier;
  SendLaunchMessage(
      message::SerializeMessageWithControlHeader(launch_application));
  return 0;
}
