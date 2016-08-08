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
 * @file src/launcher/fair_share_launcher.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class FairShareLauncher.
 */

#include <unistd.h>

#include "shared/canary_internal.h"

#include "launcher/launcher_helper.h"
#include "shared/initialize.h"

DEFINE_int32(first_application, -1, "First application id.");
DEFINE_int32(second_application, -1, "Second application id.");
DEFINE_double(share_ratio, 1,
              "Resource usage ratio between the first applicaiton "
              "and the second.");
DEFINE_double(threshold, 0.1, "Threshold when priority tuning is triggered.");
DEFINE_int32(low_priority, 101, "Low priority value.");
DEFINE_int32(high_priority, 99, "High priority value.");
DEFINE_int32(interval, 5, "Measurement interval in seconds.");

namespace {
//! Changes an application's priority.
void ChangeApplicationPriority(int application_id, int priority_level) {
  using namespace canary;  // NOLINT
  LauncherHelper launch_helper;
  message::ControlApplicationPriority control_application;
  control_application.application_id = application_id;
  control_application.priority_level = priority_level;
  auto response =
      launch_helper
          .LaunchAndWaitResponse<message::ControlApplicationPriorityResponse>(
              control_application);
  CHECK(response.succeed) << response.error_message;
}
//! Queries the used cycles of an application.
double RequestApplicationCycles(int application_id) {
  using namespace canary;  // NOLINT
  LauncherHelper launch_helper;
  message::RequestApplicationStat report_application;
  report_application.application_id = application_id;
  auto response =
      launch_helper
          .LaunchAndWaitResponse<message::RequestApplicationStatResponse>(
              report_application);
  CHECK(response.succeed) << response.error_message;
  return response.cycles;
}
//! Tunes the priorities of the two applications, FLAG means whether the first
// application should have the higher priority.
void TunePriorities(bool flag) {
  if (flag) {
    ChangeApplicationPriority(FLAGS_first_application, FLAGS_high_priority);
    ChangeApplicationPriority(FLAGS_second_application, FLAGS_low_priority);
  } else {
    ChangeApplicationPriority(FLAGS_first_application, FLAGS_low_priority);
    ChangeApplicationPriority(FLAGS_second_application, FLAGS_high_priority);
  }
}
}  // namespace

int main(int argc, char** argv) {
  using namespace canary;  // NOLINT
  InitializeCanaryWorker(&argc, &argv);
  const auto start_time = time::Clock::now();
  auto end_time = start_time;
  const double first_application_start_cycles =
      RequestApplicationCycles(FLAGS_first_application);
  const double second_application_start_cycles =
      RequestApplicationCycles(FLAGS_second_application);
  bool prioritize_first_app = (FLAGS_share_ratio > 1);
  TunePriorities(prioritize_first_app);

  double first_application_used_cycles = 0;
  double second_application_used_cycles = 0;
  while (true) {
    PCHECK((int)sleep(FLAGS_interval) == 0);
    first_application_used_cycles =
        RequestApplicationCycles(FLAGS_first_application) -
        first_application_start_cycles;
    end_time = time::Clock::now();
    printf("A %f %f\n", time::duration_to_double(end_time - start_time),
           first_application_used_cycles);
    second_application_used_cycles =
        RequestApplicationCycles(FLAGS_second_application) -
        second_application_start_cycles;
    end_time = time::Clock::now();
    printf("B %f %f\n", time::duration_to_double(end_time - start_time),
           second_application_used_cycles);
    if (prioritize_first_app) {
      if (first_application_used_cycles / second_application_used_cycles >
          FLAGS_share_ratio * (1. + FLAGS_threshold)) {
        prioritize_first_app = false;
        end_time = time::Clock::now();
        printf("PrioritizeB %f\n",
               time::duration_to_double(end_time - start_time));
        TunePriorities(prioritize_first_app);
      }
    } else {
      if (first_application_used_cycles / second_application_used_cycles <
          FLAGS_share_ratio * (1. - FLAGS_threshold)) {
        prioritize_first_app = true;
        end_time = time::Clock::now();
        printf("PrioritizeA %f\n",
               time::duration_to_double(end_time - start_time));
        TunePriorities(prioritize_first_app);
      }
    }
    fflush(stdout);
  }
  return 0;
}
