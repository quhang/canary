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
 * @file src/message/data_plane_control_message.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Data plane control messages.
 */

#ifndef CANARY_SRC_MESSAGE_DATA_PLANE_CONTROL_MESSAGE_H_
#define CANARY_SRC_MESSAGE_DATA_PLANE_CONTROL_MESSAGE_H_

#include <map>
#include <utility>
#include <string>

#include "shared/internal.h"

#include "message/message.h"

namespace canary {
namespace message {

/*
 * These messages are exchanged between the controller and a worker to control
 * the underlying data plane of Canary. E.g. how data is routed between workers,
 * and how worker membership is managed when a worker joins or leaves.
 */

/**
 * The controller assigns a WorkerId to a worker after a TCP connection is
 * built.
 */
struct AssignWorkerId {
  WorkerId assigned_worker_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(assigned_worker_id);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, ASSIGN_WORKER_ID, AssignWorkerId);

/**
 * A worker responds to the controller with its service ports.
 */
struct RegisterServicePort {
  WorkerId from_worker_id;
  std::string route_service;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, route_service);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, REGISTER_SERVICE_PORT,
                 RegisterServicePort);

/**
 * The controller updates the partition map to a worker, by refreshing the
 * entire partition map.
 */
struct UpdatePartitionMapAndWorker {
  PartitionMapVersion version_id;
  PartitionMap* partition_map = nullptr;
  std::map<WorkerId, NetworkAddress> worker_addresses;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(version_id);
    if (partition_map == nullptr) {
      partition_map = new PartitionMap();
    }
    archive(*partition_map);
    archive(worker_addresses);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_PARTITION_MAP_AND_WORKER,
                 UpdatePartitionMapAndWorker);

/**
 * The controller updates the partition map to a worker, by adding an
 * application.
 */
struct UpdatePartitionMapAddApplication {
  PartitionMapVersion version_id;
  ApplicationId add_application_id;
  PerApplicationPartitionMap* per_application_partition_map = nullptr;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(version_id, add_application_id);
    if (per_application_partition_map == nullptr) {
      per_application_partition_map = new PerApplicationPartitionMap();
    }
    archive(*per_application_partition_map);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_PARTITION_MAP_ADD_APPLICATION,
                 UpdatePartitionMapAddApplication);

/**
 * The controller updates the partition map to a worker, by dropping an
 * application.
 */
struct UpdatePartitionMapDropApplication {
  PartitionMapVersion version_id;
  ApplicationId drop_application_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(version_id, drop_application_id);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_PARTITION_MAP_DROP_APPLICATION,
                 UpdatePartitionMapDropApplication);

/**
 * The controller updates the partition map to a worker, by giving incremental
 * updates.
 */
struct UpdatePartitionMapIncremental {
  PartitionMapVersion version_id;
  PartitionMapUpdate* partition_map_update = nullptr;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(version_id);
    if (partition_map_update == nullptr) {
      partition_map_update = new PartitionMapUpdate();
    }
    archive(*partition_map_update);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_PARTITION_MAP_INCREMENTAL,
                 UpdatePartitionMapIncremental);

/**
 * The controller updates the new worker member to a worker.
 */
struct UpdateAddedWorker {
  WorkerId added_worker_id;
  NetworkAddress network_address;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(added_worker_id, network_address);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_ADDED_WORKER, UpdateAddedWorker);

/**
 * A worker notifies the controller of losing connection with another worker.
 */
struct NotifyWorkerDisconnect {
  WorkerId from_worker_id;
  WorkerId disconnected_worker_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id, disconnected_worker_id);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, NOTIFY_WORKER_DISCONNECT,
                 NotifyWorkerDisconnect);

/**
 * The controller shuts down a worker.
 */
struct ShutDownWorker {
  template <typename Archive>
  void serialize(Archive&) {}  // NOLINT
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, SHUT_DOWN_WORKER, ShutDownWorker);

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_DATA_PLANE_CONTROL_MESSAGE_H_
