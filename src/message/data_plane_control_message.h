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

#include "shared/internal.h"

#include "message/message.h"

#include <string>

namespace canary {
namespace message {

struct DataPlaneHeader {
  static const size_t kLength =
      sizeof(MessageLength) + sizeof(MessageCategoryGroup) +
      sizeof(MessageCategory);
  static const size_t kOffset0 = 0;
  static const size_t kOffset1 = kOffset0 + sizeof(MessageLength);
  static const size_t kOffset2 = kOffset1 + sizeof(MessageCategoryGroup);
  char content[kLength];
  MessageLength get_length() const {
    return *reinterpret_cast<MessageLength*>(header_content + offset0);
  }
  void set_length(MessageLength message_length) {
    memcpy(content + kOffset0, &message_length, sizeof(message_length));
  }
  MessageCategoryGroup get_category_group() const {
    return *reinterpret_cast<MessageCategoryGroup*>(header_content + offset1);
  }
  void set_category_group(MessageCategoryGroup message_category_group) {
    memcpy(content + kOffset1,
           &message_category_group, sizeof(message_category_group));
  }
  MessageCategory get_category() const {
    return *reinterpret_cast<MessageCategory*>(header_content + offset2);
  }
  void set_category(MessageCategory message_category) {
    memcpy(content + kOffset2,
           &message_category, sizeof(message_category));
  }
};

/*
 * These messages are exchanged between the controller and a worker to control
 * the underlying data plane of Canary, i.e. how data should be routed or
 * transmitted between workers, and how to manage worker membership when a
 * worker joins or leaves.
 */

/**
 * The controller assigns a WorkerId to a worker after a TCP connection is
 * built.
 */
struct AssignWorkerId {
  WorkerId assigned_worker_id;
  template<typename Archive> void serialize(Archive& archive) {
    archive(assigned_worker_id);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, ASSIGN_WORKER_ID, AssignWorkerId);

/**
 * A worker responds to the controller with its service ports.
 */
struct RegisterServicePort {
  WorkerId from_worker_id;
  std::string route_service, transmit_service;
  template<typename Archive> void serialize(Archive& archive) {
    archive(from_worker_id, route_service, transmit_service);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, REGISTER_SERVICE_PORT,
                 RegisterServicePort);

/**
 * The controller updates the partition map to a worker.
 */
struct UpdatePartitionMap {
  PartitionMapVersion version_id;
  std::map<FullPartitionId, WorkerId> partition_map;
  std::list<std::pair<FullPartitionId, WorkerId>> partition_map_update;
  template<typename Archive> void serialize(Archive& archive) {
    archive(version_id, partition_map, partition_map_update);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_PARTITION_MAP, UpdatePartitionMap);

/**
 * The controller updates the new worker member to a worker.
 */
struct UpdateAddedWorker {
  WorkerId added_worker_id;
  std::string route_service, transmit_service;
  template<typename Archive> void serialize(Archive& archive) {
    archive(added_worker_id, route_service, transmit_service);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, UPDATE_ADDED_WORKER, UpdateAddedWorker);

/**
 * A worker notifies the controller of losing connection with another worker.
 */
struct NotifyWorkerDisconnect {
  WorkerId from_worker_id;
  WorkerId disconnected_worker_id;
  template<typename Archive> void serialize(Archive& archive) {
    archive(from_worker_id, disconnected_worker_id);
  }
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, NOTIFY_WORKER_DISCONNECT,
                 NotifyWorkerDisconnect);

/**
 * The controller shuts down a worker.
 */
struct ShutDownWorker {
  template<typename Archive> void serialize(Archive&) {}
};
REGISTER_MESSAGE(DATA_PLANE_CONTROL, SHUT_DOWN_WORKER,
                 ShutDownWorker);

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_DATA_PLANE_CONTROL_MESSAGE_H_
