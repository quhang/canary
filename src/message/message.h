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
 * @file src/message/message.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Common message facilities.
 */

#ifndef CANARY_SRC_MESSAGE_MESSAGE_H_
#define CANARY_SRC_MESSAGE_MESSAGE_H_

#include <string>

#include "shared/canary_internal.h"

namespace canary {
namespace message {

/*
 * Basic types related to messages.
 */

//! The length of a message in bytes, excluding the message header.
typedef uint32_t MessageLength;

//! The category group of a message.
enum class MessageCategoryGroup : int16_t {
  INVALID = -1,
  DATA_PLANE_CONTROL = 0,
  LAUNCH_COMMAND,
  LAUNCH_RESPONSE_COMMAND,
  WORKER_COMMAND,
  CONTROLLER_COMMAND,
  APPLICATION_DATA_ROUTE,
  APPLICATION_DATA_DIRECT
};

//! The category of a message.
enum class MessageCategory : int16_t {
  INVALID = -1,
  // Data plane control.
  ASSIGN_WORKER_ID = 100,
  REGISTER_SERVICE_PORT,
  UPDATE_PARTITION_MAP_AND_WORKER,
  UPDATE_PARTITION_MAP_ADD_APPLICATION,
  UPDATE_PARTITION_MAP_DROP_APPLICATION,
  UPDATE_PARTITION_MAP_INCREMENTAL,
  UPDATE_ADDED_WORKER,
  NOTIFY_WORKER_DISCONNECT,
  SHUT_DOWN_WORKER,
  // Launch commands,
  LAUNCH_APPLICATION = 200,
  PAUSE_APPLICATION,
  RESUME_APPLICATION,
  CONTROL_APPLICATION_PRIORITY,
  REQUEST_APPLICATION_STAT,
  REQUEST_SHUTDOWN_WORKER,
  TRIGGER_SCHEDULING,
  // Launch response commands,
  LAUNCH_APPLICATION_RESPONSE = 300,
  PAUSE_APPLICATION_RESPONSE,
  RESUME_APPLICATION_RESPONSE,
  CONTROL_APPLICATION_PRIORITY_RESPONSE,
  REQUEST_APPLICATION_STAT_RESPONSE,
  REQUEST_SHUTDOWN_WORKER_RESPONSE,
  TRIGGER_SCHEDULING_RESPONSE,
  // Worker commands (from the controller).
  TEST_WORKER_COMMAND = 400,
  WORKER_LOAD_APPLICATION,
  WORKER_UNLOAD_APPLICATION,
  WORKER_LOAD_PARTITIONS,
  WORKER_UNLOAD_PARTITIONS,
  WORKER_MIGRATE_IN_PARTITIONS,
  WORKER_MIGRATE_OUT_PARTITIONS,
  WORKER_REPORT_STATUS_OF_PARTITIONS,
  WORKER_CHANGE_APPLICATION_PRIORITY,
  WORKER_PAUSE_EXECUTION,
  WORKER_INSTALL_BARRIER,
  WORKER_RELEASE_BARRIER,
  // Controller commands (from a worker).
  TEST_CONTROLLER_COMMAND = 500,
  CONTROLLER_RESPOND_MIGRATION_IN_PREPARED,
  CONTROLLER_RESPOND_MIGRATION_IN_DONE,
  CONTROLLER_RESPOND_MIGRATION_OUT_DONE,
  CONTROLLER_RESPOND_PARTITION_DONE,
  CONTROLLER_RESPOND_STATUS_OF_PARTITION,
  CONTROLLER_RESPOND_STATUS_OF_WORKER,
  CONTROLLER_RESPOND_PAUSE_EXECUTION,
  CONTROLLER_RESPOND_REACH_BARRIER,
  // Application data.
  ROUTE_DATA_UNICAST = 600,
  ROUTE_DATA_MULTICAST,
  DIRECT_DATA_MIGRATE = 700,
  DIRECT_DATA_STORAGE,
  DIRECT_DATA_HANDSHAKE
};

struct NetworkAddress {
  std::string host;
  std::string service;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(host, service);
  }
};

/*
 * Helper function for tranlating between message types and message category
 * lables.
 */

//! Gets the category group of a message.
template <typename T>
inline MessageCategoryGroup get_message_category_group() {
  return MessageCategoryGroup::INVALID;
}

//! Gets the category group of a message.
template <typename T>
inline MessageCategoryGroup get_message_category_group(const T&) {
  return MessageCategoryGroup::INVALID;
}

//! Gets the category of a message.
template <typename T>
inline MessageCategory get_message_category() {
  return MessageCategory::INVALID;
}

//! Gets the category of a message.
template <typename T>
inline MessageCategory get_message_category(const T&) {
  return MessageCategory::INVALID;
}

//! Gets the type of a message.
template <MessageCategory>
class get_message_type {};

//! Registers a message type so that the above query interfaces work.
#define REGISTER_MESSAGE(GROUP_NAME, CATEGORY_NAME, TYPE_NAME)               \
  template <>                                                                \
  inline MessageCategoryGroup get_message_category_group<TYPE_NAME>() {      \
    return MessageCategoryGroup::GROUP_NAME;                                 \
  }                                                                          \
  template <>                                                                \
  inline MessageCategoryGroup get_message_category_group(const TYPE_NAME&) { \
    return MessageCategoryGroup::GROUP_NAME;                                 \
  }                                                                          \
  template <>                                                                \
  inline MessageCategory get_message_category<TYPE_NAME>() {                 \
    return MessageCategory::CATEGORY_NAME;                                   \
  }                                                                          \
  template <>                                                                \
  inline MessageCategory get_message_category(const TYPE_NAME&) {            \
    return MessageCategory::CATEGORY_NAME;                                   \
  }                                                                          \
  template <>                                                                \
  struct get_message_type<MessageCategory::CATEGORY_NAME> {                  \
    typedef TYPE_NAME Type;                                                  \
  }

//! The trait class of message headers.
template <typename T>
class HeaderTrait {};

/**
 * The header data structure for control messages.
 */
struct ControlHeader {
  //! The message length.
  MessageLength length;
  //! The message category group.
  MessageCategoryGroup category_group;
  //! The message category.
  MessageCategory category;
  template <typename MessageType>
  void FillInMessageType() {
    category_group = get_message_category_group<MessageType>();
    category = get_message_category<MessageType>();
  }
  template <typename MessageType>
  void FillInMessageType(const MessageType&) {
    FillInMessageType<MessageType>();
  }
};

//! The header must be a POD class.
static_assert(std::is_pod<ControlHeader>::value, "Not POD type!");

template <>
struct HeaderTrait<ControlHeader> {
  typedef ControlHeader Type;
  static Type empty;
};

/**
 * The header data structure for data messages.
 */

struct DataHeader {
  //! The message length.
  MessageLength length;
  //! The message category group.
  MessageCategoryGroup category_group;
  //! The message category.
  MessageCategory category;
  //! Sequence number, and acknowledge sequence number.
  SequenceNumber sequence, ack_sequence;
  //! Header if the message is routed to a partition.
  PartitionMapVersion partition_map_version;
  ApplicationId to_application_id;
  VariableGroupId to_variable_group_id;
  PartitionId to_partition_id;
  StageId to_stage_id;

  template <typename MessageType>
  void FillInMessageType() {
    category_group = get_message_category_group<MessageType>();
    category = get_message_category<MessageType>();
  }
  template <typename MessageType>
  void FillInMessageType(const MessageType&) {
    FillInMessageType<MessageType>();
  }
};

//! The header must be a POD class.
static_assert(std::is_pod<DataHeader>::value, "Not POD type!");

template <>
struct HeaderTrait<DataHeader> {
  typedef DataHeader Type;
  static Type empty;
};

//! Examines the header of a buffer, which allows modifyication.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline T* ExamineHeader(struct evbuffer* buffer) {
  return reinterpret_cast<T*>(evbuffer_pullup(buffer, sizeof(T)));
}

// "const" also works because of its local linkage.
constexpr auto ExamineControlHeader = ExamineHeader<ControlHeader>;
constexpr auto ExamineDataHeader = ExamineHeader<DataHeader>;

//! Adds a header to a buffer, and returns the header.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline T* AddHeader(struct evbuffer* buffer) {
  CHECK_EQ(evbuffer_prepend(buffer, &HeaderTrait<T>::empty, sizeof(T)), 0);
  return ExamineHeader<T>(buffer);
}

constexpr auto AddControlHeader = AddHeader<ControlHeader>;
constexpr auto AddDataHeader = AddHeader<DataHeader>;

//! Strips the header of a buffer, and returns the header. The header must be
// deallocated after usage.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline T* StripHeader(struct evbuffer* buffer) {
  T* result = new T();
  const int bytes =
      evbuffer_remove(buffer, reinterpret_cast<char*>(result), sizeof(T));
  CHECK_EQ(bytes, static_cast<int>(sizeof(T)));
  return result;
}

constexpr auto StripControlHeader = StripHeader<ControlHeader>;
constexpr auto StripDataHeader = StripHeader<DataHeader>;

//! Removes the header of a buffer.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline void RemoveHeader(struct evbuffer* buffer) {
  CHECK_EQ(evbuffer_drain(buffer, sizeof(T)), 0);
}

constexpr auto RemoveControlHeader = RemoveHeader<ControlHeader>;
constexpr auto RemoveDataHeader = RemoveHeader<DataHeader>;

//! Serializes a messsage to a buffer, and returns the buffer.
template <typename MessageType>
inline struct evbuffer* SerializeMessage(const MessageType& message) {
  struct evbuffer* buffer = evbuffer_new();
  {
    CanaryOutputArchive archive(buffer);
    archive(message);
  }
  return buffer;
}

//! Serializes a messsage to a buffer, adds a control header, and returns the
// buffer.
template <typename MessageType>
inline struct evbuffer* SerializeMessageWithControlHeader(
    const MessageType& message) {
  struct evbuffer* buffer = SerializeMessage(message);
  // The length before adding the header.
  const auto length = evbuffer_get_length(buffer);
  auto header = AddHeader<ControlHeader>(buffer);
  header->length = length;
  header->FillInMessageType(message);
  return buffer;
}

//! Deserializes a message from a buffer, whose header has been stripped.
template <typename MessageType>
inline void DeserializeMessage(struct evbuffer* buffer, MessageType* message) {
  {
    CanaryInputArchive archive(buffer);
    archive(*message);
  }
  CHECK_EQ(evbuffer_get_length(buffer), 0u);
  evbuffer_free(buffer);
}

//! Tries to segment a message from the buffer, and returns the segmented
// message.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline struct evbuffer* SegmentMessage(struct evbuffer* buffer) {
  T* header = ExamineHeader<T>(buffer);
  if (header == nullptr) {
    return nullptr;
  }
  const auto total_length = header->length + sizeof(T);
  if (total_length > evbuffer_get_length(buffer)) {
    return nullptr;
  }
  struct evbuffer* result = evbuffer_new();
  const int bytes = evbuffer_remove_buffer(buffer, result, total_length);
  CHECK_GE(bytes, 0);
  CHECK_EQ(static_cast<size_t>(bytes), total_length);
  return result;
}

constexpr auto SegmentControlMessage = SegmentMessage<ControlHeader>;
constexpr auto SegmentDataMessage = SegmentMessage<DataHeader>;

//! Checks whether it is an integrate message.
template <typename T, typename = typename HeaderTrait<T>::Type>
inline bool CheckIsIntegrateMessage(struct evbuffer* buffer) {
  T* header = ExamineHeader<T>(buffer);
  if (header == nullptr) {
    return false;
  }
  const auto total_length = header->length + sizeof(T);
  return evbuffer_get_length(buffer) == total_length;
}

constexpr auto CheckIsIntegrateControlMessage =
    CheckIsIntegrateMessage<ControlHeader>;

constexpr auto CheckIsIntegrateDataMessage =
    CheckIsIntegrateMessage<DataHeader>;

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_MESSAGE_H_
