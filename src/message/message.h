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

#include "shared/internal.h"

namespace canary {
namespace message {

//! The length of a message in bytes, excluding the message header.
typedef uint32_t MessageLength;

//! The category group of a message.
enum class MessageCategoryGroup : int16_t {
  INVALID = -1,
  DATA_PLANE_CONTROL,
  WORKER_COMMAND,
  CONTROLLER_COMMAND,
  ROUTE_DATA,
  TRANSMIT_DATA
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
  // Worker commands (from the controller).
  TEST_WORKER_COMMAND = 200,
  // Controller commands (from a worker).
  TEST_CONTROLLER_COMMAND = 300
};

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
    typedef TYPE_NAME type;                                                  \
  }

/**
 * The header used by the control plane.
 */
struct ControlHeader {
 private:
  typedef MessageLength Type0;
  typedef MessageCategoryGroup Type1;
  typedef MessageCategory Type2;

  //! The header data structure.
  struct Header {
    Type0 length;
    Type1 category_group;
    Type2 category;
  };

  //! Used for decoding.
  union {
    char content_[sizeof(Header)];
    Header header_;
  };

  static_assert(sizeof(Header) == sizeof(Type0) + sizeof(Type1) + sizeof(Type2),
                "Needs fixes here!");

 public:
  //! The length of the header.
  static const size_t kLength = sizeof(Header);

  //! Gets message length.
  Type0 get_length() const { return header_.length; }
  //! Sets message length.
  void set_length(Type0 length) { header_.length = length; }
  //! Gets message category group.
  Type1 get_category_group() const { return header_.category_group; }
  //! Sets message category group.
  void set_category_group(Type1 category_group) {
    header_.category_group = category_group;
  }
  //! Gets message category.
  Type2 get_category() const { return header_.category; }
  //! Sets message category.
  void set_category(Type2 category) { header_.category = category; }

  //! Packs a message into a buffer with header added.
  template <typename MessageType>
  static struct evbuffer* PackMessage(const MessageType& message) {
    struct evbuffer* buffer = evbuffer_new();
    CanaryOutputArchive archive(buffer);
    archive(message);
    ControlHeader header;
    header.set_length(evbuffer_get_length(buffer));
    header.set_category_group(get_message_category_group(message));
    header.set_category(get_message_category(message));
    evbuffer_prepend(buffer, header.content_, header.kLength);
    return buffer;
  }

  //! Unpacks and consumes a buffer into a message.
  template <MessageCategory category>
  static typename get_message_type<category>::type* UnpackMessage(
      struct evbuffer* buffer) {
    typedef typename get_message_type<category>::type MessageType;
    CHECK_EQ(evbuffer_drain(buffer, kLength), 0);
    CanaryInputArchive archive(buffer);
    auto message = new MessageType();
    archive(*message);
    evbuffer_free(buffer);
    return message;
  }

  //! Extracts the header from a buffer without changing the buffer.
  bool ExtractHeader(struct evbuffer* buffer) {
    ssize_t bytes = evbuffer_copyout(buffer, content_, kLength);
    return bytes >= 0 && static_cast<size_t>(bytes) == kLength;
  }

  //! Tries to drain out a message from the buffer.
  struct evbuffer* SegmentMessage(struct evbuffer* buffer) {
    if (!ExtractHeader(buffer)) {
      return nullptr;
    }
    const size_t full_length = kLength + get_length();
    // If the full message is received.
    if (full_length > evbuffer_get_length(buffer)) {
      return nullptr;
    }
    struct evbuffer* result = evbuffer_new();
    const int bytes = evbuffer_remove_buffer(buffer, result, full_length);
    CHECK_GE(bytes, 0);
    CHECK_EQ(static_cast<size_t>(bytes), full_length);
    return result;
  }
};

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_MESSAGE_H_
