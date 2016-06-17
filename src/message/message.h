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
  SHUT_DOWN_WORKER
};

//! Gets the category group of a message.
template <typename T>
MessageCategoryGroup get_message_category_group() {
  return MessageCategoryGroup::INVALID;
}

//! Gets the category group of a message.
template <typename T>
MessageCategoryGroup get_message_category_group(const T&) {
  return MessageCategoryGroup::INVALID;
}

//! Gets the category of a message.
template <typename T>
MessageCategory get_message_category() {
  return MessageCategory::INVALID;
}

//! Gets the category of a message.
template <typename T>
MessageCategory get_message_category(const T&) {
  return MessageCategory::INVALID;
}

//! Gets the type of a message.
template <MessageCategory>
class get_message_type {};

//! Registers a message type so that the above query interfaces work.
#define REGISTER_MESSAGE(GROUP_NAME, CATEGORY_NAME, TYPE_NAME)        \
  template <>                                                         \
  MessageCategoryGroup get_message_category_group<TYPE_NAME>() {      \
    return MessageCategoryGroup::GROUP_NAME;                          \
  }                                                                   \
  template <>                                                         \
  MessageCategoryGroup get_message_category_group(const TYPE_NAME&) { \
    return MessageCategoryGroup::GROUP_NAME;                          \
  }                                                                   \
  template <>                                                         \
  MessageCategory get_message_category<TYPE_NAME>() {                 \
    return MessageCategory::CATEGORY_NAME;                            \
  }                                                                   \
  template <>                                                         \
  MessageCategory get_message_category(const TYPE_NAME&) {            \
    return MessageCategory::CATEGORY_NAME;                            \
  }                                                                   \
  template <>                                                         \
  struct get_message_type<MessageCategory::CATEGORY_NAME> {           \
    typedef TYPE_NAME type;                                           \
  }

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_MESSAGE_H_
