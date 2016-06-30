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
 * @file src/shared/internal_type.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Internal types.
 */

#ifndef CANARY_SRC_SHARED_INTERNAL_TYPE_H_
#define CANARY_SRC_SHARED_INTERNAL_TYPE_H_

#include "shared/internal_header.h"

/**
 * Makes a scoped enumerator countable.
 */
#define COUNTABLE_ENUM(T)                                         \
  inline constexpr std::underlying_type<T>::type get_value(T t) { \
    return static_cast<std::underlying_type<T>::type>(t);         \
  }                                                               \
  inline constexpr T get_next(const T& t, int inc = 1) {          \
    return static_cast<T>(get_value(t) + inc);                    \
  }                                                               \
  inline constexpr T get_prev(const T& t) {                       \
    return static_cast<T>(get_value(t) - 1);                      \
  }                                                               \
  inline T operator++(T & t) {                                    \
    t = get_next(t);                                              \
    return t;                                                     \
  }                                                               \
  inline T operator++(T & t, int) {                               \
    T result = t;                                                 \
    ++t;                                                          \
    return result;                                                \
  }                                                               \
  inline T operator--(T & t) {                                    \
    t = get_prev(t);                                              \
    return t;                                                     \
  }                                                               \
  inline T operator--(T & t, int) {                               \
    T result = t;                                                 \
    --t;                                                          \
    return result;                                                \
  }  // NOLINT

namespace canary {

/**
 * The id of a worker.
 */
enum class WorkerId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(WorkerId);

/**
 * The id of an application.
 */
enum class ApplicationId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(ApplicationId);

/**
 * The id of a variable group.
 */
enum class VariableGroupId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(VariableGroupId);

/**
 * The id of a variable.
 */
enum class VariableId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(VariableId);

/**
 * The id of a partition.
 */
enum class PartitionId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(PartitionId);

/**
 * The id of a stage.
 */
enum class StageId : int32_t { INIT = -2, INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(StageId);

/**
 * The version of the partition map.
 */
enum class PartitionMapVersion : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(PartitionMapVersion);

/**
 * The priority level.
 */
enum class PriorityLevel : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(PriorityLevel);

typedef uint64_t SequenceNumber;

struct FullPartitionId {
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, variable_group_id, partition_id);
  }
  bool operator<(const FullPartitionId& rhs) const {
    if (application_id < rhs.application_id) return true;
    if (application_id > rhs.application_id) return false;
    if (variable_group_id > rhs.variable_group_id) return true;
    if (variable_group_id < rhs.variable_group_id) return false;
    if (partition_id < rhs.partition_id) return true;
    return false;
  }
};

}  // namespace canary

namespace std {

//! Hash function for enumeration values.
template <typename T>
class hash {
  using sfinae = typename std::enable_if<std::is_enum<T>::value, T>::type;

 public:
  size_t operator()(const T& e) const {
    return std::hash<typename std::underlying_type<T>::type>()(
        canary::get_value(e));
  }
};

template <>
class hash<canary::FullPartitionId> {
 public:
  size_t operator()(const canary::FullPartitionId& e) const {
    return (std::hash<canary::ApplicationId>()(e.application_id) << 16) +
           (std::hash<canary::VariableGroupId>()(e.variable_group_id) << 8) +
           std::hash<canary::PartitionId>()(e.partition_id);
  }
};

}  // namespace std

#endif  // CANARY_SRC_SHARED_INTERNAL_TYPE_H_
