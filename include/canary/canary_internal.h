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
 * @file canary/canary_internal.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Canary top level header file.
 */

#ifndef CANARY_INCLUDE_CANARY_CANARY_INTERNAL_H_
#define CANARY_INCLUDE_CANARY_CANARY_INTERNAL_H_

// Gflags library for managing command line flags.
#include <gflags/gflags.h>

// Glog library for logging.
#include <glog/logging.h>

// Libevent networking buffer facility.
#include <event2/buffer.h>

// Cereal serialization library.
#include <cereal/cereal.hpp>

// Cereal serialization support for STL containers.
#include <cereal/types/array.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

// Fixed-width integer types.
#include <cinttypes>

// Time facility.
#include <chrono>

// C++ library related to function objects and type support.
#include <functional>
#include <type_traits>
#include <utility>

// Strings are used for printing debugging info.
#include <string>

/**
 * Marks a class as non-copyable and non-movable.
 */
#ifndef NON_COPYABLE_AND_NON_MOVABLE
#define NON_COPYABLE_AND_NON_MOVABLE(T) \
  T(const T&) = delete;                 \
  T(T&&) = delete;                      \
  T& operator=(const T&) = delete;      \
  T& operator=(T&&) = delete
#endif  // NON_COPYABLE_AND_NON_MOVABLE

/**
 * Command line flags.
 */
DECLARE_string(controller_host);
DECLARE_string(controller_service);
DECLARE_string(controller_log_dir);
DECLARE_string(controller_log_name);
DECLARE_string(launch_service);
DECLARE_string(worker_service);
DECLARE_int32(worker_threads);
DECLARE_bool(log_placement);
DECLARE_bool(use_recipe_engine);

namespace canary {
//! Wrapper for evbuffer, to be serialized/deserialized.
struct RawEvbuffer {
  RawEvbuffer() : buffer(nullptr) {}
  explicit RawEvbuffer(struct evbuffer* input) : buffer(input) {}
  struct evbuffer* buffer;
};
}  // namespace canary

namespace cereal {
/**
 * Canary serialization helper.
 */
class CanaryOutputArchive
    : public OutputArchive<CanaryOutputArchive, AllowEmptyClassElision> {
 public:
  explicit CanaryOutputArchive(struct evbuffer* output_buffer)
      : OutputArchive<CanaryOutputArchive, AllowEmptyClassElision>(this),
        internal_buffer_(CHECK_NOTNULL(output_buffer)) {}

  void saveBinary(const void* data, std::size_t size) {
    // TODO(quhang): this may cause performance problems due to fragmentation in
    // the buffer.
    const auto success = evbuffer_add(internal_buffer_, data, size);
    CHECK_EQ(success, 0) << "Failed to write " << size
                         << " bytes to output buffer!";
  }

  struct evbuffer* get_buffer() {
    return internal_buffer_;
  }

 private:
  struct evbuffer* internal_buffer_ = nullptr;
};

/**
 * Canary deserialization helper.
 */
class CanaryInputArchive
    : public InputArchive<CanaryInputArchive, AllowEmptyClassElision> {
 public:
  explicit CanaryInputArchive(struct evbuffer* input_buffer)
      : InputArchive<CanaryInputArchive, AllowEmptyClassElision>(this),
        internal_buffer_(CHECK_NOTNULL(input_buffer)) {}

  void loadBinary(void* const data, std::size_t size) {
    const auto read_size = evbuffer_remove(internal_buffer_, data, size);
    CHECK_EQ(read_size, static_cast<int>(size)) << "Failed to read " << size
                                                << " bytes from input buffer!";
  }

  struct evbuffer* get_buffer() {
    return internal_buffer_;
  }

 private:
  struct evbuffer* internal_buffer_ = nullptr;
};

// Common CanaryBinaryArchive serialization functions.

//! Saves for POD types.
template <class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
CEREAL_SAVE_FUNCTION_NAME(CanaryOutputArchive& ar, const T& t) {  // NOLINT
  ar.saveBinary(std::addressof(t), sizeof(t));
}

//! Loads for POD types.
template <class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
CEREAL_LOAD_FUNCTION_NAME(CanaryInputArchive& ar, T& t) {  // NOLINT
  ar.loadBinary(std::addressof(t), sizeof(t));
}

//! Serializes NVP types.
template <class Archive, class T>
inline CEREAL_ARCHIVE_RESTRICT(CanaryInputArchive, CanaryOutputArchive)
    CEREAL_SERIALIZE_FUNCTION_NAME(Archive& ar,            // NOLINT
                                   NameValuePair<T>& t) {  // NOLINT
  ar(t.value);
}

//! Serializes SizeTags types.
template <class Archive, class T>
inline CEREAL_ARCHIVE_RESTRICT(CanaryInputArchive, CanaryOutputArchive)
    CEREAL_SERIALIZE_FUNCTION_NAME(Archive& ar, SizeTag<T>& t) {  // NOLINT
  ar(t.size);
}

//! Saves binary data.
template <class T>
inline void CEREAL_SAVE_FUNCTION_NAME(CanaryOutputArchive& ar,    // NOLINT
                                      const BinaryData<T>& bd) {  // NOLINT
  ar.saveBinary(bd.data, static_cast<std::size_t>(bd.size));
}

//! Loads binary data
template <class T>
inline void CEREAL_LOAD_FUNCTION_NAME(CanaryInputArchive& ar,  // NOLINT
                                      BinaryData<T>& bd) {     // NOLINT
  ar.loadBinary(bd.data, static_cast<std::size_t>(bd.size));
}

//! Serialization will delete the buffer.
inline void CEREAL_SAVE_FUNCTION_NAME(
    CanaryOutputArchive& ar,  // NOLINT
    const struct canary::RawEvbuffer& buffer) {
  const bool has_buffer = (buffer.buffer != nullptr);
  ar(has_buffer);
  if (has_buffer) {
    const size_t length = evbuffer_get_length(buffer.buffer);
    ar(length);
    CHECK_EQ(evbuffer_add_buffer(ar.get_buffer(), buffer.buffer), 0);
    evbuffer_free(buffer.buffer);
  }
}

//! Deserialization will allocate the buffer.
inline void CEREAL_LOAD_FUNCTION_NAME(
    CanaryInputArchive& ar,                // NOLINT
    struct canary::RawEvbuffer& buffer) {  // NOLINT
  bool has_buffer;
  ar(has_buffer);
  if (!has_buffer) {
    buffer.buffer = nullptr;
  } else {
    size_t length;
    ar(length);
    buffer.buffer = evbuffer_new();
    CHECK_EQ(static_cast<int>(length),
             evbuffer_remove_buffer(ar.get_buffer(), buffer.buffer, length));
  }
}
}  // namespace cereal

// Registers archives for polymorphic support.
CEREAL_REGISTER_ARCHIVE(cereal::CanaryOutputArchive);
CEREAL_REGISTER_ARCHIVE(cereal::CanaryInputArchive);

// Ties input and output archives together.
CEREAL_SETUP_ARCHIVE_TRAITS(cereal::CanaryInputArchive,
                            cereal::CanaryOutputArchive);

namespace canary {

// Imports names to Canary namespace.
using cereal::CanaryInputArchive;
using cereal::CanaryOutputArchive;

}  // namespace canary

/**
 * Makes a scoped enumeration type countable.
 */
#define COUNTABLE_ENUM(T)                                                   \
  inline constexpr std::underlying_type<T>::type get_value(T t) {           \
    return static_cast<std::underlying_type<T>::type>(t);                   \
  }                                                                         \
  inline constexpr std::underlying_type<T>::type get_distance(T t1, T t2) { \
    return get_value(t2) - get_value(t1);                                   \
  }                                                                         \
  inline constexpr T get_next(const T& t, int inc = 1) {                    \
    return static_cast<T>(get_value(t) + inc);                              \
  }                                                                         \
  inline constexpr T get_prev(const T& t, int dec = 1) {                    \
    return static_cast<T>(get_value(t) - dec);                              \
  }                                                                         \
  inline T operator++(T& t) {                                               \
    t = get_next(t);                                                        \
    return t;                                                               \
  }                                                                         \
  inline T operator++(T& t, int) {                                          \
    T result = t;                                                           \
    ++t;                                                                    \
    return result;                                                          \
  }                                                                         \
  inline T operator--(T& t) {                                               \
    t = get_prev(t);                                                        \
    return t;                                                               \
  }                                                                         \
  inline T operator--(T& t, int) {                                          \
    T result = t;                                                           \
    --t;                                                                    \
    return result;                                                          \
  }  // NOLINT

namespace canary {

//! SFAINE version of the hash function.
template <typename T, bool>
struct enum_hash {
 private:
  enum_hash(enum_hash&&);
  ~enum_hash();
};

//! Hash function of enumeration types.
template <typename T>
struct enum_hash<T, true> {
  size_t operator()(const T& e) const noexcept {
    using value_type = typename std::underlying_type<T>::type;
    return std::hash<value_type>{}(static_cast<value_type>(e));
  }
};

/**
 * The id of a worker.
 */
enum class WorkerId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(WorkerId);

/**
 * The id of a launch command.
 */
enum class LaunchCommandId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(LaunchCommandId);

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

/*
 * The id of a recipe.
 */
enum class RecipeId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(RecipeId);

/*
 * The id of a recipe block.
 */
enum class RecipeBlockId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(RecipeBlockId);

/**
 * The id of a stage.
 */
enum class StageId : int32_t {
  // Represents special states.
  REACH_BARRIER = -32,
  COMPLETE,
  // Represents special commands.
  MIGRATE_IN = -16,
  MIGRATE_OUT,
  PAUSE_EXECUTION,
  INSTALL_BARRIER,
  RELEASE_BARRIER,
  CONTROL_FLOW_DECISION,
  INIT,
  // Normal stages.
  INITIALIZED_AND_UNKNOWN = -2,
  INVALID = -1,
  FIRST = 0
};
COUNTABLE_ENUM(StageId);

/**
 * The id of a statement.
 */
enum class StatementId : int32_t { INVALID = -1, FIRST = 0 };
COUNTABLE_ENUM(StatementId);

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

/**
 * Message sequence number.
 */
typedef uint64_t SequenceNumber;

/**
 * The full id of a partition.
 */
struct FullPartitionId {
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  std::string GetString() const {
    std::stringstream ss;
    ss << get_value(application_id) << "/" << get_value(variable_group_id)
       << "/" << get_value(partition_id);
    return ss.str();
  }
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, variable_group_id, partition_id);
  }
  bool operator<(const FullPartitionId& rhs) const {
    if (application_id < rhs.application_id) return true;
    if (application_id > rhs.application_id) return false;
    if (variable_group_id < rhs.variable_group_id) return true;
    if (variable_group_id > rhs.variable_group_id) return false;
    if (partition_id < rhs.partition_id) return true;
    return false;
  }
};

/**
 * The partial id of a partition.
 */
struct PartialPartitionId {
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(variable_group_id, partition_id);
  }
  bool operator<(const FullPartitionId& rhs) const {
    if (variable_group_id < rhs.variable_group_id) return true;
    if (variable_group_id > rhs.variable_group_id) return false;
    if (partition_id < rhs.partition_id) return true;
    return false;
  }
};

}  // namespace canary

namespace std {

//! Hash function for enumeration types. This is a bug in gcc-4.9, that the hash
// function for enumeration types is not implemented. Later version standard
// library fixes this, so this hash function can be removed.
template <typename T>
struct hash : canary::enum_hash<T, is_enum<T>::value> {};

//! Hash function for full partition id.
template <>
struct hash<::canary::FullPartitionId> {
  size_t operator()(const ::canary::FullPartitionId& e) const {
    return (std::hash<::canary::ApplicationId>()(e.application_id) << 16) +
           (std::hash<::canary::VariableGroupId>()(e.variable_group_id) << 8) +
           std::hash<::canary::PartitionId>()(e.partition_id);
  }
};

//! Hash function for partial partition id.
template <>
struct hash<::canary::PartialPartitionId> {
  size_t operator()(const ::canary::PartialPartitionId& e) const {
    return (std::hash<::canary::VariableGroupId>()(e.variable_group_id) << 8) +
           std::hash<::canary::PartitionId>()(e.partition_id);
  }
};

}  // namespace std

namespace canary {
namespace time {

/**
 * Use case:
 *   using namespace ::canary::time;
 *   Timepoint start = Clock::now();
 *   Timepoint end = Clock::now();
 *   Duration duration = end - start;
 *   std::cout << to_double(duration);
 */
typedef std::chrono::steady_clock Clock;
typedef std::chrono::system_clock WallClock;
//! Time point type.
typedef Clock::time_point Timepoint;
//! Time duration type.
typedef Clock::duration Duration;

//! Convert a time duration to double.
template <typename DurationType>
inline double duration_to_double(const DurationType& input) {
  return std::chrono::duration<double>(input).count();
}

template <typename TimepointType>
inline double timepoint_to_double(const TimepointType& time_point) {
  return duration_to_double(time_point.time_since_epoch());
}

}  // namespace time
}  // namespace canary
#endif  // CANARY_INCLUDE_CANARY_CANARY_INTERNAL_H_
