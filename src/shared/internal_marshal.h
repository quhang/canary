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
 * @file src/shared/internal_marshal.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Internal marshaling functionalities.
 */

#ifndef CANARY_SRC_SHARED_INTERNAL_MARSHAL_H_
#define CANARY_SRC_SHARED_INTERNAL_MARSHAL_H_

#include "shared/internal_header.h"

// Cereal library for serialization.
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

// Libevent buffer data structure.
#include "event2/buffer.h"

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
}  // namespace cereal

// Registers archives for polymorphic support.
CEREAL_REGISTER_ARCHIVE(cereal::CanaryOutputArchive);
CEREAL_REGISTER_ARCHIVE(cereal::CanaryInputArchive);

// Ties input and output archives together.
CEREAL_SETUP_ARCHIVE_TRAITS(cereal::CanaryInputArchive,
                            cereal::CanaryOutputArchive);

namespace canary {
// Imports names to Canary name space.
using cereal::CanaryInputArchive;
using cereal::CanaryOutputArchive;
}  // namespace canary

#endif  // CANARY_SRC_SHARED_INTERNAL_MARSHAL_H_
