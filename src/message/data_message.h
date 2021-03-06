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
 * @file src/message/data_message.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Messages related to data.
 */

#ifndef CANARY_SRC_MESSAGE_DATA_MESSAGE_H_
#define CANARY_SRC_MESSAGE_DATA_MESSAGE_H_

#include "shared/canary_internal.h"

#include "message/message.h"

namespace canary {
namespace message {

struct DirectDataHandshake {
  WorkerId from_worker_id;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(from_worker_id);
  }
};
REGISTER_MESSAGE(APPLICATION_DATA_DIRECT, DIRECT_DATA_HANDSHAKE,
                 DirectDataHandshake);

struct DirectDataMigrate {
  ApplicationId application_id;
  VariableGroupId variable_group_id;
  PartitionId partition_id;
  RawEvbuffer raw_buffer;
  template <typename Archive>
  void serialize(Archive& archive) {  // NOLINT
    archive(application_id, variable_group_id, partition_id);
    archive(raw_buffer);
  }
};
REGISTER_MESSAGE(APPLICATION_DATA_DIRECT, DIRECT_DATA_MIGRATE,
                 DirectDataMigrate);

}  // namespace message
}  // namespace canary
#endif  // CANARY_SRC_MESSAGE_DATA_MESSAGE_H_
