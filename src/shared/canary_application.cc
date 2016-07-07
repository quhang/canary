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
 * @file src/shared/canary_application.cc
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryApplication.
 */

#include "shared/canary_application.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <sstream>
#include <string>

namespace {
template <typename T>
bool ReasonId(T hint_id, T* result_id) {
  if (hint_id == T::INVALID) {
    return false;
  }
  if (*result_id == T::INVALID) {
    *result_id = hint_id;
    return true;
  }
  CHECK(hint_id == *result_id);
  return false;
}

template <>
bool ReasonId<int>(int hint_id, int* result_id) {
  if (hint_id == -1) {
    return false;
  }
  if (*result_id == -1) {
    *result_id = hint_id;
    return true;
  }
  CHECK_EQ(hint_id, *result_id);
  return false;
}

template <typename T>
std::string GetTypeName(T* input) {
  const char* real_name = typeid(*input).name();
  std::string result(abi::__cxa_demangle(real_name, 0, 0, nullptr));
  const std::string header("canary::TypedPartitionData");
  CHECK_EQ(result.substr(0, header.size()), header);
  result = result.substr(header.size() + 1);
  result.pop_back();
  return std::move(result);
}
}  // namespace

namespace canary {

CanaryApplication* CanaryApplication::LoadApplication(
    const std::string& binary_location,
    const std::string& application_parameter, void** handle_ptr) {
  // Loading the library.
  dlerror();  // Clears error code.
  *handle_ptr = dlopen(binary_location.c_str(), RTLD_NOW);
  const char* err = reinterpret_cast<const char*>(dlerror());
  if (err) {
    LOG(FATAL) << "Loading application error: " << err;
  }
  dlerror();  // Clears error code.

  // Loading the symbol.
  typedef void* (*FactoryMethod)();
  FactoryMethod entry_point = reinterpret_cast<FactoryMethod>(
      dlsym(*handle_ptr, "ApplicationEnterPoint"));
  err = reinterpret_cast<const char*>(dlerror());
  if (err) {
    LOG(FATAL) << "Loading application error: " << err;
  }
  auto loaded_application = reinterpret_cast<CanaryApplication*>(entry_point());

  // Instantiates the application object.
  loaded_application->LoadParameter(application_parameter);
  loaded_application->Program();
  loaded_application->FillInProgram();
  return loaded_application;
}

void CanaryApplication::UnloadApplication(void* handle,
                                          CanaryApplication* application) {
  delete application;
  dlclose(handle);
}

// Long function, but stays here to save adding functions to the external
// header.
void CanaryApplication::FillInProgram() {
  // Clears all the group id.
  for (auto& pair : variable_info_map_) {
    pair.second.variable_group_id = VariableGroupId::INVALID;
  }
  variable_group_info_map_.clear();
  for (auto& pair : statement_info_map_) {
    pair.second.variable_group_id = VariableGroupId::INVALID;
  }
  auto next_variable_group_id = VariableGroupId::FIRST;

  // Fills in group ids.
  bool progress_flag;
  do {
    progress_flag = false;
    // Fills in a group id for the first non-grouped variable.
    for (auto& pair : variable_info_map_) {
      if (pair.second.variable_group_id == VariableGroupId::INVALID) {
        pair.second.variable_group_id = next_variable_group_id++;
        progress_flag = true;
        break;
      }
    }
    // Exit if all variables are grouped.
    if (!progress_flag) {
      break;
    }
    // Propogates the newly filled in group id.
    do {
      progress_flag = false;
      for (auto& statement_pair : statement_info_map_) {
        auto& statement_info = statement_pair.second;
        auto variable_group_id = VariableGroupId::INVALID;
        for (const auto& pair : statement_info.variable_access_map) {
          ReasonId(variable_info_map_.at(pair.first).variable_group_id,
                   &variable_group_id);
        }
        if (ReasonId(variable_group_id, &statement_info.variable_group_id)) {
          // If the group id of a statement is updated.
          progress_flag = true;
          for (auto& pair : statement_info.variable_access_map) {
            ReasonId(variable_group_id,
                     &variable_info_map_.at(pair.first).variable_group_id);
          }
        }
      }
    } while (progress_flag);
  } while (true);

  // Clears the variable group info.
  for (auto group_id = VariableGroupId::FIRST;
       group_id < next_variable_group_id; ++group_id) {
    variable_group_info_map_[group_id].parallelism = -1;
    variable_group_info_map_[group_id].variable_id_set.clear();
  }

  // Fills in variable grouping.
  for (auto& pair : variable_info_map_) {
    CHECK(pair.second.variable_group_id != VariableGroupId::INVALID);
    variable_group_info_map_.at(pair.second.variable_group_id)
        .variable_id_set.insert(pair.first);
  }

  // Reasons about the parallelism of each group.
  for (auto& pair : variable_info_map_) {
    ReasonId(pair.second.parallelism,
             &variable_group_info_map_.at(pair.second.variable_group_id)
                  .parallelism);
  }
  for (auto& pair : statement_info_map_) {
    if (pair.second.variable_group_id != VariableGroupId::INVALID) {
      ReasonId(pair.second.parallelism,
               &variable_group_info_map_.at(pair.second.variable_group_id)
                    .parallelism);
    }
  }

  // Sanity checks that the parallelism is filled in correctly.
  for (auto& pair : variable_group_info_map_) {
    CHECK_GT(pair.second.parallelism, 0)
        << "Partitioning is not specified correctly!";
  }

  // Fills in parallelism metadata.
  for (auto& pair : variable_info_map_) {
    ReasonId(
        variable_group_info_map_.at(pair.second.variable_group_id).parallelism,
        &pair.second.parallelism);
  }
  for (auto& pair : statement_info_map_) {
    if (pair.second.variable_group_id != VariableGroupId::INVALID) {
      ReasonId(variable_group_info_map_.at(pair.second.variable_group_id)
                   .parallelism,
               &pair.second.parallelism);
    }
  }

  // Fills in statement metadata.
  for (auto& pair : statement_info_map_) {
    auto& statement_info = pair.second;
    switch (statement_info.statement_type) {
      case StatementType::SCATTER:
        statement_info.paired_gather_parallelism =
            statement_info_map_.at(get_next(pair.first)).parallelism;
        break;
      case StatementType::GATHER:
        statement_info.paired_scatter_parallelism =
            statement_info_map_.at(get_prev(pair.first)).parallelism;
        break;
      default:
        break;
    }
  }

  // Fills in statement metadata.
  for (auto& pair : statement_info_map_) {
    auto& statement_info = pair.second;
    if (statement_info.statement_type == StatementType::LOOP ||
        statement_info.statement_type == StatementType::WHILE) {
      auto statement_id = pair.first;
      int layer = 1;
      while (layer != 0) {
        ++statement_id;
        CHECK(statement_info_map_.find(statement_id) !=
              statement_info_map_.end())
            << "Loops unpaired!";
        switch (statement_info_map_.at(statement_id).statement_type) {
          case StatementType::LOOP:
          case StatementType::WHILE:
            ++layer;
            break;
          case StatementType::END_LOOP:
          case StatementType::END_WHILE:
            --layer;
            break;
          default:
            break;
        }
      }
      pair.second.branch_statement = get_next(statement_id);
      statement_info_map_.at(statement_id).branch_statement = pair.first;
    }
  }
}

std::string CanaryApplication::Print() const {
  std::stringstream ss;
  for (const auto& pair : variable_info_map_) {
    ss << "var#" << get_value(pair.first) << " group#"
       << get_value(pair.second.variable_group_id)
       << " partitioning=" << pair.second.parallelism
       << " type=" << GetTypeName(pair.second.data_prototype) << "\n";
  }

  for (const auto& pair : variable_group_info_map_) {
    ss << "group#" << get_value(pair.first)
       << " partitioning=" << pair.second.parallelism << "\n";
  }

  for (const auto& pair : statement_info_map_) {
    ss << "statement#" << get_value(pair.first) << " group#"
       << get_value(pair.second.variable_group_id);
    for (const auto& access_pair : pair.second.variable_access_map) {
      if (access_pair.second == VariableAccess::WRITE) {
        ss << " w(" << get_value(access_pair.first) << ")";
      } else {
        ss << " r(" << get_value(access_pair.first) << ")";
      }
    }
    switch (pair.second.statement_type) {
      case StatementType::TRANSFORM:
        ss << " transform(" << pair.second.parallelism << ")";
        break;
      case StatementType::SCATTER:
        ss << " scatter(" << pair.second.parallelism << "/"
           << pair.second.paired_gather_parallelism << ")";
        break;
      case StatementType::GATHER:
        ss << " gatter(" << pair.second.paired_scatter_parallelism << "/"
           << pair.second.parallelism << ")";
        break;
      case StatementType::LOOP:
        ss << " loop(" << pair.second.num_loop << "/"
           << get_value(pair.second.branch_statement) << ")";
        break;
      case StatementType::END_LOOP:
        ss << " end_loop(" << get_value(pair.second.branch_statement) << ")";
        break;
      case StatementType::WHILE:
        ss << " while(" << get_value(pair.second.branch_statement) << ")";
        break;
      case StatementType::END_WHILE:
        ss << " end_while(" << get_value(pair.second.branch_statement) << ")";
        break;
      default:
        LOG(FATAL) << "Intenal error!";
    }
    ss << "\n";
  }
  return ss.str();
}

}  // namespace canary
