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
 * @file include/canary/canary_application.h
 * @author Hang Qu (quhang@cs.stanford.edu)
 * @brief Class CanaryApplication.
 */

#ifndef CANARY_INCLUDE_CANARY_CANARY_APPLICATION_H_
#define CANARY_INCLUDE_CANARY_CANARY_APPLICATION_H_

#include <map>
#include <string>

#include "canary/canary_internal.h"

namespace canary {

/**
 * Interface for data of a partition.
 */
class PartitionData {
 public:
  PartitionData() {}
  ~PartitionData() {}

  //! Clones a new empty instance.
  virtual PartitionData* Clone() const = 0;

  //! Initializes the data object.
  virtual void Initialize() = 0;
  //! Clears the memory.
  virtual void Finalize() = 0;

  //! Serializes the data.
  virtual void Serialize(CanaryOutputArchive& archive) const = 0;  // NOLINT
  //! Deserializes the data.
  virtual void Deserialize(CanaryInputArchive& archive) = 0;  // NOLINT

  //! Returns the data in an untyped pointer.
  void* get_data() { return data_; }

 protected:
  void* data_ = nullptr;
};

/**
 * Represents typed data of a partition.
 */
template <typename T>
class TypedPartitionData : public PartitionData {
 public:
  TypedPartitionData() {}
  ~TypedPartitionData() {}

  PartitionData* Clone() const override { return new TypedPartitionData<T>(); }
  void Initialize() override {
    if (!data_) data_ = new T();
  }
  void Finalize() override {
    delete get_typed_data();
    data_ = nullptr;
  }

  void Serialize(CanaryOutputArchive& archive) const {  // NOLINT
    if (data_) {
      archive(*get_typed_data());
    }
  }

  void Deserialize(CanaryInputArchive& archive) override {
    if (!data_) {
      data_ = new T();
    }
    archive(*get_typed_data());
  }

  T* get_typed_data() { return reinterpret_cast<T*>(data_); }
  const T* get_typed_data() const { return reinterpret_cast<const T*>(data_); }
};

// Forward declaration.
class CanaryTaskContext;

/**
 * A user writes a Canary application by inheriting the application class.
 */
class CanaryApplication {
 public:
  //! Variable access.
  enum class VariableAccess : int32_t { INVALID = -1, READ, WRITE };

  //! Statement type.
  enum class StatementType : int32_t {
    INVALID = -1,
    TRANSFORM,
    SCATTER,
    GATHER,
    LOOP,
    END_LOOP,
    WHILE,
    END_WHILE
  };

  //! A variable handle stores a variable id and its type.
  template <typename Type>
  class VariableHandle {
   public:
    //! Copy constructor is allowed.
    VariableHandle(const VariableHandle&) = default;
    //! Copy assignment is allowed.
    VariableHandle& operator=(const VariableHandle&) = default;
    //! Gets the variable id.
    VariableId get_variable_id() const { return variable_id_; }

   private:
    //! Only the application class can construct it.
    explicit VariableHandle(VariableId variable_id)
        : variable_id_(variable_id) {}
    const VariableId variable_id_;
    friend class CanaryApplication;
  };

  /*
   * External data structures describing the application.
   */

  //! Stores information about a variable.
  struct VariableInfo {
    PartitionData* data_prototype = nullptr;
    // -1 means unknown parallelism.
    int parallelism = -1;
  };

  typedef std::map<VariableId, VariableInfo> VariableInfoMap;

  //! Stores information about a statement.
  struct StatementInfo {
    StatementType statement_type = StatementType::INVALID;
    std::function<void(CanaryTaskContext*)> void_task_function;
    std::function<int(CanaryTaskContext*)> int_task_function;
    std::function<bool(CanaryTaskContext*)> bool_task_function;
    int num_loop = 0;
    std::map<VariableId, VariableAccess> variable_access_map;
    bool pause_needed = false;
    bool track_needed = false;
  };

  typedef std::map<StatementId, StatementInfo> StatementInfoMap;

 public:
  CanaryApplication() {}
  virtual ~CanaryApplication() {}
  NON_COPYABLE_AND_NON_MOVABLE(CanaryApplication);

  //! Declares a variable.
  template <typename Type>
  VariableHandle<Type> DeclareVariable(int parallelism = -1) {
    CHECK(parallelism == -1 || parallelism >= 1)
        << "Partitioning of a variable is a positive number.";
    const auto variable_id = AllocateVariableId();
    auto& variable_info = variable_info_map_[variable_id];
    variable_info.data_prototype = new TypedPartitionData<Type>();
    variable_info.parallelism = parallelism;
    return VariableHandle<Type>(variable_id);
  }

  /*
   * Declares the variable access and other attributes of a statement.
   */

  //! Reads a variable.
  template <typename Type>
  void ReadAccess(VariableHandle<Type> variable_handle) {
    ReadAccess(variable_handle.get_variable_id());
  }

  //! Reads a variable.
  void ReadAccess(VariableId variable_id) {
    auto iter = staged_buffer_access_.find(variable_id);
    if (iter == staged_buffer_access_.end()) {
      staged_buffer_access_[variable_id] = VariableAccess::READ;
    }
  }

  //! Writes a variable.
  template <typename Type>
  void WriteAccess(VariableHandle<Type> variable_handle) {
    WriteAccess(variable_handle.get_variable_id());
  }

  //! Writes a variable.
  void WriteAccess(VariableId variable_id) {
    auto iter = staged_buffer_access_.find(variable_id);
    if (iter == staged_buffer_access_.end()) {
      staged_buffer_access_[variable_id] = VariableAccess::WRITE;
    } else {
      // Write access overwrites read access.
      iter->second = VariableAccess::WRITE;
    }
    staged_has_writer_ = true;
  }

  //! Adds a pausing point.
  void PauseNeeded() { staged_pause_needed_ = true; }

  //! Adds a tracking pointer.
  void TrackNeeded() { staged_track_needed_ = true; }

  /*
   * Declares a statement.
   */

  void Transform(std::function<void(CanaryTaskContext*)> task_function) {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::TRANSFORM;
    statement.void_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Scatter(std::function<void(CanaryTaskContext*)> task_function) {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::SCATTER;
    statement.void_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  // Returns the number of more messages expected.
  void Gather(std::function<int(CanaryTaskContext*)> task_function) {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::GATHER;
    statement.int_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Loop(int loop) {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::LOOP;
    statement.num_loop = loop;
    ClearStagedState();
  }

  void EndLoop() {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_LOOP;
    ClearStagedState();
  }

  void While(std::function<bool(CanaryTaskContext*)> task_function) {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::WHILE;
    statement.bool_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void EndWhile() {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_WHILE;
    ClearStagedState();
  }

 public:
  /*
   * Global interface.
   */
  virtual void Program() = 0;
  virtual void LoadParameter(const std::string& parameter) = 0;
  virtual std::string SaveParameter() = 0;

  const VariableInfoMap* get_variable_info_map() { return &variable_info_map_; }
  const StatementInfoMap* get_statement_info_map() {
    return &statement_info_map_;
  }

 private:
  VariableId AllocateVariableId() { return next_variable_id_++; }
  VariableId next_variable_id_ = VariableId::FIRST;

  StatementId AllocateStatementId() { return next_statement_id_++; }
  StatementId next_statement_id_ = StatementId::FIRST;

  VariableInfoMap variable_info_map_;
  StatementInfoMap statement_info_map_;

  //! Applies staged states to a statement.
  void ApplyStagedState(StatementInfo* statement_info) {
    // Adds a dumb writing variable.
    if (!staged_has_writer_) {
      auto dumb_variable = DeclareVariable<bool>();
      WriteAccess(dumb_variable);
    }
    if (statement_info->statement_type == StatementType::WHILE) {
      for (auto pair : staged_buffer_access_) {
        auto& parallelism = variable_info_map_.at(pair.first).parallelism;
        if (parallelism == -1) {
          parallelism = 1;
        }
        CHECK_EQ(parallelism, 1)
            << "WHILE statement accesses multi-partitioning variable!";
      }
    }
    statement_info->variable_access_map = std::move(staged_buffer_access_);
    statement_info->pause_needed = staged_pause_needed_;
    statement_info->track_needed = staged_track_needed_;
  }

  //! Clears staged states.
  void ClearStagedState() {
    staged_has_writer_ = false;
    staged_buffer_access_.clear();
    staged_pause_needed_ = false;
    staged_track_needed_ = false;
  }

  //! Stores the metadata of the upcoming statement.
  bool staged_has_writer_ = false;
  std::map<VariableId, VariableAccess> staged_buffer_access_;
  bool staged_pause_needed_ = false;
  bool staged_track_needed_ = false;
};

}  // namespace canary

//! Registers an application.
#ifndef REGISTER_APPLICATION
#define REGISTER_APPLICATION(APPLICATION_CLASS)                    \
  extern "C" {                                                     \
  void* ApplicationEnterPoint() {                                  \
    static_assert(std::is_base_of<::canary::CanaryApplication,     \
                                  APPLICATION_CLASS>::value,       \
                  "The class did not inheret CanaryApplication!"); \
    return static_cast<void*>(new APPLICATION_CLASS());            \
  }                                                                \
  }
#endif  // REGISTER_APPLICATION

#endif  // CANARY_INCLUDE_CANARY_CANARY_APPLICATION_H_
