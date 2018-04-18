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
#include <set>
#include <sstream>
#include <string>
#include "cereal/archives/xml.hpp"

#include "canary/canary_internal.h"

namespace canary {

/**
 * A simple rate limiter for testing.
 */
class RateLimiter {
 public:
  RateLimiter() {
    PCHECK(0 == pthread_spin_init(&internal_lock_, PTHREAD_PROCESS_PRIVATE));
  }
  virtual ~RateLimiter() {
    pthread_spin_destroy(&internal_lock_);
  }
  // limit_rate in Hz. -1 means non-blocking.
  void Initialize(double limit_rate) {
    pthread_spin_lock(&internal_lock_);
    started_ = false;
    interval_ = 1. / limit_rate;
    pthread_spin_unlock(&internal_lock_);
  }
  bool Join() {
    if (interval_ < 0) {
      return false;
    }
    pthread_spin_lock(&internal_lock_);
    if (started_) {
      time::Timepoint now_time;
      do {
        now_time = time::Clock::now();
      } while (time::duration_to_double(now_time - last_access_) < interval_);
      last_access_ = now_time;
    } else {
      started_ = true;
      last_access_ = time::Clock::now();
    }
    pthread_spin_unlock(&internal_lock_);
    return true;
  }

 private:
  bool started_ = false;
  time::Timepoint last_access_;
  // In seconds.
  double interval_ = 1;
  pthread_spinlock_t internal_lock_;
};

/**
 * Interface for data of a partition.
 */
class PartitionData {
 public:
  PartitionData() {}
  virtual ~PartitionData() {}

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
  virtual ~TypedPartitionData() {}

  PartitionData* Clone() const override { return new TypedPartitionData<T>(); }
  void Initialize() override {
    if (!data_) data_ = new T();
  }
  void Finalize() override {
    delete get_typed_data();
    data_ = nullptr;
  }

  void Serialize(CanaryOutputArchive& archive) const override {  // NOLINT
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
    FLUSH_BARRIER,  // wait till all previous stages on this partition are done
    BARRIER, // all previous stages complete, all subsequent stages wait
    UPDATE_PLACEMENT,
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
    //! Record partition for this variable handle.
    bool RecordInitialPartitionPlacement(
      CanaryApplication *app, const std::vector<int> &placement) {
      // Get variable info.
      const VariableInfoMap *info_map = app->get_variable_info_map();
      assert(info_map->find(variable_id_) != info_map->end());
      VariableInfo info = info_map->at(variable_id_);
      VariableGroupId group_id = info.variable_group_id;
      assert(group_id != VariableGroupId::INVALID);
      // Check if placement for variable group already exists. If yes, verify
      // that new placement matches previously supplied placement.
      if (int(placement.size()) != info.parallelism) {
        assert(false);
        return false;
      }
      VariableGroupPlacementMap *initial_group_placement =
        app->initial_variable_group_placement();
      if (initial_group_placement->find(group_id) !=
          initial_group_placement->end()) {
        const std::vector<int> &existing_placement =
          initial_group_placement->at(group_id);
        assert(existing_placement.size() == placement.size());
        for (size_t i = 0; i < placement.size(); ++i) {
          if (existing_placement[i] != placement[i]) {
            assert(false);
            return false;
          }  // if existing_placement[i] != placement[i]
        }  // for p in placement
      }  else {
        (*initial_group_placement)[group_id] = placement;
      }  // if else for group_id in group_placement
      return true;
    }  // RecordInitialPartitionPlacement

    void SetUpdatePlacement(CanaryApplication *app, bool update_placement) {
      VariableGroupId group_id =
        app->get_variable_info_map()->at(variable_id_).variable_group_id;
      app->set_update_placement_for_variable(variable_id_, update_placement);
      app->set_update_placement_for_variable_group(group_id, update_placement);
    }  // SetUpdatePlacement

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
    bool update_placement = false;
    // Partially filled in.
    int parallelism = -1;
    // Filled in.
    VariableGroupId variable_group_id = VariableGroupId::INVALID;
  };

  typedef std::map<VariableId, VariableInfo> VariableInfoMap;

  //! Stores information about a group.
  struct VariableGroupInfo {
    bool update_placement = false;
    // Filled in.
    std::set<VariableId> variable_id_set;
    int parallelism = -1;
  };
  typedef std::map<VariableGroupId, VariableGroupInfo> VariableGroupInfoMap;

  //! Stores information about a statement.
  struct StatementInfo {
    StatementType statement_type = StatementType::INVALID;
    std::function<void(CanaryTaskContext*)> void_task_function;
    std::function<int(CanaryTaskContext*)> int_task_function;
    std::function<bool(CanaryTaskContext*)> bool_task_function;
    std::string name;
    int num_loop = 0;
    std::map<VariableId, VariableAccess> variable_access_map;
    bool pause_needed = false;
    bool track_needed = false;

    // Start timer and include this statement in the time.
    bool compute_timer_start = false;
    // Include this statement in the time.
    bool compute_timer_active = false;

    // Filled in.
    VariableGroupId variable_group_id = VariableGroupId::INVALID;
    int parallelism = -1;
    StatementId branch_statement = StatementId::INVALID;
    int paired_scatter_parallelism = -1;
    int paired_gather_parallelism = -1;
  };

  typedef std::map<StatementId, StatementInfo> StatementInfoMap;

  typedef std::map<VariableGroupId, std::vector<int> > VariableGroupPlacementMap;

 public:
  //! Dynamically loads an application, returns nullptr if it cannot be loaded.
  static CanaryApplication* LoadApplication(
      const std::string& binary_location,
      const std::string& application_parameter, void** handle_ptr);
  //! Dynamically unloads an application.
  static void UnloadApplication(void* handle, CanaryApplication* application);

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

  // NOTE: By chinmayee
  //! Adds time start.
  void ComputeTimerStart() {
    compute_timer_started_ = true;
    compute_timer_active_ = true;
  }

  // NOTE: By chinmayee
  //! Adds time end.
  void ComputeTimerEnd() {
    compute_timer_active_ = false;
  }

  /*
   * Declares a statement.
   */

  void Transform(std::function<void(CanaryTaskContext*)> task_function,
                 std::string name = "unnamed_transform") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::TRANSFORM;
    statement.void_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Scatter(std::function<void(CanaryTaskContext*)> task_function,
               std::string name = "unnamed_scatter") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::SCATTER;
    statement.void_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  // Returns the number of more messages expected.
  void Gather(std::function<int(CanaryTaskContext*)> task_function,
              std::string name = "unnamed_gather") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::GATHER;
    statement.int_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  // Force all previous stages to complete before this runs (add them all
  // to before set).
  void FlushBarrier(std::function<void(CanaryTaskContext*)> task_function,
                    std::string name = "unnamed_flush_barrier") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::FLUSH_BARRIER;
    statement.void_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  // NOTE: By chinmayee
  // All previously spawned incomplete stages complete before this can run.
  // All subsequent stages wait for this to complete.
  void Barrier(std::function<void(CanaryTaskContext*)> task_function,
               std::string name = "unnamed_barrier") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::BARRIER;
    statement.void_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  // NOTE: By chinmayee
  // Force all previous stages to complete before running this. This introduces
  // a control dependency -- it waits till controller sends OK to proceed.
  void UpdatePlacement(std::function<void(CanaryTaskContext*)> task_function,
                       std::string name = "unnamed_update_placement") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::UPDATE_PLACEMENT;
    statement.void_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Loop(int loop, std::string name = "unnamed_loop") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::LOOP;
    statement.name = name;
    statement.num_loop = loop;
    ClearStagedState();
  }

  void EndLoop(std::string name = "unnamed_end_loop") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_LOOP;
    statement.name = name;
    ClearStagedState();
  }

  void While(std::function<bool(CanaryTaskContext*)> task_function,
             std::string name = "unnamed_while") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::WHILE;
    statement.bool_task_function = std::move(task_function);
    statement.name = name;
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void EndWhile(std::string name = "unnamed_end_while") {
    auto& statement = statement_info_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_WHILE;
    statement.name = name;
    ClearStagedState();
  }

 public:
  /*
   * Global interface.
   */
  virtual void Program() = 0;
  // NOTE: By chinmayee
  virtual void ComputeInitialPartitionPlacement(int /*num_running_workers*/) {
    // This function runs only at controller after loading application.
    // Implement this if application information/hints are to be used
    // for placing partitions.
    // Do nothing by default.
  }
  // NOTE: By chinmayee
  virtual void SetMigratableVariables() {
    // Determine which variables to update placement for, runs only at controller.
  }
  virtual void LoadParameter(const std::string& parameter) = 0;
  template <typename T>
  void LoadFlag(const std::string& name, T& value,   // NOLINT
                cereal::XMLInputArchive& archive) {  // NOLINT
    try {
      archive(cereal::make_nvp(name, value));
    } catch (cereal::Exception&) {
    }
  }

  //! Fills in missing metadata.
  void FillInProgram();
  //! Prints the program in a string.
  std::string Print() const;
  //! Update placement.
  void SetMigratableVariablesInternal();

  const VariableInfoMap* get_variable_info_map() const {
    return &variable_info_map_;
  }
  // NOTE: By chinmayee
  void set_update_placement_for_variable(
    VariableId vid, bool update_placement) {
    variable_info_map_[vid].update_placement = update_placement;
  }
  const VariableGroupInfoMap* get_variable_group_info_map() const {
    return &variable_group_info_map_;
  }
  void set_update_placement_for_variable_group(
    VariableGroupId gid, bool update_placement) {
    variable_group_info_map_[gid].update_placement = update_placement;
  }
  const StatementInfoMap* get_statement_info_map() const {
    return &statement_info_map_;
  }
  VariableGroupPlacementMap* initial_variable_group_placement() {
    return &initial_variable_group_placement_;
  }

 private:
  VariableId AllocateVariableId() { return next_variable_id_++; }
  VariableId next_variable_id_ = VariableId::FIRST;

  StatementId AllocateStatementId() { return next_statement_id_++; }
  StatementId next_statement_id_ = StatementId::FIRST;

  VariableInfoMap variable_info_map_;
  VariableGroupInfoMap variable_group_info_map_;

  // Partition information -- map from variable group to partition id.
  VariableGroupPlacementMap initial_variable_group_placement_;

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
    statement_info->compute_timer_start = compute_timer_started_;
    statement_info->compute_timer_active = compute_timer_active_;
  }

  //! Clears staged states.
  void ClearStagedState() {
    staged_has_writer_ = false;
    staged_buffer_access_.clear();
    staged_pause_needed_ = false;
    staged_track_needed_ = false;
    compute_timer_started_ = false;
  }

  //! Stores the metadata of the upcoming statement.
  bool staged_has_writer_ = false;
  std::map<VariableId, VariableAccess> staged_buffer_access_;
  bool staged_pause_needed_ = false;
  bool staged_track_needed_ = false;
  bool compute_timer_started_ = false;
  bool compute_timer_active_ = false;
};

// NOTE: By chinmayee
/*
 * Partition placement .
 */
class PartitionPlacement {
  public:
    PartitionPlacement() = delete;
    PartitionPlacement(int np) : num_partitions_(np) {
      placement_.resize(np);
      invalid_ = false;
    }

    int GetNumPartitions() const { return num_partitions_; }

    void PlacePartitionOnWorker(int pid, int wid) {
      assert(pid >= 0);
      assert(pid < num_partitions_);
      placement_[pid] = wid;
    }  // PlacePartitionOnWorker

    int GetPartitionPlacement(int pid) const {
      return placement_[pid];
    }  // GetPartitionPlacement

    const std::vector<int>& GetPlacementData() const {
      return placement_;
    }  // GetPlacementData

    void SetPlacementData(const std::vector<int> &placement) {
      assert(int(placement.size()) == num_partitions_);
      placement_ = placement;
    }  // SetPlacementData

    bool IsSameAs(const PartitionPlacement& other) const {
      if (num_partitions_ != other.num_partitions_) {
        return false;
      }
      if (invalid_ != other.invalid_) {
        return false;
      }
      for (int i = 0; i < num_partitions_; ++i) {
        if (placement_[i] != other.placement_[i]) {
          return false;
        }
      }  // for i < num_partitions_
      return true;
    }  // IsSameAs

    std::string Print() const {
      std::stringstream ss;
      ss << "Number of partitions = " << num_partitions_ << std::endl;
      for (int i = 0; i < num_partitions_; ++i) {
        ss << "* partition " << i << " : worker " << placement_[i] << std::endl;
      }
      return ss.str();
    }  // Print

    bool IsValid() const { return !invalid_; }
    bool IsInvalid() const { return invalid_; }
    void SetInvalid(bool invalid = true) { invalid_ = invalid; }

  private:
    bool invalid_;
    int num_partitions_;
    std::vector<int> placement_;
};  // class PartitionPlacement

// NOTE: By chinmayee
/*
 * PartitionHistory class computes and stores partition placements to use as
 * simulation evolves.
 */
class PartitionHistory {
  public:
    PartitionHistory()
      : num_partitions_(0), history_len_(0), last_time_(0),
        previous_placement_(0), active_placement_(-1) {
      previous_placement_.SetInvalid(true);
    }
    PartitionHistory(int np)
      : num_partitions_(np), history_len_(0), last_time_(0),
        previous_placement_(np), active_placement_(-1) {
      previous_placement_.SetInvalid(true);
    } 

    void SetNumPartitions(int np) {
      assert(history_len_ == 0);
      num_partitions_ = np;
    }
    int GetNumPartitions() const { return num_partitions_; }
    int GetHistoryLen() const { return history_len_; }

    void SetLastTime(float last_time) { last_time_ = last_time; }
    float GetLastTime() const { return last_time_; }

    void AddPartitionPlacement(float time, const PartitionPlacement &placement) {
      assert(num_partitions_ > 0);
      assert(num_partitions_ == placement.GetNumPartitions());
      assert(history_len_ >= 0);
      assert(int(history_.size()) == history_len_);
      assert(int(times_.size()) == history_len_);
      if (times_.size() > 0) {
        assert(time > times_.back());
      }
      times_.push_back(time);
      history_.push_back(placement);
      previous_placement_ = placement;
      history_len_++;
    }  // AddPartitionPlacement

    void AddPartitionPlacement(float time, const std::vector<int> &placement) {
      assert(num_partitions_ > 0);
      assert(num_partitions_ == int(placement.size()));
      assert(history_len_ >= 0);
      assert(int(history_.size()) == history_len_);
      assert(int(times_.size()) == history_len_);
      if (times_.size() > 0) {
        CHECK(time > times_.back()) << "Received placement for time " <<
          time << " which is less than already received " << times_.back();
      }
      times_.push_back(time);
      history_.push_back(PartitionPlacement(num_partitions_));
      history_.back().SetPlacementData(placement);
      previous_placement_ = history_.back();
      history_len_++;
    }  // AddPartitionPlacement

    const PartitionPlacement &GetPlacement(int i) const {
      assert(i >= 0);
      assert(i < history_len_);
      return history_[i];
    }  // Placement

    bool AreEquivalentPlacements(int i, int j) const {
      return history_[i].IsSameAs(history_[j]);
    }  // AreEquivalentPlacements

    float GetTime(int i) const {
      assert(i >= 0);
      assert(i < history_len_);
      return times_[i];
    }  // GetTime

    const PartitionPlacement &PreviousPlacement() const {
      return previous_placement_;
    }  // PreviousPlacement
    void SetPreviousPlacement(const PartitionPlacement &placement) {
      previous_placement_ = placement;
    }  // SetPreviousPlacement

    int ActivePlacement() const {
      assert(active_placement_ < history_len_);
      return active_placement_;
    }  // ActivePlacement
    void SetActivePlacement(int active_placement) {
      CHECK(active_placement >= 0);
      CHECK(active_placement < history_len_);
      active_placement_ = active_placement;
    }  // SetPreviousPlacement

    const std::vector<VariableGroupId> &UpdateVariableGroups() const {
      return update_variable_groups_;
    }
    void SetUpdateVariableGroups(
      const std::vector<VariableGroupId> &update_variable_groups) {
      update_variable_groups_ = update_variable_groups;
    }

    // NOTE: By chinmayee
    bool UpdatePlacementInProgress() const {
      return update_placement_in_progress_;
    }
    void SetUpdatePlacementInProgress(bool update_placement_in_progress) {
      update_placement_in_progress_ = update_placement_in_progress;
    }

    // NOTE: By chinmayee
    bool UpdatePlacementReceivingRequests() const {
      return update_placement_receiving_requests_;
    }
    void SetUpdatePlacementReceivingRequests(
      bool update_placement_receiving_requests) {
      update_placement_receiving_requests_ = update_placement_receiving_requests;
    }

    // NOTE: By chinmayee
    int UpdatePlacementRequestsReceived() const {
      return update_placement_requests_received_;
    }
    void SetUpdatePlacementRequestsReceived(
      int update_placement_requests_received) {
      update_placement_requests_received_ = update_placement_requests_received;
    }
    void AddUpdatePlacementRequestReceived() {
      update_placement_requests_received_ += 1;
    }

    // NOTE: By chinmayee
    int UpdatePlacementRequestsExpected() const {
      return update_placement_requests_expected_;
    }
    void SetUpdatePlacementRequestsExpected(
      int update_placement_requests_expected) {
      update_placement_requests_expected_ = update_placement_requests_expected;
    }

    float UpdatePlacementTime() const {
      return placement_request_time_;
    }
    void SetUpdatePlacementTime(float placement_time) {
      placement_request_time_ = placement_time;
    }

    // Determine which partition to use next -- determine the index for the
    // partition for greatest time <= placement_request_time_.
    int DetermineNextPlacement() {
      float thresh = 1e-2;
      assert(history_len_ > 0);
      assert(history_len_ = int(times_.size()));
      assert(times_.size() == history_.size());
      for (int i = history_len_-1; i >= 0; --i) {
        if (times_[i] <= placement_request_time_ + thresh) {
          return i;
        }
      }  // for int i
      return 0;
    }  // DetermineNextPlacement

    // Add/clear partitions that have sent request.
    void AddToPendingResponse(const FullPartitionId partition_id) {
      pending_response_.push_back(partition_id);
    }
    void ClearPendingResponse() {
      pending_response_.clear();
    }
    const std::vector<FullPartitionId> &PendingResponse() {
      return pending_response_;
    }

    void Clear() {
      if (history_.size() > 0) {
        previous_placement_ = history_.back();
      }
      times_.clear();
      history_.clear();
      history_len_ = 0;
    }  // Clear

  private:
    int num_partitions_;
    int history_len_;
    std::vector<PartitionPlacement> history_;
    std::vector<float> times_;
    // Need last time to record how far ahead coarse sim has run. There may
    // not be an update in partition placement, and we need to know that it is
    // ok to use current partitioning till this time.
    float last_time_;
    // Used only at worker, not at controller.
    PartitionPlacement previous_placement_;
    // Used only at controller, not at worker.
    int active_placement_;
    std::vector<VariableGroupId> update_variable_groups_;
    bool update_placement_in_progress_ = false;
    bool update_placement_receiving_requests_ = false;
    int update_placement_requests_received_ = 0;
    int update_placement_requests_expected_ = 0;
    float placement_request_time_ = 0;
    // Partitions that have sent update request -- need to respond back to these.
    std::vector<FullPartitionId> pending_response_;
};  // class PartitionHistory

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
