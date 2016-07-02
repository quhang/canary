#include <map>
#include <string>

namespace canary {

// TODO. ApplicationId, VariableId. StatementId.

//! Variable access.
enum class VariableAccess : int32_t {
  INVALID = -1, READ, WRITE
};

enum class StatementType : int32_t {
  INVALID = -1, TRANSFORM, SCATTER, GATHER, LOOP, END_LOOP, WHILE, END_WHILE
};

class TaskContext;

/**
 * A user writes an application by inheriting the application class.
 */
class Application {
 public:
  //! An variable handle stores a variable id and its type.
  template <typename Type> class VariableHandle {
   public:
    //! Variable type.
    typedef Type VariableType;
    //! Copy constructor is allowed.
    VariableHandle(const VariableHandle&) = default;
    //! Copy assignment is allowed.
    VariableHandle& operator= (const VariableHandle&) = default;
    //! Gets the variable id.
    VariableId get_variable_id() const { return variable_id_; }

   private:
    //! Only the application class can construct it.
    explicit VariableHandle(VariableId variable_id)
        : variable_id_(variable_id) {}
    const VariableId variable_id_;
  };

  //! An statement handle stores information about a statement.
  struct StatementHandle {
    StatementType statement_type = StatementType::INVALID;
    std::function<void(TaskContext*)> void_task_function;
    std::function<int(TaskContext*)> int_task_function;
    std::function<bool(TaskContext*)> bool_task_function;
    int num_loop = 0;
    std::map<VariableId, VariableAccess> variable_access_map;
    pause_needed = false;
    track_needed = false;
  };

 public:
  template <typename Type>
  VariableHandle<Type> DeclareVariable(int parallelism = -1) {
    if (parallelism != -1) {
      CHECK_GE(parallelism, 1)
          << "Partitioning of a variable is a positive number.";
    }
    const auto variable_id = AllocateVariableId();
    variable_partitioning_map_[variable_id] = parallelism;
    return VariableHandle<Type>(variable_id);
  }

  template <typename Type>
  void ReadAccess(VariableHandle<Type> variable_handle) {
    ReadAccess(variable_handle.get_variable_id());
  }

  void ReadAccess(VariableId variable_id) {
    auto iter = staged_buffer_access_.find(variable_id);
    if (iter == staged_buffer_access_.end()) {
      staged_buffer_access_[variable_id] = VariableAccess::READ;
    }
  }

  template <typename Type>
  void WriteAccess(VariableHandle<Type> variable_handle) {
    WriteAccess(variable_handle.get_variable_id());
  }

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

  void PauseNeeded() {
    staged_pause_needed_ = true;
  }

  void TrackNeeded() {
    staged_track_needed_ = true;
  }

  void Transform(std::function<void(TaskContext*)> task_function) {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::TRANSFORM;
    statement.void_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Scatter(std::function<void(TaskContext*)> task_function) {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::SCATTER;
    statement.void_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Gather(std::function<int(TaskContext*)> task_function) {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::GATHER;
    statement.int_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void Loop(int loop) {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::LOOP;
    statement.num_loop = loop;
    ClearStagedState();
  }

  void EndLoop() {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_LOOP;
    ClearStagedState();
  }

  void While(std::function<bool(TaskContext*)> task_function) {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::WHILE;
    statement.bool_task_function = std::move(task_function);
    ApplyStagedState(&statement);
    ClearStagedState();
  }

  void EndWhile() {
    auto& statement = statement_handle_map_[AllocateStatementId()];
    statement.statement_type = StatementType::END_WHILE;
    ClearStagedState();
  }

 public:
  virtual void Program() = 0;
  virtual void LoadParamater(const std::string& parameter) = 0;
  virtual std::string SaveParameter() = 0;

  const std::map<VariableId, int>& get_variable_partitioning_map() {
    return variable_partitioning_map_;
  }
  const std::map<StatementId, StatementHandle>& get_statement_handle_map_() {
    return statement_handle_map_;
  }

 private:
  VariableId AllocateVariableId() {
    return next_variable_id_++;
  }
  VariableId next_variable_id_ = VariableId::FIRST;

  StagementId AllocateStatementId() {
    return next_statement_id_++;
  }
  StatementId next_statement_id_ = StatementId::FIRST;

  // -1 means unknown.
  std::map<VariableId, int> variable_partitioning_map_;
  std::map<StatementId, StatementHandle> statement_handle_map_;

  //! Applies staged states to a statement.
  void ApplyStagedState(StatementHandle* statement_handle) {
    // Adds a dumb writing variable.
    if (!staged_has_writer_) {
      auto dumb_variable = DeclareVariable<bool>();
      WriteVariable(dump_variable);
    }
    if (statement_handle->statement_type == StatementType::WHILE) {
      for (auto pair : staged_buffer_access_) {
        if (variable_partitioning_map_.at[pair.first] == -1) {
          variable_partitioning_map_.at[pair.first] = 1;
        }
        CHECK_EQ(variable_partitioning_map_.at[pair.first], 1)
            << "WHILE statement accesses multi-partitioning variable!";
      }
    }
    statement_handle->variable_access_map = std::move(staged_buffer_access_);
    statement_handle->pause_needed = staged_pause_needed_;
    statement_handle->track_needed = staged_track_needed_;
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

class WorkerSendDataInterface;
/**
 * The context of a task.
 */
class TaskContext {
 public:
  template <typename T> void Broadcast(const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(data);
    }
    BroadcastInternal(buffer);
  }

  template<typename T, typename Function>
  std::enable_if_t<
  std::is_same<std::decay_t<std::result_of_t<Function(T, T)>>, T>::value, T>
  Reduce(T initial, Function combiner) {
    std::vector<T> receive_data(Gather());
    for (auto& element : receive_data) {
      initial = combiner(initial, element);
    }
    return initial;
  }

  template <typename T, typename Function>
  std::enable_if_t<std::is_void<std::result_of_t<Function(T, T*)>>::value, void>
  Reduce(T* initial, Function combiner) {
    CHECK_NOTNULL(initial);
    std::vector<T> receive_data(Gather());
    for (auto& element : receive_data) {
      combiner(element, initial);
    }
  }

  template <typename T> void Scatter(int partition_id, const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(data);
    }
    ScatterInternal(partition_id, buffer);
  }

  template <typename T> std::vector<T> Gather() {
    std::vector<T> receive_data(receive_buffer_.size());
    auto iter = receive_data.front();
    for (auto buffer : receive_buffer_) {
      {
        CanaryInputArchive archive(buffer);
        archive(*iter);
      }
      evbuffer_free(buffer);
      ++iter;
    }
    receive_buffer_.clear();
    return std::move(receive_data);
  }

  template <typename T> void OrderedScatter(int partition_id, const T& data) {
    struct evbuffer* buffer = evbuffer_new();
    {
      CanaryOutputArchive archive(buffer);
      archive(GetPartitionId());
      archive(data);
    }
    ScatterInternal(partition_id, buffer);
  }
  template <typename T> std::map<int, T> OrderedGather() {
    std::map<int, T> receive_buffer;
    int src_partition_id;
    for (auto buffer : receive_buffer_) {
      CanaryInputArchive archive(buffer);
      archive(src_partition_id);
      archive(receive_buffer[src_partition_id]);
    }
    return std::move(receive_buffer);
  }

  template <typename T>
  const T& ReadVariable(VariableHandle<T> handle) {
    auto pointer =
        dynamic_cast<TypedPartitionData<T>*>(ReadVariableInternal(handle));
    CHECK(pointer != nullptr) << "Invalid variable read.";
    return *pointer->get_data();
  }

  template <typename T>
  T* WriteVariable(Application::VariableHandle<T> handle) {
    auto pointer =
        dynamic_cast<TypedPartitionData<T>*>(WriteVariableInternal(handle));
    CHECK(pointer != nullptr) << "Invalid variable read.";
    return pointer->get_data();
  }

  int GetGatherSize() const {
    return static_cast<int>(receive_buffer_.size());
  }
  int GetScatterParallelism() const {
    return scatter_partitioning_;
  }
  int GetGatherParallelism() const {
    return gather_partitioning_;
  }
  int GetPartitionId() const {
    return self_partition_id_;
  }

 private:
  void BroadcastInternal(struct evbuffer* buffer);
  void ScatterInternal(int partition_id, struct evbuffer* buffer);
  void* ReadVariableInternal(VariableId variable_id);
  void* WriteVariableInternal(VariableId variable_id);
  WorkerSendDataInterface* send_data_interface_;
  std::list<struct evbuffer*> receive_buffer_;
  std::map<VariableId, PartitionData*> read_partition_data_map_;
  std::map<VariableId, PartitionData*> write_partition_data_map_;
  int self_partition_id_;
  int scatter_partitioning_ = -1;
  int gather_partitioning_ = -1;
  ApplicationId application_id_;
  VariableGroupId gather_variable_group_id_;
  StageId gather_stage_id_;
};

}  // namespace canary

#ifndef REGISTER_APPLICATION
#define REGISTER_APPLICATION(APPLICATION_CLASS)
#endif  // REGISTER_APPLICATION

#ifndef EXPECT_GATHER_SIZE
#define EXPECT_GATHER_SIZE(x)  \
    do {  \
      const int buffer_size = task_context->GatherSize();  \
      CHECK_NE(buffer_size, -1);  \
      CHECK_LE(buffer_size, static_cast<int>(x));  \
      if (buffer_size != static_cast<int>(x)) return static_cast<int>(x);  \
    } while (0)
#endif  // EXPECT_GATHER_SIZE

