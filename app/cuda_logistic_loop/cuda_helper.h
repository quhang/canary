#include <array>
#include <vector>

namespace canary {

/**
 * Class GpuTensorStore holds a GPU buffer allocated by cudaMalloc.
 * T is the stored data type, i.e. float or double. Dimension is the dimension
 * of the underlying tensor.
 */
template<typename T, size_t Dimension> class GpuTensorStore {
 public:
  /// The data type for the tensor rank.
  typedef std::array<size_t, Dimension> RankType;
  /// Initializes the tensor store as empty.
  GpuTensorStore() {}
  /// Initializes the tensor store given the rank.
  GpuTensorStore(const RankType& ranks) {
    ranks_ = ranks;
    if (!HasValidRank() || !Allocate(get_num_elements())) {
      Reset();
    }
  }
  virtual ~GpuTensorStore() { Reset(); }

  /// None-copiable and none-movable.
  GpuTensorStore(const GpuTensorStore&) = delete;
  GpuTensorStore(GpuTensorStore&&) = delete;
  GpuTensorStore& operator=(const GpuTensorStore&) = delete;
  GpuTensorStore& operator=(GpuTensorStore&&) = delete;

  void Resize(const RankType& ranks) {
    Reset();
    ranks_ = ranks;
    if (!HasValidRank() || !Allocate(get_num_elements())) {
      Reset();
      for (auto& elem : ranks_) { elem = 0; }
    }
  }

  /// Returns the GPU pointer to the buffer.
  void* get_data() const { return data_; }
  // Returns the rank.
  RankType get_ranks() const { return ranks_; }
  size_t get_num_elements() const {
    size_t result = 1;
    for (size_t elem : ranks_) { result *= elem; }
    return result;
  }

  /// Serialization function used by cereal.
  template<class Archive>
  void save(Archive& archive) const {
    archive(ranks_);
    // Serialization quits if the rank is invalid.
    if (!HasValidRank()) return;
    std::vector<T> temp;
    SaveToHostVector(&temp);
    archive(temp);
  }

  /// Deserialization function used by cereal.
  template<class Archive>
  void load(Archive & archive) {
    archive(ranks_);
    // Deserialization quits if the rank is invalid.
    if (!HasValidRank()) return;
    Allocate(get_num_elements());
    std::vector<T> temp;
    archive(temp);
    LoadFromHostVector(temp);
  }

  /// Loads the GPU buffer from ``input``, assuming the ranks have been filled
  // in correctly.
  void LoadFromHostVector(const std::vector<T>& input);
  /// Saves the GPU buffer into ``output``.
  void SaveToHostVector(std::vector<T>* output) const;
  /// Transfers the buffer to the host.
  std::vector<T> ToHost() const {
    std::vector<T> result;
    SaveToHostVector(&result);
    return result;
  }
  void ToDevice(const std::vector<T>& input) {
    if (Dimension == 1) {
      ranks_ = {input.size()};
      if (Allocate(get_num_elements())) {
        LoadFromHostVector(input);
      } else {
        fprintf(stderr, "Error alloc in GpuTensorStore::ToDevice!\n");
      }
    } else {
      fprintf(stderr, "Wrong dim in GpuTensorStore::ToDevice!\n");
    }
  }

 private:
  // Resets the buffer, including ``data_`` and ``true_size_``.
  void Reset();
  // Allocates the buffer, including ``data_`` and ``true_size_``.
  bool Allocate(size_t num_elements);
  bool HasValidRank() const {
    for (size_t elem : ranks_) {
      if (elem == 0) return false;
    }
    return true;
  };

  /// GPU data pointer.
  void* data_ = nullptr;
  size_t true_size_ = 0;
  RankType ranks_;
};

namespace app {

void GenerateRandomData(const std::vector<double> reference,
                        GpuTensorStore<double, 2>* x_data,
                        GpuTensorStore<double, 1>* y_data);

void UpdateWeight(const GpuTensorStore<double, 2>& x_data,
                  const GpuTensorStore<double, 1>& y_data,
                  const GpuTensorStore<double, 1>& w_data,
                  GpuTensorStore<double, 1>* g_data);
void UpdateWeightTuned(const GpuTensorStore<double, 2>& x_data,
                       const GpuTensorStore<double, 1>& y_data,
                       const GpuTensorStore<double, 1>& w_data,
                       GpuTensorStore<double, 1>* g_data);

}  // namespace app

}  // namespace canary
