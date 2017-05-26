#include "cuda_helper.h"

#include <cuda.h>
#include "unsupported/Eigen/CXX11/Tensor"

namespace canary {

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::LoadFromHostVector(
    const std::vector<T>& input) {
  if (input.size() != get_num_elements()) {
    fprintf(stderr,
            "Deserialization for the GpuTensorStore failed internally!\n");
  } else {
    cudaMemcpy(data_, input.data(), input.size(), cudaMemcpyHostToDevice);
  }
}

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::SaveToHostVector(
    std::vector<T>* input) const {
  if (data_) {
    input->resize(get_num_elements(), 0);
    cudaMemcpy(input->data(), data_, input->size() * sizeof(T),
               cudaMemcpyDeviceToHost);
  }
}

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::Reset() {
  if (data_) {
    cudaFree(data_);
  }
  data_ = nullptr;
  for (int& elem : ranks_) { elem = 0; }
}

template<typename T, size_t Dimension>
bool GpuTensorStore<T, Dimension>::Allocate(size_t num_elements) {
  return cudaMalloc(&data_, num_elements * sizeof(T)) == 0;
}


}  // namespace canary
