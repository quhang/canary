#define EIGEN_USE_GPU

#include "cuda_helper.h"

#include <cuda.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/Tensor"

namespace canary {

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::LoadFromHostVector(
    const std::vector<T>& input) {
  if (input.size() != get_num_elements()) {
    fprintf(stderr,
            "Deserialization for the GpuTensorStore failed internally!\n");
  } else {
    cudaMemcpy(data_, input.data(), input.size() * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  }
}

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::SaveToHostVector(
    std::vector<T>* input) const {
  if (data_) {
    input->resize(get_num_elements(), 0);
    cudaMemcpy(input->data(), data_, input->size() * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }
}

template<typename T, size_t Dimension>
void GpuTensorStore<T, Dimension>::Reset() {
  if (data_) {
    cudaFree(data_);
  }
  data_ = nullptr;
  true_size_ = 0;
}

template<typename T, size_t Dimension>
bool GpuTensorStore<T, Dimension>::Allocate(size_t num_elements) {
  if (true_size_ >= num_elements * sizeof(T)) {
    return true;
  } else {
    Reset();
    if (cudaMalloc(&data_, num_elements * sizeof(T)) == 0) {
      true_size_ = num_elements * sizeof(T);
      return true;
    } else {
      return false;
    }
  }
}

template class GpuTensorStore<float, 1>;
template class GpuTensorStore<float, 2>;
template class GpuTensorStore<float, 3>;
template class GpuTensorStore<double, 1>;
template class GpuTensorStore<double, 2>;
template class GpuTensorStore<double, 3>;

namespace app {

Eigen::CudaStreamDevice cuda_stream(0);
Eigen::GpuDevice gpu_device(&cuda_stream);

void GenerateRandomData(const std::vector<double> reference,
                        GpuTensorStore<double, 2>* x_data,
                        GpuTensorStore<double, 1>* y_data) {
  const size_t dim = x_data->get_ranks()[0];
  const size_t samples = x_data->get_ranks()[1];
  if (reference.size() != dim || y_data->get_ranks()[0] != samples) {
    fprintf(stderr, "Dimension mismatch for GPU execution!\n");
    return;
  }
  Eigen::TensorMap<Eigen::Tensor<double, 2>> x_tensor(
      (double*)x_data->get_data(), dim, samples);
  x_tensor.device(gpu_device) = x_tensor.random();
  Eigen::TensorMap<Eigen::Tensor<double, 2>> y_tensor(
      (double*)y_data->get_data(), 1, samples);
  GpuTensorStore<double, 1> w_data(std::array<size_t, 1>{dim});
  w_data.ToDevice(reference);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> w_tensor(
      (double*)w_data.get_data(), 1, dim);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  y_tensor.device(gpu_device) =
    (w_tensor.contract(x_tensor, product_dims) > y_tensor.constant(0)).select(
        y_tensor.constant(1), y_tensor.constant(-1));
}

void UpdateWeight(const GpuTensorStore<double, 2>& x_data,
                  const GpuTensorStore<double, 1>& y_data,
                  const GpuTensorStore<double, 1>& w_data,
                  GpuTensorStore<double, 1>* g_data) {
  const size_t dim = x_data.get_ranks()[0];
  const size_t samples = x_data.get_ranks()[1];
  Eigen::TensorMap<Eigen::Tensor<double, 2>> x_tensor(
      (double*)(x_data.get_data()), dim, samples);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> y_tensor(
      (double*)(y_data.get_data()), 1, samples);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> w_tensor(
      (double*)(w_data.get_data()), 1, dim);

  g_data->Resize({dim});
  Eigen::TensorMap<Eigen::Tensor<double, 2>> g_tensor(
      (double*)(g_data->get_data()), 1, dim);

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  auto dot = w_tensor.contract(x_tensor, product_dims);
  auto factor = y_tensor * (
      y_tensor.constant(1.) / (y_tensor.constant(1.) + (-y_tensor * dot).exp())
      - y_tensor.constant(1.));
  Eigen::array<Eigen::IndexPair<int>, 1> new_dims = { Eigen::IndexPair<int>(1, 1) };
  g_tensor.device(gpu_device) = x_tensor.contract(factor, new_dims);

//      const auto& feature = task_context->ReadVariable(d_feature);
//      const auto& local_w = task_context->ReadVariable(d_local_w).ToHost();
//      std::vector<double> local_gradient_buffer(DIMENSION, 0);
//      for (const auto& pair : feature) {
//        const Point& point = pair.first;
//        const bool flag = pair.second;
//        const auto dot = std::inner_product(local_w.begin(), local_w.end(),
//                                            point.begin(), 0.);
//        const double factor =
//            flag ? +(1. / (1. + std::exp(-dot)) - 1.)
//                 : -(1. / (1. + std::exp(+dot)) - 1.);
//        for (int i = 0; i < DIMENSION; ++i) {
//          local_gradient_buffer[i] += factor * point[i];
//        }
//      }
}

}  // namespace app


}  // namespace canary
