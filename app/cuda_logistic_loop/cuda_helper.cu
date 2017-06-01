#define EIGEN_USE_GPU

#include "cuda_helper.h"

#include <cuda.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cublas_v2.h"

namespace canary {

cublasHandle_t cublas_handle;
bool cublas_initialized = false;




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
  if (!cublas_initialized) {
    cublas_initialized = true;
    cublasCreate(&cublas_handle);
  }
  const size_t dim = x_data->get_ranks()[0];
  const size_t samples = x_data->get_ranks()[1];
  if (reference.size() != dim || y_data->get_ranks()[0] != samples) {
    fprintf(stderr, "Dimension mismatch for GPU execution!\n");
    return;
  }
  // Call into the Eigen library to generate random numbers.
  Eigen::TensorMap<Eigen::Tensor<double, 2>> x_tensor(
      (double*)x_data->get_data(), dim, samples);
  x_tensor.device(gpu_device) = x_tensor.random() - x_tensor.constant(0.5);

  Eigen::TensorMap<Eigen::Tensor<double, 1>> y_tensor((double*)y_data->get_data(), samples);
  GpuTensorStore<double, 1> w_data;
  w_data.ToDevice(reference);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> w_tensor((double*)w_data.get_data(), dim, 1);

  // Call into the Eigen library to compute the labels.
  Eigen::array<int, 2> bcast1({1, int(samples)});
  Eigen::array<int, 1> dims1({0});
  y_tensor.device(gpu_device) =
    ((w_tensor.broadcast(bcast1) * x_tensor).sum(dims1) > y_tensor.constant(0)).select(
        y_tensor.constant(1), y_tensor.constant(-1));
  
}

// General version with no assumptiong about dim.
//#if __CUDA_ARCH__ < 600
//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//                              (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                               __longlong_as_double(assumed)));
//
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}
//#endif
//
//__global__ void ComputeDotProduct(double* w_data, double* x_data, double* factor_data, int dim) {
//  if (threadIdx.x == 0) {
//    factor_data[blockIdx.x] = 0;
//  }
//  __syncthreads();
//  int result = w_data[threadIdx.x] * x_data[blockIdx.x * dim + threadIdx.x];
//  // Synchronous instructions within a warp to reduce the sum.
//  // Assume WARP_SIZE = 32.
//  result += __shfl_down(result, 16);
//  result += __shfl_down(result, 8);
//  result += __shfl_down(result, 4);
//  result += __shfl_down(result, 2);
//  result += __shfl_down(result, 1);
//  if (threadIdx.x % 32 == 0) atomicAdd(&factor_data[blockIdx.x], result);
//}

// Simpler version assuming dim <= 32.
__global__ void ComputeDotProduct(double* w_data, double* x_data, double* factor_data, int dim) {
  int result = w_data[threadIdx.x] * x_data[blockIdx.x * dim + threadIdx.x];
  // Synchronous instructions within a warp to reduce the sum.
  result += __shfl_down(result, 16);
  result += __shfl_down(result, 8);
  result += __shfl_down(result, 4);
  result += __shfl_down(result, 2);
  result += __shfl_down(result, 1);
  factor_data[blockIdx.x] = result;
}

__global__ void UpdateFactorKernel(double* factor_data, double* y_data, int samples) {
  int index = blockIdx.x * 32 + threadIdx.x;
  if (index < samples) {
    factor_data[index] = y_data[index] * (1. / (1. + exp(-y_data[index] * factor_data[index])) - 1.);
  }
}

void UpdateWeight(const GpuTensorStore<double, 2>& x_data,
                  const GpuTensorStore<double, 1>& y_data,
                  const GpuTensorStore<double, 1>& w_data,
                  GpuTensorStore<double, 1>* g_data) {
  const size_t dim = x_data.get_ranks()[0];
  const size_t samples = x_data.get_ranks()[1];
  //Eigen::TensorMap<Eigen::Tensor<double, 2>> x_tensor((double*)(x_data.get_data()), dim, samples);
  //Eigen::TensorMap<Eigen::Tensor<double, 1>> y_tensor((double*)(y_data.get_data()), samples);
  //Eigen::TensorMap<Eigen::Tensor<double, 2>> w_tensor((double*)(w_data.get_data()), dim, 1);
  g_data->Resize({dim});
  //Eigen::TensorMap<Eigen::Tensor<double, 1>> g_tensor((double*)(g_data->get_data()), dim);
  GpuTensorStore<double, 1> factor_data;
  factor_data.Resize({samples});
  //Eigen::TensorMap<Eigen::Tensor<double, 1>> factor_tensor((double*)(factor_data.get_data()), samples);

  double alpha = 1;
  double beta = 0;
  cublasStatus_t return_status;
  ComputeDotProduct<<<samples, dim>>>((double*)w_data.get_data(), (double*)x_data.get_data(), (double*)factor_data.get_data(), dim);
  // Call into cuBLAS library to compute the matrix multiplication.
  // Result: gemv is 3x slower than the handwritten kernel.
  //cublasStatus_t return_status =
  //  cublasDgemv(cublas_handle, CUBLAS_OP_T,
  //              dim, samples, &alpha,
  //              (double*)x_data.get_data(), dim,
  //              (double*)w_data.get_data(), 1,
  //              &beta,
  //              (double*)factor_data.get_data(), 1);
  // Result: gemm is no faster than gemv.
  //cublasStatus_t return_status =
  //  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //              1, samples, dim, &alpha,
  //              (double*)w_data.get_data(), 1,
  //              (double*)x_data.get_data(), dim,
  //              &beta,
  //              (double*)factor_data.get_data(), 1);
  //if (return_status != 0) {
  //  fprintf(stderr, "cublas error %d\n", int(return_status));
  //}

  // Call into the Eigen library to compute the factors.
  UpdateFactorKernel<<<(samples + 31) / 32, 32>>>((double*)factor_data.get_data(), (double*)y_data.get_data(), samples);
  
  // Call into cuBLAS library to compute the matrix multiplication.
  return_status =
    cublasDgemv(cublas_handle, CUBLAS_OP_N,
                dim, samples, &alpha,
                (double*)x_data.get_data(), dim,
                (double*)factor_data.get_data(), 1,
                &beta,
                (double*)g_data->get_data(), 1);
  if (return_status != 0) {
    fprintf(stderr, "cublas error %d\n", int(return_status));
  }
  cudaDeviceSynchronize();
}

}  // namespace app


}  // namespace canary
