#define EIGEN_USE_GPU

#include "cuda_helper.h"

#include <cuda.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/Tensor"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include "cublas_v2.h"

namespace canary {

// The global variable that saves the cuBLAS handler.
cublasHandle_t cublas_handle;
bool cublas_initialized = false;
// The global variable that saves the GPU device handler.
Eigen::CudaStreamDevice cuda_stream(0);
Eigen::GpuDevice gpu_device(&cuda_stream);

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

struct PseudoRandomGenerator {
 double low_, high_;
 __host__ __device__ PseudoRandomGenerator(double low, double high) : low_(low), high_(high) {};
 __host__ __device__ float operator()(const unsigned int n) const {
   thrust::default_random_engine rng;
   thrust::uniform_real_distribution<double> dist(low_, high_);
   rng.discard(n);
   return dist(rng);
  }
};

/*
 * Generate random features and their labels.
 */
void GenerateRandomData(const std::vector<double> reference,
                        GpuTensorStore<double, 2>* x_data,
                        GpuTensorStore<double, 1>* y_data) {
  // Initialize cuBLAS here.
  if (!cublas_initialized) {
    cublas_initialized = true;
    cublasCreate(&cublas_handle);
  }
  const size_t dim = x_data->get_ranks()[0];
  const size_t samples = x_data->get_ranks()[1];
  if (reference.size() != dim || y_data->get_ranks()[0] != samples) {
    fprintf(stderr, "Mismatched dimensions in GenerateRandomData!\n");
    return;
  }
  Eigen::TensorMap<Eigen::Tensor<double, 2>> x_tensor((double*)x_data->get_data(), dim, samples);
  // The Eigen library cannot generate random numbers correctly.
  // x_tensor.device(gpu_device) = x_tensor.random() - x_tensor.constant(0.5);
  // Instead, use thrust to generate random numbers.
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + dim * samples,
		    thrust::device_ptr<double>((double*)x_data->get_data()),
                    PseudoRandomGenerator(-0.5, 0.5));
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
  // Wait for GPU computations to complete.
  cudaDeviceSynchronize();
}

// Unused kernels.
// // General version with no assumptiong about dim.
// #if __CUDA_ARCH__ < 600
// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
// 
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));
// 
//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);
// 
//     return __longlong_as_double(old);
// }
// #endif
// 
// __global__ void ComputeDotProduct(double* w_data, double* x_data, double* factor_data, int dim) {
//   if (threadIdx.x == 0) {
//     factor_data[blockIdx.x] = 0;
//   }
//   __syncthreads();
//   int result = w_data[threadIdx.x] * x_data[blockIdx.x * dim + threadIdx.x];
//   // Synchronous instructions within a warp to reduce the sum.
//   // Assume WARP_SIZE = 32.
//   result += __shfl_down(result, 16);
//   result += __shfl_down(result, 8);
//   result += __shfl_down(result, 4);
//   result += __shfl_down(result, 2);
//   result += __shfl_down(result, 1);
//   if (threadIdx.x % 32 == 0) atomicAdd(&factor_data[blockIdx.x], result);
// }

// // Simpler version assuming dim <= 32.
// __global__ void ComputeDotProduct(double* w_data, double* x_data, double* factor_data, int dim) {
//   int result = w_data[threadIdx.x] * x_data[blockIdx.x * dim + threadIdx.x];
//   // Synchronous instructions within a warp to reduce the sum.
//   result += __shfl_down(result, 16);
//   result += __shfl_down(result, 8);
//   result += __shfl_down(result, 4);
//   result += __shfl_down(result, 2);
//   result += __shfl_down(result, 1);
//   factor_data[blockIdx.x] = result;
// }

__global__ void UpdateFactorKernel(double* factor_data, double* y_data, int samples) {
  int index = blockIdx.x * 32 + threadIdx.x;
  if (index < samples) {
    factor_data[index] = y_data[index] * (1. / (1. + exp(-y_data[index] * factor_data[index])) - 1.);
  }
}

/*
 * Use existing libraries to update the weights.
 */
void UpdateWeight(const GpuTensorStore<double, 2>& x_data,
                  const GpuTensorStore<double, 1>& y_data,
                  const GpuTensorStore<double, 1>& w_data,
                  GpuTensorStore<double, 1>* g_data) {
  const size_t dim = x_data.get_ranks()[0];
  const size_t samples = x_data.get_ranks()[1];
  g_data->Resize({dim});
  GpuTensorStore<double, 1> factor_data;
  factor_data.Resize({samples});
  double alpha = 1;
  double beta = 0;
  cublasStatus_t return_status;

  // Three approaches: (1) gemv is 3x slower than the handwritten kernel. (2) gemm is no faster than gemv.
  // Approach 1: a hand-written kernel.
  // ComputeDotProduct<<<samples, dim>>>((double*)w_data.get_data(), (double*)x_data.get_data(), (double*)factor_data.get_data(), dim);
  // Approach 2: cuBLAS/gemv.
  return_status =
    cublasDgemv(cublas_handle, CUBLAS_OP_T,
                dim, samples, &alpha,
                (double*)x_data.get_data(), dim,
                (double*)w_data.get_data(), 1,
                &beta,
                (double*)factor_data.get_data(), 1);
  if (return_status != 0) {
    fprintf(stderr, "cuBLAS error %d\n", int(return_status));
  }
  // Approach 3: cuBLAS/gemm.
  // return_status =
  //  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //              1, samples, dim, &alpha,
  //              (double*)w_data.get_data(), 1,
  //              (double*)x_data.get_data(), dim,
  //              &beta,
  //              (double*)factor_data.get_data(), 1);
  // if (return_status != 0) {
  //   fprintf(stderr, "cublas error %d\n", int(return_status));
  // }

  // Manually update the factors.
  UpdateFactorKernel<<<(samples + 31) / 32, 32>>>((double*)factor_data.get_data(), (double*)y_data.get_data(), samples);
  // Use the cuBLAS library to compute the matrix multiplication.
  return_status =
    cublasDgemv(cublas_handle, CUBLAS_OP_N,
                dim, samples, &alpha,
                (double*)x_data.get_data(), dim,
                (double*)factor_data.get_data(), 1,
                &beta,
                (double*)g_data->get_data(), 1);
  if (return_status != 0) {
    fprintf(stderr, "cuBLAS error %d\n", int(return_status));
  }
  // Wait for GPU computations to complete.
  cudaDeviceSynchronize();
}

/*
 * Compute the gradient and reduce by 32x.
 */
__global__ void ComputeGradientPart(double* x_data, double* y_data, double* w_data, int dim, int samples, int num_warps,
                                    double* interg_data) {
  int base_index = blockIdx.x * 32 + threadIdx.x;
  double factor = 0;
  // CAUTION: hard-coded dim.
  double x_data_buffer[20];
  if (base_index < samples) {
    for (int i = 0; i < dim; ++i) {
      x_data_buffer[i] = x_data[base_index * dim + i];
      factor += x_data_buffer[i] * w_data[i];
    }
    const double y_data_buffer = y_data[base_index];
    factor = y_data_buffer * (1. / (1. + exp(-y_data_buffer * factor)) - 1.);
  }
  double temp;
  for (int i = 0; i < dim; ++i) {
    temp = factor * x_data_buffer[i];
    // Every wrap of size 32 sums up the gradients sychronously. No synchronizaiton is needed.
    temp += __shfl_down(temp, 16);
    temp += __shfl_down(temp, 8);
    temp += __shfl_down(temp, 4);
    temp += __shfl_down(temp, 2);
    temp += __shfl_down(temp, 1);
    if (threadIdx.x == 0) {
      interg_data[i * num_warps + blockIdx.x] = temp;
    }
  }
}

void UpdateWeightTuned(const GpuTensorStore<double, 2>& x_data,
		const GpuTensorStore<double, 1>& y_data,
		const GpuTensorStore<double, 1>& w_data,
		GpuTensorStore<double, 1>* g_data) {
  const size_t dim = x_data.get_ranks()[0];
  const size_t samples = x_data.get_ranks()[1];
  int threads_per_block = 32;
  int num_blocks = (samples + threads_per_block - 1) / threads_per_block;
  int num_warps = (samples + 31) / 32;
  GpuTensorStore<double, 1> interg_data;
  interg_data.Resize({num_warps * dim});
  ComputeGradientPart<<<num_blocks, threads_per_block>>>(
      (double*)x_data.get_data(), (double*)y_data.get_data(), (double*)w_data.get_data(),
      dim, samples, num_warps, (double*)interg_data.get_data());
  std::vector<double> result(dim);
  for (int i = 0; i < dim; ++i) {
    // Thrust reduction is highly optimized, and this kernel is not the dominant computation part.
    result[i] = thrust::reduce(
		    thrust::device_ptr<double>((double*)interg_data.get_data() + i * num_warps),
                    thrust::device_ptr<double>((double*)interg_data.get_data() + (i+1) * num_warps),
		    0., thrust::plus<double>());
  }
  g_data->ToDevice(result);
  // Wait for GPU computations to complete.
  cudaDeviceSynchronize();
}

}  // namespace app

}  // namespace canary
