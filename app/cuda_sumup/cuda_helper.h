#include <thrust/device_vector.h>
namespace cuda_internal {
void Assign(thrust::device_vector<float>* output, long numbers, float value);
float Reduce(const thrust::device_vector<float>& input);
void Load(const std::vector<float>& input,
          thrust::device_vector<float>* output);
void Store(const thrust::device_vector<float>& input,
           std::vector<float>* output);
}  // namespace internal
