#include <thrust/device_vector.h>
namespace cuda_internal {
void Assign(thrust::device_vector<double>* output, long numbers, double value);
double Reduce(const thrust::device_vector<double>& input);
void Load(const std::vector<double>& input,
          thrust::device_vector<double>* output);
void Store(const thrust::device_vector<double>& input,
           std::vector<double>* output);
}  // namespace internal
