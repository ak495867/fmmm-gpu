#include "device_memory.h"

namespace lcn::oracle {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace lcn::oracle
