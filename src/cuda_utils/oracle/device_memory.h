#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace lcn::oracle {

void cuda_check(cudaError_t status, const char* operation);

// Small move-only RAII owner used by the oracle ABI.  A zero-sized buffer is
// represented by a null pointer and is never passed to cudaMalloc/cudaMemcpy.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        if (count == 0) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc");
        count_ = count;
    }

    void upload(const T* source, std::size_t count) {
        if (count != count_) throw std::logic_error("device upload size mismatch");
        if (count != 0) cuda_check(cudaMemcpy(ptr_, source, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }

    void download(T* destination, std::size_t count) const {
        if (count != count_) throw std::logic_error("device download size mismatch");
        if (count != 0) cuda_check(cudaMemcpy(destination, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }

    void reset() noexcept {
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    void move_from(DeviceBuffer& other) noexcept {
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace lcn::oracle
