#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

namespace
{

    inline void gpuAssert(cudaError_t code, const char *file, int line)
    {
        if (code != cudaSuccess)
        {
            std::cerr << "\n===== CUDA ERROR =====\n"
                      << "Message: " << cudaGetErrorString(code) << "\n"
                      << "File: " << file << "\n"
                      << "Line: " << line << "\n"
                      << "======================\n";
            throw std::runtime_error("");
        }
    }

    template <typename T>
    __global__ void vectorAddKernel(const T *a, const T *b, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] + b[i];
    }

    template <typename T>
    __global__ void vectorSubKernel(const T *a, const T *b, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] - b[i];
    }

    template <typename T>
    __global__ void vectorMulKernel(const T *a, const T *b, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] * b[i];
    }

    template <typename T>
    __global__ void vectorDivKernel(const T *a, const T *b, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] / b[i];
    }

    template <typename T>
    __global__ void scalarMulKernel(const T *a, T scalar, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] * scalar;
    }

    template <typename T>
    __global__ void scalarDivKernel(const T *a, T scalar, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] / scalar;
    }

    template <typename T>
    __global__ void convertKernel(const T *src, T *dst, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            dst[i] = src[i];
    }

    template <typename T>
    __global__ void vectorAddScalarKernel(const T *a, T scalar, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] + scalar;
    }

    template <typename T>
    __global__ void vectorSubScalarKernel(const T *a, T scalar, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = a[i] - scalar;
    }

    template <typename T>
    __global__ void scalarSubVectorKernel(const T *a, T scalar, T *c, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
            c[i] = scalar - a[i];
    }

};

namespace viceCudaMath
{
    template <typename T>
    class vectorGPU
    {
    private:
        T *data;
        size_t sz;
        size_t cap;
        int threads;
        int blocks;

        void updateLaunchConfig()
        {
            threads = 256;
            if (sz == 0)
            {
                blocks = 1;
                return;
            }
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            blocks = std::min((int)((sz + threads - 1) / threads), prop.multiProcessorCount * 32);
        }

    public:
        vectorGPU() : data(nullptr), sz(0), cap(0), threads(256), blocks(1) {}
        ~vectorGPU()
        {
            if (data)
                CUDA_CHECK(cudaFree(data));
        }

        vectorGPU(const vectorGPU &other) : data(nullptr), sz(0), cap(0), threads(256), blocks(1)
        {
            if (other.sz > 0)
            {
                resize(other.sz);
                CUDA_CHECK((cudaMemcpy(data, other.data, other.sz * sizeof(T), cudaMemcpyDeviceToDevice)));
                sz = other.sz;
            }
        }

        vectorGPU(size_t n, T value = T{}) : data(nullptr), sz(0), cap(0), threads(256), blocks(1)
        {
            assign(n, value);
        }

        vectorGPU(vectorGPU &&other) noexcept
            : data(other.data), sz(other.sz), cap(other.cap), threads(other.threads), blocks(other.blocks)
        {
            other.data = nullptr;
            other.sz = 0;
            other.cap = 0;
        }

        void resize(size_t new_cap)
        {
            if (new_cap == cap)
                return;

            T *new_data = nullptr;
            CUDA_CHECK(cudaMalloc(&new_data, new_cap * sizeof(T)));

            if (data && sz > 0)
            {
                size_t copy_size = (sz < new_cap) ? sz : new_cap;
                CUDA_CHECK(cudaMemcpy(new_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(data));
                if (new_cap < sz)
                    sz = new_cap;
            }

            data = new_data;
            cap = new_cap;
            updateLaunchConfig();
        }

        void push_back(const T &value)
        {
            if (sz == cap)
                resize(cap ? cap * 2 : 1);
            CUDA_CHECK(cudaMemcpy(data + sz, &value, sizeof(T), cudaMemcpyHostToDevice));
            ++sz;
            updateLaunchConfig();
        }

        void pop_back()
        {
            if (sz > 0)
                --sz;
            updateLaunchConfig();
        }

        vectorGPU &operator=(const std::vector<T> &host_vec)
        {
            size_t n = host_vec.size();
            if (n > cap)
                resize(n);
            CUDA_CHECK(cudaMemcpy(data, host_vec.data(), n * sizeof(T), cudaMemcpyHostToDevice));
            sz = n;
            updateLaunchConfig();
            return *this;
        }

        vectorGPU &operator=(const vectorGPU &other)
        {
            if (this == &other)
                return *this;
            if (other.sz > cap)
                resize(other.sz);
            CUDA_CHECK(cudaMemcpy(data, other.data, other.sz * sizeof(T), cudaMemcpyDeviceToDevice));
            sz = other.sz;
            updateLaunchConfig();
            return *this;
        }

        vectorGPU &operator=(vectorGPU &&other) noexcept
        {
            if (this == &other)
                return *this;
            if (data)
                CUDA_CHECK(cudaFree(data));
            data = other.data;
            sz = other.sz;
            cap = other.cap;
            threads = other.threads;
            blocks = other.blocks;
            other.data = nullptr;
            other.sz = 0;
            other.cap = 0;
            return *this;
        }

        size_t size() const { return sz; }
        size_t capacity() const { return cap; }
        void clear()
        {
            sz = 0;
            updateLaunchConfig();
        }
        T *device_data() const { return data; }

        void print() const
        {
            if (sz == 0)
            {
                std::cout << "[]\n";
                return;
            }
            T *host_array = new T[sz];
            CUDA_CHECK(cudaMemcpy(host_array, data, sz * sizeof(T), cudaMemcpyDeviceToHost));
            std::cout << "[";
            for (size_t i = 0; i < sz; i++)
            {
                std::cout << host_array[i];
                if (i < sz - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
            delete[] host_array;
        }

        vectorGPU operator+(const vectorGPU &rhs) const
        {
            vectorGPU result;
            if (sz != rhs.sz)
                return result;
            result.resize(sz);
            result.sz = sz;
            vectorAddKernel<<<blocks, threads>>>(data, rhs.data, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        vectorGPU operator-(const vectorGPU &rhs) const
        {
            vectorGPU result;
            if (sz != rhs.sz)
                return result;
            result.resize(sz);
            result.sz = sz;
            vectorSubKernel<<<blocks, threads>>>(data, rhs.data, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        vectorGPU operator*(const vectorGPU &rhs) const
        {
            vectorGPU result;
            if (sz != rhs.sz)
                return result;
            result.resize(sz);
            result.sz = sz;
            vectorMulKernel<<<blocks, threads>>>(data, rhs.data, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        vectorGPU operator/(const vectorGPU &rhs) const
        {
            vectorGPU result;
            if (sz != rhs.sz)
                return result;
            result.resize(sz);
            result.sz = sz;
            vectorDivKernel<<<blocks, threads>>>(data, rhs.data, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        vectorGPU operator*(T scalar) const
        {
            vectorGPU result;
            result.resize(sz);
            result.sz = sz;
            scalarMulKernel<<<blocks, threads>>>(data, scalar, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        friend vectorGPU operator*(T scalar, const vectorGPU &vec)
        {
            return vec * scalar;
        }

        vectorGPU operator/(T scalar) const
        {
            vectorGPU result;
            result.resize(sz);
            result.sz = sz;
            scalarDivKernel<<<blocks, threads>>>(data, scalar, result.data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        friend vectorGPU operator/(T scalar, const vectorGPU &vec)
        {
            return vec / scalar;
        }

        template <typename U>
        explicit vectorGPU(const vectorGPU<U> &other)
        {
            sz = other.size();
            cap = sz;
            threads = 256;
            blocks = 1;
            updateLaunchConfig();

            if (sz > 0)
            {
                CUDA_CHECK(cudaMalloc(&data, sz * sizeof(T)));
                convertKernel<<<blocks, threads>>>(other.device_data(), data, sz);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        T get(size_t i) const
        {

            T value;
            CUDA_CHECK(cudaMemcpy(&value, data + i, sizeof(T), cudaMemcpyDeviceToHost));
            return value;
        }

        T operator[](size_t i)
        {
            return get(i);
        }

        void set(size_t i, const T &value)
        {

            CUDA_CHECK(cudaMemcpy(data + i, &value, sizeof(T), cudaMemcpyHostToDevice));
        }

        auto operator+(T scalar) const
        {

            vectorGPU<T> result;
            result.resize(sz);
            result.sz = sz;
            vectorAddScalarKernel<<<blocks, threads>>>(data, scalar, result.device_data(), sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        friend auto operator+(T scalar, const vectorGPU &vec)
        {
            return vec + scalar;
        }

        auto operator-(T scalar) const
        {

            vectorGPU<T> result;
            result.resize(sz);
            result.sz = sz;
            vectorSubScalarKernel<<<blocks, threads>>>(data, scalar, result.device_data(), sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        friend auto operator-(T scalar, const vectorGPU &vec)
        {

            vectorGPU<T> result;
            result.resize(vec.sz);
            result.sz = vec.sz;
            scalarSubVectorKernel<<<vec.blocks, vec.threads>>>(vec.data, scalar, result.device_data(), vec.sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        }

        std::vector<T> to_host() const
        {
            std::vector<T> host(sz);

            if (sz > 0)
            {
                CUDA_CHECK(cudaMemcpy(host.data(), data, sz * sizeof(T), cudaMemcpyDeviceToHost));
            }

            return host;
        }

        vectorGPU &operator+=(const vectorGPU &rhs)
        {
            if (sz != rhs.sz)
                throw std::runtime_error("Size mismatch in vectorGPU += operator");

            vectorAddKernel<<<blocks, threads>>>(data, rhs.data, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator-=(const vectorGPU &rhs)
        {
            if (sz != rhs.sz)
                throw std::runtime_error("Size mismatch in vectorGPU -= operator");

            vectorSubKernel<<<blocks, threads>>>(data, rhs.data, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator*=(const vectorGPU &rhs)
        {
            if (sz != rhs.sz)
                throw std::runtime_error("Size mismatch in vectorGPU *= operator");

            vectorMulKernel<<<blocks, threads>>>(data, rhs.data, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator/=(const vectorGPU &rhs)
        {
            if (sz != rhs.sz)
                throw std::runtime_error("Size mismatch in vectorGPU /= operator");

            vectorDivKernel<<<blocks, threads>>>(data, rhs.data, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator+=(T scalar)
        {
            vectorAddScalarKernel<<<blocks, threads>>>(data, scalar, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator-=(T scalar)
        {
            vectorSubScalarKernel<<<blocks, threads>>>(data, scalar, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator*=(T scalar)
        {
            scalarMulKernel<<<blocks, threads>>>(data, scalar, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        vectorGPU &operator/=(T scalar)
        {
            scalarDivKernel<<<blocks, threads>>>(data, scalar, data, sz);
            CUDA_CHECK(cudaDeviceSynchronize());
            return *this;
        }

        void assign(size_t n, T value = T{})
        {
            if (n > cap)
                resize(n);

            std::vector<T> temp(n, value);
            CUDA_CHECK(cudaMemcpy(data, temp.data(), n * sizeof(T), cudaMemcpyHostToDevice));

            sz = n;
            updateLaunchConfig();
        }

        void assign(const T *host_data, size_t n)
        {
            if (n > cap)
                resize(n);

            CUDA_CHECK(cudaMemcpy(data, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
            sz = n;
            updateLaunchConfig();
        }
    };
}