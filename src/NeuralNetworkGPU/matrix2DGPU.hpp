#pragma once

#include "vectorGPU.hpp"
#include <stdexcept>
#include <iostream>

namespace
{
    template <typename T>
    __global__ void matMulKernel(const T *A, const T *B, T *C, size_t n, size_t m, size_t p)
    {
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        size_t strideRow = blockDim.y * gridDim.y;
        size_t strideCol = blockDim.x * gridDim.x;

        for (size_t r = row; r < n; r += strideRow)
        {
            for (size_t c = col; c < p; c += strideCol)
            {
                T sum = 0;
                for (size_t k = 0; k < m; ++k)
                    sum += A[r * m + k] * B[k * p + c];
                C[r * p + c] = sum;
            }
        }
    }
}

namespace viceCudaMath
{
    template <typename T>
    class matrix2DGPU
    {
    private:
        vectorGPU<T> data;
        size_t rows, cols;

        size_t index(size_t r, size_t c) const
        {
            if (r >= rows || c >= cols)
                throw std::out_of_range("matrix2DGPU index out of range");
            return r * cols + c;
        }

    public:
        matrix2DGPU() : rows(0), cols(0) {}
        matrix2DGPU(size_t r, size_t c, T value = T{}) : rows(r), cols(c)
        {
            data.assign(r * c, value);
        }
        matrix2DGPU(const matrix2DGPU &other) : rows(other.rows), cols(other.cols), data(other.data) {}
        matrix2DGPU(matrix2DGPU &&other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data))
        {
            other.rows = 0;
            other.cols = 0;
        }

        void assign(size_t r, size_t c, T value = T{})
        {
            rows = r;
            cols = c;
            data.assign(r * c, value);
        }
        void assign(const T *host_data, size_t r, size_t c)
        {
            rows = r;
            cols = c;
            data.assign(host_data, r * c);
        }

        size_t num_rows() const { return rows; }
        size_t num_cols() const { return cols; }

        T get(size_t r, size_t c) const { return data.get(index(r, c)); }
        void set(size_t r, size_t c, T value) { data.set(index(r, c), value); }

        T *device_data() const { return data.device_data(); }

        void print() const
        {
            if (rows == 0 || cols == 0)
            {
                std::cout << "[]\n";
                return;
            }
            std::cout << "matrix2DGPU (" << rows << "x" << cols << "):\n";
            for (size_t r = 0; r < rows; ++r)
            {
                std::cout << "[ ";
                for (size_t c = 0; c < cols; ++c)
                    std::cout << get(r, c) << " ";
                std::cout << "]\n";
            }
        }

        // Operaciones con escalar
        matrix2DGPU &operator*=(T scalar)
        {
            data *= scalar;
            return *this;
        }
        matrix2DGPU &operator/=(T scalar)
        {
            data /= scalar;
            return *this;
        }
        matrix2DGPU operator*(T scalar) const
        {
            matrix2DGPU result(*this);
            result *= scalar;
            return result;
        }
        matrix2DGPU operator/(T scalar) const
        {
            matrix2DGPU result(*this);
            result /= scalar;
            return result;
        }

        // Suma y resta de matrices
        matrix2DGPU &operator+=(const matrix2DGPU &rhs)
        {
            if (rows != rhs.rows || cols != rhs.cols)
                throw std::runtime_error("Size mismatch");
            data += rhs.data;
            return *this;
        }
        matrix2DGPU &operator-=(const matrix2DGPU &rhs)
        {
            if (rows != rhs.rows || cols != rhs.cols)
                throw std::runtime_error("Size mismatch");
            data -= rhs.data;
            return *this;
        }
        matrix2DGPU operator+(const matrix2DGPU &rhs) const
        {
            matrix2DGPU result(*this);
            result += rhs;
            return result;
        }
        matrix2DGPU operator-(const matrix2DGPU &rhs) const
        {
            matrix2DGPU result(*this);
            result -= rhs;
            return result;
        }

        // Multiplicación matricial
        matrix2DGPU operator*(const matrix2DGPU &rhs) const
        {
            if (cols != rhs.rows)
                throw std::runtime_error("Size mismatch for matrix multiplication");

            matrix2DGPU result(rows, rhs.cols);
            dim3 threads(16, 16);
            dim3 blocks((rhs.cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

            matMulKernel<<<blocks, threads>>>(data.device_data(), rhs.data.device_data(), result.data.device_data(), rows, cols, rhs.cols);
            CUDA_CHECK(cudaDeviceSynchronize());

            return result;
        }

        // Operadores de asignación
        matrix2DGPU &operator=(const matrix2DGPU &other)
        {
            if (this == &other)
                return *this;
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            return *this;
        }
        matrix2DGPU &operator=(matrix2DGPU &&other) noexcept
        {
            if (this == &other)
                return *this;
            rows = other.rows;
            cols = other.cols;
            data = std::move(other.data);
            other.rows = 0;
            other.cols = 0;
            return *this;
        }

        std::vector<T> to_host() const
        {
            return data.to_host();
        }
    };
}