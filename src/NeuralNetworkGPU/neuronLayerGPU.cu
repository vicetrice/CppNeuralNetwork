#include "neuronLayerGPU.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

namespace
{
    __device__ float relu(float x)
    {
        return x > 0.0f ? x : 0.0f;
    }

    __global__ void forwardKernel(
        const float *weights,
        const float *input,
        const float *biases,
        float *output,
        int output_size,
        int input_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int row = idx; row < output_size; row += stride)
        {
            float sum = 0.0f;

            for (int k = 0; k < input_size; ++k)
                sum += weights[row * input_size + k] * input[k];

            float z = sum + biases[row];

            output[row] = relu(z);
        }
    }
}

namespace vicetriceNN
{
    neuronLayerGPU::neuronLayerGPU(int in_size, int out_size)
        : input_size(in_size),
          output_size(out_size),
          weights(out_size, in_size, 0.0f),
          biases(out_size, 1, 0.0f)
    {

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(
            0.0f, std::sqrt(2.0f / input_size));

        std::vector<float> weight_data(output_size * input_size);

        for (auto &w : weight_data)
            w = dist(rng);

        weights.assign(weight_data.data(), output_size, input_size);

        std::vector<float> bias_data(output_size, 0.0f);
        biases.assign(bias_data.data(), output_size, 1);

        threads_per_block = 256;
        blocks_weights = (output_size * input_size + threads_per_block - 1) / threads_per_block;
        blocks_bias = (output_size + threads_per_block - 1) / threads_per_block;
        block_input = (input_size + threads_per_block - 1) / threads_per_block;
    }

    neuronLayerGPU::~neuronLayerGPU() = default;

    viceCudaMath::matrix2DGPU<float>
    neuronLayerGPU::forward(
        const viceCudaMath::matrix2DGPU<float> &input) const
    {
        if (input.num_rows() != input_size || input.num_cols() != 1)
            throw std::runtime_error("Input size mismatch in forward");

        viceCudaMath::matrix2DGPU<float> output(output_size, 1, 0.0f);

        forwardKernel<<<blocks_weights, threads_per_block>>>(
            weights.device_data(),
            input.device_data(),
            biases.device_data(),
            output.device_data(),
            output_size,
            input_size);

        CUDA_CHECK(cudaDeviceSynchronize());

        return output;
    }
}