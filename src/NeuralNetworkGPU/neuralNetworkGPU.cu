#include "neuralNetworkGPU.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

using namespace vicetriceNN;
using namespace viceCudaMath;

namespace
{

    __global__ void computeNewDeltaKernel(
        const float *W,
        const float *old_delta,
        const float *prev_activation,
        float *new_delta,
        int n_inputs,
        int n_neurons)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= n_inputs)
            return;

        float sum = 0.0f;

        for (int i = 0; i < n_neurons; ++i)
            sum += W[i * n_inputs + j] * old_delta[i];

        float grad = (prev_activation[j] > 0.0f) ? 1.0f : 0.0f;

        new_delta[j] = sum * grad;
    }

    __global__ void updateWeightsKernel(
        float *W,
        const float *delta,
        const float *prev_output,
        int n_neurons,
        int n_inputs,
        float learning_rate,
        float lambda)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_neurons * n_inputs)
            return;

        int i = idx / n_inputs;
        int j = idx % n_inputs;

        float grad = delta[i] * prev_output[j] + lambda * W[idx];

        W[idx] -= learning_rate * grad;
    }

    __global__ void updateBiasesKernel(
        float *B,
        const float *delta,
        int n_neurons,
        float learning_rate)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_neurons)
            return;

        B[idx] -= learning_rate * delta[idx];
    }

    __global__ void softmaxVectorKernel(float *data, int n)
    {
        float max_val = data[0];

        for (int i = 1; i < n; ++i)
            if (data[i] > max_val)
                max_val = data[i];

        float sum = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            data[i] = expf(data[i] - max_val);
            sum += data[i];
        }

        for (int i = 0; i < n; ++i)
            data[i] /= sum;
    }
}

void neuralNetworkGPU::addLayer(int input_size, int output_size)
{
    layers.emplace_back(input_size, output_size);
}

matrix2DGPU<float> neuralNetworkGPU::forwardGPU(
    const matrix2DGPU<float> &input,
    std::vector<matrix2DGPU<float>> &layer_outputs) const
{
    layer_outputs.clear();
    layer_outputs.push_back(input);

    matrix2DGPU<float> current = input;

    for (size_t l = 0; l < layers.size(); ++l)
    {
        current = std::move(layers[l].forward(current));

        if (l == layers.size() - 1)
            current = std::move(softmax(current));

        layer_outputs.push_back(current);
    }

    return current;
}

void neuralNetworkGPU::backwardGPU(
    const matrix2DGPU<float> &target,
    const std::vector<matrix2DGPU<float>> &layer_outputs)
{
    int L = static_cast<int>(layers.size());

    matrix2DGPU<float> delta = layer_outputs[L] - target;

    for (int l = L - 1; l >= 0; --l)
    {
        const matrix2DGPU<float> &prev_output = layer_outputs[l];
        matrix2DGPU<float> new_delta;

        if (l > 0)
            new_delta = computeNewDeltaGPU(prev_output, delta, layers[l]);

        computeWeights(layers[l], delta, prev_output);

        if (l > 0)
            delta = new_delta;
    }
}

matrix2DGPU<float> neuralNetworkGPU::predictGPU(
    const matrix2DGPU<float> &input) const
{
    std::vector<matrix2DGPU<float>> dummy;
    return forwardGPU(input, dummy);
}

float neuralNetworkGPU::cross_entropy_loss(
    const std::vector<float> &output,
    int label)
{
    const float epsilon = 1e-9f;
    return -std::log(output[label] + epsilon);
}

void neuralNetworkGPU::train(
    const mnist_images &dataset,
    int epochs,
    int batch_size)
{
    size_t num_samples = dataset.size();
    std::vector<size_t> indices(num_samples);

    for (size_t i = 0; i < num_samples; i++)
        indices[i] = i;

    std::mt19937 rng(0);

    for (int e = 0; e < epochs; ++e)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        float epoch_loss = 0.0f;

        for (size_t start = 0; start < num_samples; start += batch_size)
        {
            size_t end = std::min(start + batch_size, num_samples);

            for (size_t idx = start; idx < end; ++idx)
            {
                size_t i = indices[idx];
                const auto &img = dataset.getImage(i);

                std::vector<float> input_vec(img.size());
                for (size_t p = 0; p < img.size(); ++p)
                    input_vec[p] = img[p].r / 255.0f;

                matrix2DGPU<float> input(input_vec.size(), 1);
                input.assign(input_vec.data(), input_vec.size(), 1);

                int label = dataset.getLabel(i);

                std::vector<matrix2DGPU<float>> layer_outputs;
                matrix2DGPU<float> output = forwardGPU(input, layer_outputs);

                auto host_output = output.to_host();
                epoch_loss += cross_entropy_loss(host_output, label);

                std::vector<float> target_vec(host_output.size(), 0.0f);
                target_vec[label] = 1.0f;

                matrix2DGPU<float> target(target_vec.size(), 1);
                target.assign(target_vec.data(), target_vec.size(), 1);

                backwardGPU(target, layer_outputs);
            }
        }

        loss = epoch_loss / num_samples;
        std::cout << "Epoch " << e + 1 << "/" << epochs
                  << ", Loss: " << loss << "\n";
    }
}

float neuralNetworkGPU::evaluate(const mnist_images &dataset) const
{
    size_t correct = 0;
    size_t num_samples = dataset.size();

    for (size_t i = 0; i < num_samples; ++i)
    {
        const auto &img = dataset.getImage(i);

        std::vector<float> input_vec(img.size());
        for (size_t p = 0; p < img.size(); ++p)
            input_vec[p] = img[p].r / 255.0f;

        matrix2DGPU<float> input(input_vec.size(), 1);
        input.assign(input_vec.data(), input_vec.size(), 1);

        auto output = predictGPU(input);
        auto host_output = output.to_host();

        size_t predicted = 0;
        float max_val = host_output[0];

        for (size_t j = 1; j < host_output.size(); ++j)
        {
            if (host_output[j] > max_val)
            {
                max_val = host_output[j];
                predicted = j;
            }
        }

        if (static_cast<int>(predicted) == dataset.getLabel(i))
            correct++;
    }

    return float(correct) / num_samples;
}

matrix2DGPU<float> neuralNetworkGPU::computeNewDeltaGPU(
    const matrix2DGPU<float> &prev_output,
    const matrix2DGPU<float> &old_delta,
    const neuronLayerGPU &layer)
{
    int n_inputs = prev_output.num_rows();
    int n_neurons = old_delta.num_rows();

    matrix2DGPU<float> new_delta(n_inputs, 1, 0.0f);

    computeNewDeltaKernel<<<layer.getNumBlocksInput(), layer.getThreadsPerBlock()>>>(
        layer.getWeights().device_data(),
        old_delta.device_data(),
        prev_output.device_data(),
        new_delta.device_data(),
        n_inputs,
        n_neurons);

    CUDA_CHECK(cudaDeviceSynchronize());

    return new_delta;
}

void neuralNetworkGPU::computeWeights(
    neuronLayerGPU &layer,
    const matrix2DGPU<float> &delta,
    const matrix2DGPU<float> &prev_output)
{
    int n_neurons = delta.num_rows();
    int n_inputs = prev_output.num_rows();

    updateWeightsKernel<<<layer.getNumBlocksWeights(), layer.getThreadsPerBlock()>>>(
        layer.getWeights().device_data(),
        delta.device_data(),
        prev_output.device_data(),
        n_neurons,
        n_inputs,
        learning_rate,
        lambda);

    CUDA_CHECK(cudaDeviceSynchronize());

    updateBiasesKernel<<<layer.getNumBlocksBias(), layer.getThreadsPerBlock()>>>(
        layer.getBiases().device_data(),
        delta.device_data(),
        n_neurons,
        learning_rate);

    CUDA_CHECK(cudaDeviceSynchronize());
}

matrix2DGPU<float> neuralNetworkGPU::softmax(
    const matrix2DGPU<float> &input) const
{
    matrix2DGPU<float> result = input;

    int n = input.num_rows();

    softmaxVectorKernel<<<1, 1>>>(result.device_data(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}