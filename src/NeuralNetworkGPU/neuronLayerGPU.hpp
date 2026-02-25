#pragma once

#include "matrix2DGPU.hpp"
#include <stdexcept>

namespace vicetriceNN
{
    class neuronLayerGPU
    {
    private:
        int input_size;
        int output_size;

        viceCudaMath::matrix2DGPU<float> weights;
        viceCudaMath::matrix2DGPU<float> biases;

        int threads_per_block;
        int blocks_weights;
        int blocks_bias;
        int block_input;

    public:
        neuronLayerGPU(int in_size, int out_size);
        ~neuronLayerGPU();

        viceCudaMath::matrix2DGPU<float> forward(const viceCudaMath::matrix2DGPU<float> &input) const;

        inline viceCudaMath::matrix2DGPU<float> &getWeights() { return weights; }
        inline const viceCudaMath::matrix2DGPU<float> &getWeights() const { return weights; }

        inline viceCudaMath::matrix2DGPU<float> &getBiases() { return biases; }
        inline const viceCudaMath::matrix2DGPU<float> &getBiases() const { return biases; }

        inline int getThreadsPerBlock() const { return threads_per_block; }
        inline int getNumBlocksWeights() const { return blocks_weights; }
        inline int getNumBlocksBias() const { return blocks_bias; }
        inline int getNumBlocksInput() const { return block_input; }
        inline int getInputSize() const {return input_size;}
        inline int getOutputSize() const {return output_size;}
    };
}