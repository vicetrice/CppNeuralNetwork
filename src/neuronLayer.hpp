#pragma once
#include <vector>

namespace vicetriceNN
{
    class neuronLayer
    {
    private:
        int input_size;
        int output_size;

        std::vector<std::vector<float>> weights;
        std::vector<float> biases;

    public:
        neuronLayer(int in_size, int out_size);
        ~neuronLayer();

        std::vector<float> forward(const std::vector<float> &input) const;

        inline std::vector<std::vector<float>> &getWeights() { return weights; }
        inline std::vector<float> &getBiases() { return biases; }

        inline std::vector<std::vector<float>> getWeights() const { return weights; }
        inline std::vector<float> getBiases() const { return biases; }

        inline int getInputSize() const { return input_size; }
        inline int getOutputSize() const { return output_size; }

    };
}
