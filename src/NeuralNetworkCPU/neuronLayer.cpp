#include "neuronLayer.hpp"
#include <random>

namespace vicetriceNN
{

    neuronLayer::neuronLayer(int in_size, int out_size)
        : input_size(in_size),
          output_size(out_size),
          weights(in_size * out_size),
          biases(out_size, 0.0f)
    {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_size));

        for (auto &w : weights)
            w = dist(rng);
    }

    std::vector<float> neuronLayer::forward(const std::vector<float> &input) const
    {
        std::vector<float> output(output_size);

        for (int o = 0; o < output_size; ++o)
        {
            float sum = biases[o];

            const int row_offset = o * input_size;

            for (int i = 0; i < input_size; ++i)
            {
                sum += weights[row_offset + i] * input[i];
            }

            output[o] = relu(sum);
        }

        return output;
    }

}