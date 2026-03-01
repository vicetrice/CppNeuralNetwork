#pragma once
#include <vector>
#include "utils.hpp"

namespace vicetriceNN
{
    class neuronLayer
    {
    private:
        int input_size;
        int output_size;

        // MATRIZ PLANA
        std::vector<float> weights;
        std::vector<float> biases;

    public:
        neuronLayer(int in_size, int out_size);
        ~neuronLayer() = default;

        std::vector<float>
        forward(const std::vector<float> &input) const;

        /* ===== ACCESS ===== */

        inline std::vector<float> &getWeights()
        {
            return weights;
        }

        inline const std::vector<float> &getWeights() const
        {
            return weights;
        }

        inline std::vector<float> &getBiases()
        {
            return biases;
        }

        inline const std::vector<float> &getBiases() const
        {
            return biases;
        }

        inline int getInputSize() const { return input_size; }
        inline int getOutputSize() const { return output_size; }

        /* acceso helper */
        inline float &W(int o, int i)
        {
            return weights[o * input_size + i];
        }

        inline float W(int o, int i) const
        {
            return weights[o * input_size + i];
        }
    };
}