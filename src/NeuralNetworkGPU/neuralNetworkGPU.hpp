#pragma once
#include <vector>
#include <string>
#include "neuronLayerGPU.hpp"
#include "matrix2DGPU.hpp"
#include "mnist_images.hpp"

namespace vicetriceNN
{
    class neuralNetworkGPU
    {
    private:
        std::vector<neuronLayerGPU> layers;
        float learning_rate = 0.001f;
        float lambda = 0.0f;
        float loss;

        static float cross_entropy_loss(const std::vector<float> &output, int label);

        viceCudaMath::matrix2DGPU<float> computeNewDeltaGPU(const viceCudaMath::matrix2DGPU<float> &prev_output, const viceCudaMath::matrix2DGPU<float> &old_delta, const neuronLayerGPU &layer);
        void computeWeights(neuronLayerGPU &layer, const viceCudaMath::matrix2DGPU<float> &delta, const viceCudaMath::matrix2DGPU<float> &prev_output);
        viceCudaMath::matrix2DGPU<float> softmax(const viceCudaMath::matrix2DGPU<float> &input) const;

    public:
        neuralNetworkGPU() = default;
        ~neuralNetworkGPU() = default;

        void addLayer(int input_size, int output_size);

        viceCudaMath::matrix2DGPU<float> forwardGPU(const viceCudaMath::matrix2DGPU<float> &input,
                                                    std::vector<viceCudaMath::matrix2DGPU<float>> &layer_outputs) const;

        void backwardGPU(const viceCudaMath::matrix2DGPU<float> &target,
                         const std::vector<viceCudaMath::matrix2DGPU<float>> &layer_outputs);

        inline void setLearningRate(float lr) { learning_rate = lr; }
        inline void setLambda(float lmbd) { lambda = lmbd; }

        void train(const mnist_images &dataset, int epochs, int batch_size);
        float evaluate(const mnist_images &dataset) const;

        viceCudaMath::matrix2DGPU<float> predictGPU(const viceCudaMath::matrix2DGPU<float> &input) const;

        inline size_t getNumLayers() const { return layers.size(); }
        inline const neuronLayerGPU &getLayer(size_t idx) const { return layers[idx]; }

        inline float getLoss() const { return loss; }
        inline float getLearningRate() const { return learning_rate; }
        inline float getLambda() const { return lambda; }
    };
}