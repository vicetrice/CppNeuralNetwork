#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "mnist_images.hpp"
#include "neuralNetwork.hpp"

int main()
{
    vicetriceNN::mnist_images dataset;
    if (!dataset.load("../archive/train-images.idx3-ubyte",
                      "../archive/train-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST\n";
        return 1;
    }

    vicetriceNN::mnist_images test;
    if (!test.load("../archive/t10k-images.idx3-ubyte",
                      "../archive/t10k-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST\n";
        return 1;
    }

    int input_size = dataset.getRows() * dataset.getCols();
    int hidden1 = 16;
    int hidden2 = 16;
    int output_size = 10;

    float learning_rate = 0.01f;

    vicetriceNN::neuralNetwork net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);
    net.setLearningRate(learning_rate);

    std::cout << "No se encontraron pesos guardados.\n";
    std::cout << "Entrenando desde cero...\n";

    int epochs = 5;
    int batch_size = 32;

    net.train(dataset, epochs, batch_size);

    

    std::cout << "acc: " << net.evaluate(test) * 100.0f << "%\n";

    

    return 0;
}
