#include <iostream>
#include "mnist_images.hpp"
#include "neuralNetwork.hpp"

int main()
{

    vicetriceNN::mnist_images test_dataset;
    if (!test_dataset.load("../archive/t10k-images.idx3-ubyte",
                           "../archive/t10k-labels.idx1-ubyte"))
    {
        std::cerr << "Error cargando MNIST t10k (prueba)\n";
        return 1;
    }

    int input_size = test_dataset.getRows() * test_dataset.getCols();
    int hidden1 = 128;
    int hidden2 = 64;
    int output_size = 10;

    vicetriceNN::neuralNetwork net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);

    if (!net.loadWeights("weights_MNIST/weights.bin"))
    {
        std::cerr << "No se encontraron pesos guardados (weights.bin)\n";
        return 1;
    }

    std::cout << "Pesos cargados correctamente desde weights.bin\n";

    float accuracy = net.evaluate(test_dataset);

    std::cout << "Precision en el conjunto de prueba t10k: "
              << accuracy * 100.0f << "%\n";

    return 0;
}
