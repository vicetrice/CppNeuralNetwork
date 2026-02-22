#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include "neuralNetwork.hpp"

using namespace vicetriceNN;




std::vector<float> centerByBoundingBox(const std::vector<float> &img)
{
    const int size = 28;
    int minX = size, minY = size;
    int maxX = 0, maxY = 0;

    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            if (img[y * size + x] > 0.1f)
            {
                minX = std::min(minX, x);
                minY = std::min(minY, y);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
            }

    if (minX >= maxX || minY >= maxY)
        return img;

    int boxWidth = maxX - minX + 1;
    int boxHeight = maxY - minY + 1;
    int offsetX = (size - boxWidth) / 2;
    int offsetY = (size - boxHeight) / 2;

    std::vector<float> centered(size * size, 0.0f);
    for (int y = 0; y < boxHeight; y++)
        for (int x = 0; x < boxWidth; x++)
            centered[(y + offsetY) * size + (x + offsetX)] =
                img[(y + minY) * size + (x + minX)];

    return centered;
}




void saveBMP28x28(const std::string &filename, const std::vector<float> &img)
{
    const int width = 28;
    const int height = 28;
    std::ofstream file(filename, std::ios::binary);

    uint32_t fileSize = 54 + 1024 + width * height;
    uint32_t pixelOffset = 54 + 1024;

    file.put('B');
    file.put('M');
    file.write(reinterpret_cast<char *>(&fileSize), 4);

    uint32_t reserved = 0;
    file.write(reinterpret_cast<char *>(&reserved), 4);
    file.write(reinterpret_cast<char *>(&pixelOffset), 4);

    uint32_t dibSize = 40;
    file.write(reinterpret_cast<char *>(&dibSize), 4);
    file.write(reinterpret_cast<const char *>(&width), 4);
    file.write(reinterpret_cast<const char *>(&height), 4);

    uint16_t planes = 1;
    uint16_t bpp = 8;
    file.write(reinterpret_cast<char *>(&planes), 2);
    file.write(reinterpret_cast<char *>(&bpp), 2);

    uint32_t compression = 0;
    uint32_t imageSize = width * height;
    uint32_t ppm = 2835;
    uint32_t colorsUsed = 256;
    uint32_t importantColors = 0;

    file.write(reinterpret_cast<char *>(&compression), 4);
    file.write(reinterpret_cast<char *>(&imageSize), 4);
    file.write(reinterpret_cast<char *>(&ppm), 4);
    file.write(reinterpret_cast<char *>(&ppm), 4);
    file.write(reinterpret_cast<char *>(&colorsUsed), 4);
    file.write(reinterpret_cast<char *>(&importantColors), 4);

    for (int i = 0; i < 256; i++)
    {
        file.put(i);
        file.put(i);
        file.put(i);
        file.put(0);
    }

    for (int y = height - 1; y >= 0; y--)
        for (int x = 0; x < width; x++)
        {
            uint8_t value = uint8_t(std::clamp(img[y * width + x], 0.0f, 1.0f) * 255.0f);
            file.put(value);
        }

    file.close();
}







std::vector<float> rotateFlipEMNIST(const std::vector<float> &img)
{
    const int size = 28;

    
    std::vector<float> flipped(size * size, 0.0f);
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            flipped[y * size + x] = img[y * size + (size - 1 - x)];
        }
    }

    
    int rotation = 270; 

    std::vector<float> rotated(size * size, 0.0f);

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            switch (rotation)
            {
            case 0: 
                rotated[y * size + x] = flipped[y * size + x];
                break;
            case 90: 
                rotated[x * size + (size - 1 - y)] = flipped[y * size + x];
                break;
            case 180: 
                rotated[(size - 1 - y) * size + (size - 1 - x)] = flipped[y * size + x];
                break;
            case 270: 
                rotated[(size - 1 - x) * size + y] = flipped[y * size + x];
                break;
            default:
                rotated[y * size + x] = flipped[y * size + x]; 
            }
        }
    }

    
    std::vector<float> flipped2(size * size, 0.0f);
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            flipped2[y * size + x] = rotated[y * size + (size - 1 - x)];
        }
    }

    return flipped2;
}




std::vector<float> loadBMP28x28BW_EMNIST(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "No se pudo abrir BMP\n";
        return {};
    }

    uint16_t bfType;
    file.read(reinterpret_cast<char *>(&bfType), 2);
    if (bfType != 0x4D42)
    {
        std::cerr << "No es BMP valido\n";
        return {};
    }

    uint32_t pixelOffset;
    file.seekg(10);
    file.read(reinterpret_cast<char *>(&pixelOffset), 4);

    int32_t width, height;
    file.seekg(18);
    file.read(reinterpret_cast<char *>(&width), 4);
    file.read(reinterpret_cast<char *>(&height), 4);

    uint16_t bpp;
    file.seekg(28);
    file.read(reinterpret_cast<char *>(&bpp), 2);
    if (bpp != 8 && bpp != 24)
    {
        std::cerr << "Solo BMP 8 o 24 bits soportado\n";
        return {};
    }

    file.seekg(pixelOffset);

    std::vector<uint8_t> data;
    if (bpp == 8)
    {
        data.resize(width * height);
        file.read(reinterpret_cast<char *>(data.data()), data.size());
    }
    else
    {
        data.resize(width * height * 3);
        file.read(reinterpret_cast<char *>(data.data()), data.size());

        std::vector<uint8_t> gray(width * height);
        for (int i = 0; i < width * height; i++)
        {
            int idx = i * 3;
            uint8_t r = data[idx + 2], g = data[idx + 1], b = data[idx + 0];
            gray[i] = uint8_t(0.299f * r + 0.587f * g + 0.114f * b);
        }
        data = gray;
    }

    
    std::vector<float> resized(28 * 28, 0.0f);
    float scaleX = float(width) / 28.0f, scaleY = float(height) / 28.0f;
    for (int y = 0; y < 28; y++)
        for (int x = 0; x < 28; x++)
        {
            int srcX = std::min(int(x * scaleX), width - 1);
            int srcY = std::min(int(y * scaleY), height - 1);
            resized[y * 28 + x] = float(data[srcY * width + srcX]) / 255.0f;
        }

    
    auto rotated = rotateFlipEMNIST(resized);

    
    auto centered = centerByBoundingBox(rotated);

    
    saveBMP28x28("debug_emnist.bmp", centered);

    return centered;
}




int main()
{
    int input_size = 28 * 28;
    int hidden1 = 256;
    int hidden2 = 128;
    int output_size = 47; 

    neuralNetwork net;
    net.addLayer(input_size, hidden1);
    net.addLayer(hidden1, hidden2);
    net.addLayer(hidden2, output_size);

    const std::string weights_path = "weights_EMNIST/emnist_balanced_weights.bin";
    if (!net.loadWeights(weights_path))
    {
        std::cerr << "No se encontraron pesos entrenados.\n";
        return 1;
    }
    std::cout << "Pesos cargados correctamente.\n";

    net.setLearningRate(0.0001f);
    net.setLambda(0.0f);

    static const std::string emnist_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";

    while (true)
    {
        std::cout << "\nIngrese path a BMP ('exit' para salir): ";
        std::string bmp_path;
        std::cin >> bmp_path;
        if (bmp_path == "exit")
            break;

        auto input_vec = loadBMP28x28BW_EMNIST(bmp_path);
        if (input_vec.empty())
            continue;

        auto output_probs = net.predict(input_vec);

        size_t predicted = 0;
        float max_val = output_probs[0];
        for (size_t i = 1; i < output_probs.size(); i++)
        {
            if (output_probs[i] > max_val)
            {
                max_val = output_probs[i];
                predicted = i;
            }
        }

        std::cout << "\nProbabilidades por caracter:\n";
        for (size_t i = 0; i < output_probs.size(); i++)
        {
            char c = (i < emnist_labels.size()) ? emnist_labels[i] : '?';
            std::cout << c << " -> " << output_probs[i] << "\n";
        }

        if (predicted < emnist_labels.size())
            std::cout << "\nCaracter predicho: " << emnist_labels[predicted] << "\n";
        else
            std::cout << "\nPrediccion invalida\n";
    }

    return 0;
}
