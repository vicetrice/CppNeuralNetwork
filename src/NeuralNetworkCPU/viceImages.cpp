#include "viceImages.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace vicebmpUtils
{

    BMPImage loadBMP(const std::string &filename)
    {
        BMPImage img{};
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "No se pudo abrir BMP: " << filename << "\n";
            return img;
        }

        uint16_t bfType;
        file.read(reinterpret_cast<char *>(&bfType), 2);
        if (bfType != 0x4D42)
        {
            std::cerr << "No es BMP válido\n";
            return img;
        }

        uint32_t pixelOffset;
        file.seekg(10);
        file.read(reinterpret_cast<char *>(&pixelOffset), 4);

        int32_t width, height;
        file.seekg(18);
        file.read(reinterpret_cast<char *>(&width), 4);
        file.read(reinterpret_cast<char *>(&height), 4);
        img.width = width;
        img.height = height;

        uint16_t bpp;
        file.seekg(28);
        file.read(reinterpret_cast<char *>(&bpp), 2);

        if (bpp != 8 && bpp != 24)
        {
            std::cerr << "Solo BMP 8 o 24 bits soportado\n";
            return img;
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
                uint8_t r = data[idx + 2];
                uint8_t g = data[idx + 1];
                uint8_t b = data[idx + 0];
                gray[i] = uint8_t(0.299 * r + 0.587 * g + 0.114 * b);
            }
            data = gray;
        }

        std::vector<float> pixels(width * height);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixels[y * width + x] = float(data[(height - 1 - y) * width + x]) / 255.0f;

        img.pixels = pixels;
        return img;
    }

    BMPImage resize28x28(const BMPImage &src)
    {
        BMPImage out;
        out.width = 28;
        out.height = 28;
        out.pixels.resize(28 * 28, 0.0f);

        float scaleX = float(src.width) / 28.0f;
        float scaleY = float(src.height) / 28.0f;

        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
            {
                int srcX = std::min(int(x * scaleX), src.width - 1);
                int srcY = std::min(int(y * scaleY), src.height - 1);
                out.pixels[y * 28 + x] = src.pixels[srcY * src.width + srcX];
            }

        return out;
    }

    BMPImage centerImage(const BMPImage &img)
    {
        const int size = img.width;
        int minX = size, minY = size, maxX = 0, maxY = 0;

        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                if (img.pixels[y * size + x] > 0.1f)
                {
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                }

        if (minX >= maxX || minY >= maxY)
            return img;

        int boxW = maxX - minX + 1;
        int boxH = maxY - minY + 1;
        int offsetX = (size - boxW) / 2;
        int offsetY = (size - boxH) / 2;

        BMPImage out;
        out.width = size;
        out.height = size;
        out.pixels.resize(size * size, 0.0f);

        for (int y = 0; y < boxH; y++)
            for (int x = 0; x < boxW; x++)
                out.pixels[(y + offsetY) * size + (x + offsetX)] =
                    img.pixels[(y + minY) * size + (x + minX)];

        return out;
    }

    void saveBMP28x28(const std::string &filename, const BMPImage &img)
    {
        std::ofstream file(filename, std::ios::binary);
        int width = img.width, height = img.height;
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
        file.write(reinterpret_cast<char *>(&width), 4);
        file.write(reinterpret_cast<char *>(&height), 4);

        uint16_t planes = 1, bpp = 8;
        file.write(reinterpret_cast<char *>(&planes), 2);
        file.write(reinterpret_cast<char *>(&bpp), 2);

        uint32_t compression = 0, imageSize = width * height;
        uint32_t ppm = 2835, colorsUsed = 256, importantColors = 0;
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
                uint8_t v = uint8_t(std::clamp(img.pixels[y * width + x], 0.0f, 1.0f) * 255.0f);
                file.put(v);
            }
        file.close();
    }

    BMPImage preprocessBMP(const std::string &filename)
    {
        auto img = loadBMP(filename);
        if (img.pixels.empty())
            return img;
        auto resized = resize28x28(img);
        auto centered = centerImage(resized);
        saveBMP28x28("debug_centered.bmp", centered);
        return centered;
    }

}
