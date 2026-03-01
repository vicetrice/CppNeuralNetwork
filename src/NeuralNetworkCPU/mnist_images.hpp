#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "utils.hpp"
#include <cstdlib>

namespace vicetriceNN
{

    class mnist_images
    {
    private:
        std::vector<std::vector<float>> images;
        std::vector<unsigned char> labels;
        int rows = 0, cols = 0;

#if defined(_WIN32) || defined(_MSC_VER)
#define SWAP32(x) _byteswap_ulong(x)
#else
#define SWAP32(x) __builtin_bswap32(x)
#endif

        static inline uint32_t swapEndian(uint32_t val)
        {
            return SWAP32(val);
        }

        bool loadImages(const std::string &filename);
        bool loadLabels(const std::string &filename);

    public:
        mnist_images();
        ~mnist_images();

        bool load(const std::string &images_file, const std::string &labels_file);

        inline int getRows() const { return rows; }
        inline int getCols() const { return cols; }
        inline size_t size() const { return images.size(); }

        const inline std::vector<float> &getImage(size_t idx) const { return images[idx]; }
        unsigned char inline getLabel(size_t idx) const { return labels[idx]; }

        bool saveBMP(const std::string &filename, size_t idx) const;

        

    };
}
