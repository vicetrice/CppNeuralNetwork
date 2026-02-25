#pragma once
#include <vector>
#include <string>

namespace vicebmpUtils {

struct BMPImage {
    int width;
    int height;
    std::vector<float> pixels;
};

BMPImage loadBMP(const std::string &filename);
BMPImage resize28x28(const BMPImage &src);
BMPImage centerImage(const BMPImage &img);
void saveBMP28x28(const std::string &filename, const BMPImage &img);
BMPImage preprocessBMP(const std::string &filename);

}
