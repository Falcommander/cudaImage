#pragma once
#include <string>
void grayscaleStains(const std::string file);
void grayscaleCase(const std::string file, size_t casePerLine = 7);
void grayscaleWithoutOne(const std::string file);
void colored_sobel(const std::string file);
void convolution_matrix(const std::string file);
void launch();
void choiceGrayscaleCase(const std::string file, const int choiceProc);
inline bool file_exist(const std::string& file);
void andyWarhol(const std::string file, const int duplicationNumber = 4);
void choiceAndyWarhol(const std::string file, int choiceProc);

void grayscaleStainsCPU(const std::string file);
void grayscaleCaseCPU(const std:: string file, const size_t casePerLine = 7);
void grayscaleWithoutOneCPU(const std::string file);
void andyWarholCPU(const std::string file, const int duplicationNumber = 4);
void colored_sobelCPU(const std::string file);
void convolution_matrixCPU(const std::string file);