#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "utils.cuh"
#include <random>

using namespace std;

__global__ void duplicateImageWarhol(unsigned char* in, unsigned char* out, std::size_t cols, std::size_t rows, int duplicationNumber)
{
	auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int square;

	if (tidx < cols && tidy < rows)
	{
		square = cuSquare(duplicationNumber);

		//For each line
		for (int i = 0; i < square; i++) {

			auto index = 3 * (tidy * cols / square + tidx / square) + 3 * cols * rows / square * i;

			if (out[index] == 0 && out[index + 1] == 0 && out[index + 2] == 0) {
				out[index] = in[3 * (tidy * cols + tidx)];
				out[index + 1] = in[3 * (tidy * cols + tidx) + 1];
				out[index + 2] = in[3 * (tidy * cols + tidx) + 2];
			}
		}
	}
}


__global__ void colorizeImageWarhol(unsigned char* in, unsigned char* out, std::size_t cols, std::size_t rows, unsigned int* caseType, int duplicationNumber)
{
	auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

	const int effectNumber = 6;

	__shared__ int square;

	if (tidx < cols && tidy < rows)
	{
		square = cuSquare(duplicationNumber);

		auto index = 3 * (tidy * cols + tidx);

		//For each case
		for (int i = 1; i < square + 1; i++) {
			for (int j = 1; j < square + 1; j++) {
				if (tidx < cols / square * i && tidy < rows / square * j) {
					if (out[index] == 0 && out[index + 1] == 0 && out[index + 2] == 0) {

						//Without first channel
						if (caseType[(i - 1) * square + j - 1] % effectNumber == 0) {
							//out[index] = in[index];
							out[index + 1] = in[index + 1];
							out[index + 2] = in[index + 2];
						}
						//Without second channel
						else if (caseType[(i - 1) * square + j - 1] % effectNumber == 1) {
							out[index] = in[index];
							//out[index + 1] = in[index + 1];
							out[index + 2] = in[index + 2];
						}
						//Without third channel
						else if (caseType[(i - 1) * square + j - 1] % effectNumber == 2) {
							out[index] = in[index];
							out[index + 1] = in[index + 1];
							//out[index + 2] = in[index + 2];
						}
						//Grayscale
						else if (caseType[(i - 1) * square + j - 1] % effectNumber == 3) {
							out[index] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
							out[index + 1] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
							out[index + 2] = (307 * in[index] + 604 * in[index + 1] + 113 * in[index + 2]) / 1024;
						}
						//Invert
						else if (caseType[(i - 1) * square + j - 1] % effectNumber == 4) {
							out[index] = in[index] ^ 0x00ffffff;
							out[index + 1] = in[index + 1] ^ 0x00ffffff;
							out[index + 2] = in[index + 2] ^ 0x00ffffff;
						}
						else if (caseType[(i - 1) * square + j - 1] % effectNumber == 5) {
							out[index] = in[index];
							out[index + 1] = in[index + 1];
							out[index + 2] = in[index + 2];
						}
						//If you add effect here, please update the effect number (const int)
					}
				}
			}
		}
	}
}


void andyWarhol(const int duplicationNumber = 4)
{
	srand(time(0));

	cv::Mat m_in = cv::imread("photo.jpg", cv::IMREAD_UNCHANGED);

	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > duplicated(3 * rows * cols);
	std::vector< unsigned char > out(3 * rows * cols);
	cv::Mat image_duplicated(rows, cols, CV_8UC3, duplicated.data());
	cv::Mat image_out(rows, cols, CV_8UC3, out.data());

	unsigned char* base_d;
	unsigned char* duplicated_d;
	unsigned char* out_d;
	unsigned int* caseType_d;

	//Generate unique int for each case of the array
	std::vector<unsigned int> caseType(duplicationNumber);
	std::iota(caseType.begin(), caseType.end(), 0);
	std::shuffle(caseType.begin(), caseType.end(), default_random_engine(0));
	//std::for_each(caseType.begin(), caseType.end(), [](unsigned int& o) {cout << o << " "; });
	cout << endl;

	#pragma region Event & Timer

	auto start = std::chrono::system_clock::now();
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaEventRecord(cudaStart);

	#pragma endregion

	cudaMalloc(&base_d, 3 * rows * cols);
	cudaMalloc(&duplicated_d, 3 * rows * cols);
	cudaMalloc(&out_d, 3 * rows * cols);

	cudaMalloc(&caseType_d, sizeof(unsigned int) * duplicationNumber);

	cudaMemcpy(base_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy(caseType_d, caseType.data() , sizeof(unsigned int) * duplicationNumber, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((cols - 1) / block.x + 1, (rows - 1) / block.y + 1); //(4,4)

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;

	duplicateImageWarhol << <grid, block >> > (base_d, duplicated_d, cols, rows, duplicationNumber);
	colorizeImageWarhol << <grid, block >> > (duplicated_d, out_d, cols, rows, caseType_d, duplicationNumber);

	cudaMemcpy(duplicated.data(), duplicated_d, 3 * rows * cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(out.data(), out_d, 3 * rows * cols, cudaMemcpyDeviceToHost);

	#pragma region Event & Timer


	cudaEventRecord(cudaStop);
	cudaEventSynchronize(cudaStop);
	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
	std::cout << "Temps kernel: " << elapsedTime << std::endl;
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);
	auto err = cudaGetLastError();

	std::cout << "Erreur: " << err << std::endl;

	std::cout << ms << " ms" << std::endl;

	#pragma endregion

	cv::imwrite("duplicated.jpg", image_duplicated);
	cv::imwrite("out.jpg", image_out);

	cudaFree(base_d);
	cudaFree(duplicated_d);
	cudaFree(out_d);
}