#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "utils.cuh"

using namespace std;

__global__ void duplicateImageWarhol(unsigned char* in, unsigned char* out, std::size_t cols, std::size_t rows, int duplicationNumber = 4)
{
	auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidx < cols && tidy < rows)
	{
		int square = cuSquare(duplicationNumber);

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


__global__ void colorizeImageWarhol(unsigned char* in, unsigned char* out, std::size_t cols, std::size_t rows, int duplicationNumber = 4)
{
	//auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	//auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

	//if (tidx < cols && tidy < rows)
	//{
	//	int square = cuSquare(duplicationNumber);

	//	//For each line
	//	for (int i = 0; i < square; i++) {

	//		auto index = 3 * (tidy * cols / square / 2 + tidx / square) + 3 * cols * rows / square * i;

	//		if (out[index] == 0 && out[index + 1] == 0 && out[index + 2] == 0) {
	//			out[index] = in[3 * (tidy * cols + tidx)];
	//			out[index + 1] = in[3 * (tidy * cols + tidx) + 1];
	//			out[index + 2] = in[3 * (tidy * cols + tidx) + 2];
	//		}
	//	}
	//}
}


void andyWarhol()
{
	cv::Mat m_in = cv::imread("ecureuil.jpg", cv::IMREAD_UNCHANGED);

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

	cudaMemcpy(base_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);

	dim3 block(16, 64);
	dim3 grid((cols - 1) / block.x + 1, (rows - 1) / block.y + 1); //(4,4)

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;

	duplicateImageWarhol << <grid, block >> > (base_d, duplicated_d, cols, rows);
	colorizeImageWarhol << <grid, block >> > (duplicated_d, out_d, cols, rows);

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