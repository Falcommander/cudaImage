#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;

__global__ void greyscaleWithoutOneKernel(unsigned char* rgb, unsigned char* g, const size_t cols, const size_t rows)
{
	auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
	auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidx < cols && tidy < rows)
	{
		if (50 <= rgb[3 * (tidy * cols + tidx)] && rgb[3 * (tidy * cols + tidx)] <= 200
			&& 20 <= rgb[3 * (tidy * cols + tidx) + 1] && rgb[3 * (tidy * cols + tidx) + 1] <= 180
			&& 10 <= rgb[3 * (tidy * cols + tidx) + 2] && rgb[3 * (tidy * cols + tidx) + 2] <= 160) {

			g[3 * (tidy * cols + tidx)] = (
				307 * rgb[3 * (tidy * cols + tidx)]
				+ 604 * rgb[3 * (tidy * cols + tidx) + 1]
				+ 113 * rgb[3 * (tidy * cols + tidx) + 2]
				) / 1024;
			g[3 * (tidy * cols + tidx) + 1] = (
				307 * rgb[3 * (tidy * cols + tidx)]
				+ 604 * rgb[3 * (tidy * cols + tidx) + 1]
				+ 113 * rgb[3 * (tidy * cols + tidx) + 2]
				) / 1024;
			g[3 * (tidy * cols + tidx) + 2] = (
				307 * rgb[3 * (tidy * cols + tidx)]
				+ 604 * rgb[3 * (tidy * cols + tidx) + 1]
				+ 113 * rgb[3 * (tidy * cols + tidx) + 2]
				) / 1024;
		}

		if (g[3 * (tidy * cols + tidx)] == 0 && g[3 * (tidy * cols + tidx) + 1] == 0 && g[3 * (tidy * cols + tidx) + 2] == 0) {
			g[3 * (tidy * cols + tidx)] = rgb[3 * (tidy * cols + tidx)];
			g[3 * (tidy * cols + tidx) + 1] = rgb[3 * (tidy * cols + tidx) + 1];
			g[3 * (tidy * cols + tidx) + 2] = rgb[3 * (tidy * cols + tidx) + 2];
		}
	}
}


void grayscaleWithoutOne()
{
	cv::Mat m_in = cv::imread("ecureuil.jpg", cv::IMREAD_UNCHANGED);

	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > g(3 * rows * cols);
	cv::Mat m_out(rows, cols, CV_8UC3, g.data());

	unsigned char* rgb_d;
	unsigned char* g_d;

	auto start = std::chrono::system_clock::now();
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaEventRecord(cudaStart);

	cudaMalloc(&rgb_d, 3 * rows * cols);
	cudaMalloc(&g_d, 3 * rows * cols);

	cudaMemcpy(rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((cols - 1) / block.x + 1, (rows - 1) / block.y + 1); //(4,4)

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;

	greyscaleWithoutOneKernel << <grid, block >> > (rgb_d, g_d, cols, rows);


	cudaMemcpy(g.data(), g_d, 3 * rows * cols, cudaMemcpyDeviceToHost);

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

	cv::imwrite("out.jpg", m_out);

	cudaFree(rgb_d);
	cudaFree(g_d);
}