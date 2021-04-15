#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;

void greyscaleWithoutOneCPUKernel(unsigned char* rgb, unsigned char* g, const size_t cols, const size_t rows)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			if (50 <= rgb[3 * (i * cols + j)] && rgb[3 * (i * cols + j)] <= 200
				&& 20 <= rgb[3 * (i * cols + j) + 1] && rgb[3 * (i * cols + j) + 1] <= 180
				&& 10 <= rgb[3 * (i * cols + j) + 2] && rgb[3 * (i * cols + j) + 2] <= 160) {

				g[3 * (i * cols + j)] = (
					307 * rgb[3 * (i * cols + j)]
					+ 604 * rgb[3 * (i * cols + j) + 1]
					+ 113 * rgb[3 * (i * cols + j) + 2]
					) / 1024;
				g[3 * (i * cols + j) + 1] = (
					307 * rgb[3 * (i * cols + j)]
					+ 604 * rgb[3 * (i * cols + j) + 1]
					+ 113 * rgb[3 * (i * cols + j) + 2]
					) / 1024;
				g[3 * (i * cols + j) + 2] = (
					307 * rgb[3 * (i * cols + j)]
					+ 604 * rgb[3 * (i * cols + j) + 1]
					+ 113 * rgb[3 * (i * cols + j) + 2]
					) / 1024;
			}

			if (g[3 * (i * cols + j)] == 0 && g[3 * (i * cols + j) + 1] == 0 && g[3 * (i * cols + j) + 2] == 0) {
				g[3 * (i * cols + j)] = rgb[3 * (i * cols + j)];
				g[3 * (i * cols + j) + 1] = rgb[3 * (i * cols + j) + 1];
				g[3 * (i * cols + j) + 2] = rgb[3 * (i * cols + j) + 2];
			}
		}
	}
}


void grayscaleWithoutOneCPU()
{
	cv::Mat m_in = cv::imread("ecureuil.jpg", cv::IMREAD_UNCHANGED);

	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	std::vector< unsigned char > g(3 * rows * cols);
	cv::Mat m_out(rows, cols, CV_8UC3, g.data());

	cout << "rows : " << rows << endl;
	cout << "cols : " << cols << endl;

	auto start = std::chrono::system_clock::now();

	greyscaleWithoutOneCPUKernel(rgb, g.data(), cols, rows);

	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("out.jpg", m_out);
}