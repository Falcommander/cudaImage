#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;

void grayscaleCaseCPUKernel(unsigned char* rgb, unsigned char* g, const size_t cols, const size_t rows, const size_t casePerLine)
{
	float caseNumber = cols / ((int)(cols / casePerLine) + 1);

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			for (size_t k = 0; k < caseNumber; k++)
			{
				for (size_t l = 0; l < caseNumber; l++)
				{
					if (k % 2 == 0) {
						if (l % 2 == 0) {
							if (cols / caseNumber * k < j && j < cols / caseNumber * (k + 1)
								&& rows / caseNumber * l < i && i < rows / caseNumber * (l + 1))
							{
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
						}
					}
					else {
						if (l % 2 == 1) {
							if (cols / caseNumber * k < j && j < cols / caseNumber * (k + 1)
								&& rows / caseNumber * l < i && i < rows / caseNumber * (l + 1))
							{
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
						}
					}

					//Colorize all others pixels
					if (g[3 * (i * cols + j)] == 0 && g[3 * (i * cols + j) + 1] == 0 && g[3 * (i * cols + j) + 2] == 0) {
						g[3 * (i * cols + j)] = rgb[3 * (i * cols + j)];
						g[3 * (i * cols + j) + 1] = rgb[3 * (i * cols + j) + 1];
						g[3 * (i * cols + j) + 2] = rgb[3 * (i * cols + j) + 2];
					}
				}
			}
		}

	}
}

void grayscaleCaseCPU(size_t casePerLine = 7)
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

	grayscaleCaseCPUKernel(rgb, g.data(), cols, rows, casePerLine);

	auto stop = std::chrono::system_clock::now();

	auto duration = stop - start;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << ms << " ms" << std::endl;

	cv::imwrite("out.jpg", m_out);

}