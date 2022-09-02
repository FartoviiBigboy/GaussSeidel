
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <omp.h>
#include <string>

const int threadsPerBlock = 256;
int blocksPerGrid;

using namespace std;



int imin(int a, int b) {
	return (a < b ? a : b);
}


double getRandomNumber(double min, double max)
{
	static const double fraction = 1.0 / ((double)(RAND_MAX)+1.0);
	return rand() * fraction * (max - min + 1.0) + min;
}




__global__ void tempSum(const double* matrixA, const double* vectorB, double* vectorX, double* vectorSum, int* N, int* numberOfIteration) {

	int tempNumberOfIteration = (*numberOfIteration);

	int tid;

	tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < (*N)) {

		if (tempNumberOfIteration == 0) {
			vectorSum[tid] = vectorX[tid];
			vectorX[tid] = vectorB[tid] / matrixA[tid * (*N) + tid];

		}

		if (tempNumberOfIteration == 1) {
			for (int i = tid + 1; i < (*N); i++) {
				vectorX[tid] -= vectorSum[i] * matrixA[tid * (*N) + i] / matrixA[tid * (*N) + tid];

			}

		}

		if ((tid > (tempNumberOfIteration - 2)) && (tempNumberOfIteration > 1)) {
			vectorX[tid] -= vectorX[tempNumberOfIteration - 2] * matrixA[tid * (*N) + tempNumberOfIteration - 2] / matrixA[tid * (*N) + tid];
		}


		tid += blockDim.x * gridDim.x;
	}

	tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0) {
		if ((*numberOfIteration) != (*N)) {
			(*numberOfIteration) = (*numberOfIteration) + 1;
		}
		else {
			(*numberOfIteration) = 0;
		}
	}

}





int main(int argc, char* argv[])
{

	srand(time(NULL));

	ifstream input;

	string fileName;

	int menuKey;

	int N;
	int* d_N;

	double* matrixA;
	double* vectorB;
	double* vectorX;
	double* vectorSum;

	double* d_matrixA;
	double* d_vectorB;
	double* d_vectorX;
	double* d_vectorSum;

	int numberOfIteration;
	int* d_numberOfIteration;

	double Eps;

	double time1;
	double time2;

	double* multipleAnswer;


	cout << "Choose epsilon:" << endl;
	cin >> Eps;

	cout << "Choose mode for loading data: " << endl << "1 - generate" << endl << "2 - from file" << endl;
	cin >> menuKey;

	if (menuKey == 1) {
		double min;
		double max;
		cout << "Enter N: " << endl;
		cin >> N;
		cout << "Enter min and max separated by space: " << endl;
		cin >> min >> max;

		matrixA = new double[N * N];
		vectorB = new double[N];
		vectorX = new double[N];
		vectorSum = new double[N];


		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N + 1; j++) {
				if (j == N) {
					vectorB[i] = getRandomNumber(min, max);
				}
				else {
					matrixA[i * N + j] = getRandomNumber(min, max);
				}
			}
		}

		double megaSum = 0;

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (i != j) {
					megaSum += fabs(matrixA[i * N + j]);
				}
			}
		}

		for (int i = 0; i < N; i++) {
			matrixA[i * N + i] < 0 ? matrixA[i * N + i] -= megaSum * 1 : matrixA[i * N + i] += megaSum * 1;
		}

	}
	if (menuKey == 2) {
		cin.get();
		cout << "Choose file name for matrix:" << endl;
		getline(cin, fileName);


		input.open(fileName.c_str(), ios_base::in);
		input >> N;

		matrixA = new double[N * N];
		vectorB = new double[N];
		vectorX = new double[N];
		vectorSum = new double[N];

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N + 1; j++) {
				if (j == N) {
					input >> vectorB[i];
				}
				else {
					input >> matrixA[i * N + j];
				}
			}
		}
		input.close();
	}



	for (int i = 0; i < N; i++) {
		vectorX[i] = 0;
		vectorSum[i] = 0;
	}

	numberOfIteration = 0;

	blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);



	cudaMalloc((void**)&d_matrixA, N * N * sizeof(double));
	cudaMalloc((void**)&d_vectorB, N * sizeof(double));
	cudaMalloc((void**)&d_vectorX, N * sizeof(double));
	cudaMalloc((void**)&d_vectorSum, N * sizeof(double));

	cudaMalloc((void**)&d_N, sizeof(int));

	cudaMalloc((void**)&d_numberOfIteration, sizeof(int));


	cudaMemcpy(d_matrixA, matrixA, N * N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectorB, vectorB, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectorX, vectorX, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectorSum, vectorSum, N * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_numberOfIteration, &numberOfIteration, sizeof(int), cudaMemcpyHostToDevice);


	bool check = true;

	int countOfOperation = 0;

	time1 = omp_get_wtime();

	while ((check != false) && (countOfOperation <= 100)) {
		check = false;
		for (int i = 0; i < N + 1; i++) {
			tempSum << <blocksPerGrid, threadsPerBlock >> > (d_matrixA, d_vectorB, d_vectorX, d_vectorSum, d_N, d_numberOfIteration);
		}
		cudaMemcpy(vectorX, d_vectorX, N * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(vectorSum, d_vectorSum, N * sizeof(double), cudaMemcpyDeviceToHost);


		countOfOperation++;
		for (int i = 0; i < N; i++) {
			if (fabs(vectorX[i] - vectorSum[i]) > Eps) {
				check = true;
				break;
			}
		}
		if (check == true) {
			continue;
		}
		else {
			break;
		}


	}

	time1 = omp_get_wtime() - time1;

	multipleAnswer = new double[N];

	for (int i = 0; i < N; i++) {
		multipleAnswer[i] = vectorX[i];
	}


	cudaFree(d_matrixA);
	cudaFree(d_vectorB);
	cudaFree(d_vectorX);

	cudaFree(d_N);
	cudaFree(d_numberOfIteration);

	cout << countOfOperation << endl;

	for (int i = 0; i < N; i++) {
		vectorX[i] = 0;
	}

	double* tempVec = new double[N];
	for (int i = 0; i < N; i++) {
		tempVec[i] = 0;
	}



	check = true;

	countOfOperation = 0;

	time2 = omp_get_wtime();

	while ((check != false) && (countOfOperation <= 100)) {
		check = false;

		for (int j = 0; j < N; j++) {
			vectorSum[j] = vectorX[j];
			vectorX[j] = vectorB[j] / matrixA[j * N + j];
			for (int k = 0; k < N; k++) {
				if (k == j) {
					continue;
				}
				vectorX[j] = vectorX[j] - ((matrixA[j * N + k] / matrixA[j * N + j]) * tempVec[k]);
				tempVec[j] = vectorX[j];
			}
		}

		countOfOperation++;
		for (int i = 0; i < N; i++) {
			if (fabs(vectorX[i] - vectorSum[i]) > Eps) {
				check = true;
				break;
			}
		}
		if (check == true) {
			continue;
		}
		else {
			break;
		}


	}

	time2 = omp_get_wtime() - time2;

	cout << countOfOperation << endl;

	cout << endl;

	check = false;

	for (int i = 0; i < N; i++) {
		if (fabs(vectorX[i] - multipleAnswer[i]) > Eps) {
			check = true;
			break;
		}
	}

	if (!check) {
		cout << "Values are equal!" << endl;
	}
	else {
		cout << "Values are not equal!" << endl;
	}

	/*for (int j = 0; j < N; j += 100) {
		double sum = 0;
		for (int i = 0; i < N; i++) {
			sum += matrixA[j * N + i] * vectorX[i];
		}
		cout << sum << " " << vectorB[j] << endl;
	}*/


	cout << "Time for multiple: " << time1 << endl;
	cout << "Time for single: " << time2 << endl;


	ofstream output;

	cout << "Do you want to save CLAY?: " << endl << "1 - yes" << endl << "2 - no" << endl;
	cin >> menuKey;
	if (menuKey == 1) {
		cout << "Choose file name for CLAY: " << endl;
		cin.get();
		getline(cin, fileName);
		output.open(fileName.c_str(), ios_base::out);
		output << N << endl;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N + 1; j++) {
				if (j == N) {
					output << vectorB[i] << endl;
				}
				else {
					output << matrixA[i * N + j] << " ";
				}
			}
		}
		output.close();


	}
	cout << "Do you want to save multiple and single answers: " << endl << "1 - yes" << endl << "2 - no" << endl;
	cin >> menuKey;
	if (menuKey == 1) {
		cout << "Choose file name for multiple: " << endl;
		cin.get();
		getline(cin, fileName);
		output.open(fileName.c_str(), ios_base::out);
		output << N << endl;
		for (int i = 0; i < N; i++) {
			output << multipleAnswer[i] << endl;
		}
		output.close();

		cout << "Choose file name for single: " << endl;
		getline(cin, fileName);
		output.open(fileName.c_str(), ios_base::out);
		output << N << endl;
		for (int i = 0; i < N; i++) {
			output << vectorX[i] << endl;
		}
		output.close();
	}

	return 0;
}
