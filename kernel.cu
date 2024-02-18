#include "cuda_runtime.h" //
#include "device_launch_parameters.h" //
#include <stdio.h> //
#include <stdlib.h> //
#include <time.h> // 
#include <intrin.h> //

#define N 1024
#define BLOCK_SIZE 16



//>>>>CPU<<<<//
// Изменить двумерную массив на одномерный

long double GtriangleMatrix(long double* start, int n) {
	long double det = 1;
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			double factor = start[i * n + k] / start[k * n + k];
			for (int j = k; j < n; j++) {
				start[i * n + j] -= factor * start[k * n + j];
			}
		}
	}
	for (int i = 0; i < N; i++) {
		det = det * start[i * n + i];
	}
	/*printf("\ndeterminant >>>>>>>>>>>>>>>>>>>>>>>>> %.2Lf\n", det);*/

	return det;
}

__global__ void Mykernel(long double *startMatrix, int ndiag, long double* determinant_device,int n) {
	
	long double factor = 0.00;
	
	//int x = threadIdx.x + blockDim.x * blockIdx.x; //номер нити+ количество блоков в стеке в направлении x+номер блока в котором находится нить

	int ia = blockIdx.x * blockDim.x + threadIdx.x;
	int ja = blockIdx.y * blockDim.y + threadIdx.y;



	if ((ja < n) && (ja > ia) && (ia == ndiag) && (startMatrix[ja * n + ia] != 0)) {

		if (startMatrix[ia * n + ia] == 0) {
			// Если элемент на главной диагонали равен нулю, найдем строку с ненулевым элементом и обменяем их
			for (int k = ia + 1; k < n; k++) {
				if (startMatrix[k * n + ia] != 0) {
					// Обменять строки ia и k
					for (int l = 0; l < n; l++) {
						long double temp = startMatrix[ia * n + l];
						startMatrix[ia * n + l] = startMatrix[k * n + l];
						startMatrix[k * n + l] = temp;
					}
					// Обменять знак определителя
					*determinant_device = -*determinant_device;
					break;
				}
			}
		}

		//startMatrix[ja * n + ia] = 0.0;

		factor = startMatrix[ja * n + ia] / startMatrix[ia * n + ia];

		//startMatrix[ja * n + ia] = factor;

		//нужно синхронизировать threads чтобы избавить от неверных значений!
		
		for (int k = 0; k < n; k++) {

			startMatrix[ja * n + ia + k] = startMatrix[ia * n + ia + k] * factor - startMatrix[ja * n + ia + k]; //startMatrix[ja * n + ia + k] * factor; //-startMatrix[ia * n + ia + k];
			//__syncthreads();
			//startMatrix[ja * n + ia + k] -= startMatrix[ia * n + ia + k];
		}
		
		//*determinant_device *= startMatrix[ia * n + ia];
		//*determinant_device = factor;

	}
	if ((ja == ia) && (ja < n) && (ia == ndiag)) {
		*determinant_device = *determinant_device * startMatrix[ia * n + ia];
	}

}

void FuncPrintMatrix(long double *startMatrix) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%.2Lf ", startMatrix[i * N + j]);
		}
		printf("\n");
	}
}

int main() {

	srand(time(NULL));
	//long double a[N][N] = { {7, 4, 9}, {0, 6, -3}, {4, -10 ,-4} };
	//long double a[N][N] = { {1, 2, 3}, {4, 5, 6}, {7, 8 ,9} }; // изначальная матрица
	//long double a[N][N] = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} };
	//long double b[N][N];

	/*long double** a = (long double**)malloc(N * sizeof(long double*));
	for (int i = 0; i < N; i++) {
		a[i] = (long double*)malloc(N * sizeof(long double));
		for (int j = 0; j < N; j++) {
			a[i][j] = rand();
		}
	}*/

	float milliseconds = 0;

	long double* a;		//Для матрицы, заданной динамически 
	long double* b;     //Для CPU
	long double numBytes = N * N * sizeof(long double);
	long double* adev;
	long double determinant_host = 1;
	long double* determinant_device;

	a = (long double*)malloc(sizeof(long double) * N * N);




	b = (long double*)malloc(sizeof(long double) * N * N);
	adev = (long double*)malloc(sizeof(long double) * N * N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			long double set_number = rand();
			a[i * N + j] = set_number;
			b[i * N + j] = set_number;
		}
	}


	//>>>>CPU<<<<//
	long double start, end;
	long double cpu_time_used;

	start = clock();
	long double determinant_CPU = GtriangleMatrix(b, N);
	end = clock();
	printf(">>>>>>>>>>>>>>>>>>>>>>>>>det_CPU = %.2Lf\n", determinant_CPU);

	cpu_time_used = ((end - start) * 1000.0 / CLOCKS_PER_SEC);
	printf(">>>Time elapsed for CPU: %Lf milliseconds\n\n", cpu_time_used);

	//FuncPrintMatrix(b);



	//>>>>timer_cuda<<<<//
	cudaEvent_t startTimerGPU, stopTimerGPU;
	cudaEventCreate(&startTimerGPU);
	cudaEventCreate(&stopTimerGPU);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);


	cudaEventRecord(startTimerGPU);

	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&determinant_device, sizeof(long double));

	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(determinant_device, &determinant_host, sizeof(long double), cudaMemcpyHostToDevice);

	//Mykernel << <blocks, threads >> > (adev, 0, N);	
	// Запуск для каждого столбца под главной диагональю

	for (int i = 0; i < N - 1; i++) {
		Mykernel <<< blocks, threads >>> (adev, i, determinant_device, N);
	}

	cudaEventRecord(stopTimerGPU);
	cudaEventSynchronize(stopTimerGPU);

	cudaMemcpy(a, adev, numBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(&determinant_host, determinant_device, sizeof(long double), cudaMemcpyDeviceToHost);

	cudaFree(adev);
	cudaFree(determinant_device);



	//Временно 
	/*long double value = 1;
	for (int i = 0; i < N; i++) {
		value *= a[i * N + i];
	}
	determinant_host = value;*/


	long double value = a[(N - 1) * N + (N - 1)];
	determinant_host *= value;

	//determinant_host *= a[N - 1][N - 1];		//Для статичной двумерной матрицы
	//printf(">>>>%.2Lf\n", a[N - 1][N - 1]);


	//    >>>Для проверки приведенипя к треугольному виду!!!<<<    //
	//FuncPrintMatrix(a);
	printf("\n>>>>>>>>>>>>>>>>>>>>>>>>>det = %.2Lf\n", determinant_host);
	//printf("\n");
	//FuncPrintMatrix(b);


	cudaEventElapsedTime(&milliseconds, startTimerGPU, stopTimerGPU);
	printf(">>>Time elapsed for GPU: %f milliseconds\n", milliseconds);



	cudaDeviceReset();
}