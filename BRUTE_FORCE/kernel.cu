#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <climits>
#include <time.h>

__constant__ int EdgeTable[50];
__constant__ int VertexTable[50];

struct ResultData
{
	int thread_id;
	int color_index;
};

void Initialize_graph(int * e_t, int * v_t, int * n, int * e)
{
	//TODO read data
}

cudaError_t ColorWithCuda(int *c, int *e, int *size_c, int *size_e, bool* flag, int * result);

__device__ bool TestColoriage(int * coloriage, int size_v)
{
	bool result = true;
	int current_vertex = 0;
	while (result && current_vertex != size_v)
	{
		//get edges range
		int start_edge_index = VertexTable[current_vertex];
		int end_edge_index = VertexTable[current_vertex+1];

		for (int i = start_edge_index; i < end_edge_index; i++)
		{
			if (coloriage[current_vertex] == coloriage[EdgeTable[i]])
			{
				result = false;
				break;
			}
		}
		current_vertex += 1;
	}
	return result;
}

__global__ void BruteForceKernel(const int *n, int *output, bool * found_flag)
{
	int *colors = new int[*n];

	uint64_t blockId_grid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	uint64_t threads_per_block = blockDim.x*blockDim.y*blockDim.z;
	//numer w¹tku
	uint64_t tid = blockId_grid * threads_per_block + threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	//liczba kolorowañ
	uint64_t all_results = pow(3, *n);
	//dzielimy przedzia³ kolorowañ (funkcja numeru w¹tku i iloœci w¹tków) - ka¿dy pe³ny, ostatni resztki z dzielenia
	uint64_t threads = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;
	uint64_t colors_count = (uint64_t)(all_results / threads + 1);
	uint64_t start_key = tid * colors_count;
	uint64_t end_key = start_key + colors_count;

	while (!(*found_flag) && start_key < end_key)
	{
		uint64_t tmp_key = start_key;
		//mapowanie klucza na kolorowanie (system trójkowy)
		for (int i = 0; i < *n; i++)
		{
			colors[i] = tmp_key % 3;
			tmp_key = tmp_key / 3;
		}
		//testowanie kolorowania
		//ob³o¿yæ ogólnocudowym mutexem
		if (TestColoriage(colors, *n) && !*found_flag)
		{
			*found_flag = true;
			output = colors;
		}
		//kolejny klucz
		start_key += 1;
	}
}

int main()
{
	int *edge_table, *vertex_table;
	int *n, *e;
	int *output;
	bool flag = false;

	//read and translate graph
	Initialize_graph(edge_table, vertex_table, n, e);

    // Add vectors in parallel.
    cudaError_t cudaStatus = ColorWithCuda(edge_table, vertex_table, n, e, &flag, output);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printf("Done. Found = %b",flag);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t ColorWithCuda(int *e_t, int *v_t, int *n, int *e, bool * flag, int * output)
{
    int *dev_e_t = 0;
    int *dev_v_t = 0;
    int *dev_n = 0;
	int *dev_output = 0;
	bool *dev_flag = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors and two values (four input, one output).
    cudaStatus = cudaMalloc((void**)&dev_e_t, (*e) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_v_t, (*n) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_output, (*n) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_flag, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU const memory.
    cudaStatus = cudaMemcpyToSymbol(EdgeTable, e_t, (*e) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpyToSymbol(VertexTable, v_t, (*n) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    cudaStatus = cudaMemcpy(dev_n, n, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_flag, flag, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
	BruteForceKernel <<<1, 512>>>(dev_n, dev_output);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_output, (*n) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(flag, dev_flag, sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_e_t);
    cudaFree(dev_v_t);
    cudaFree(dev_n);
	cudaFree(dev_flag);
    cudaFree(dev_output);
    
    return cudaStatus;
}
