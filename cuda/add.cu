// add.cu
#include "add.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
// Kernel function to add the elements of two arrays
// __global__ 变量声明符，作用是将add函数变成可以在GPU上运行的函数
// __global__ 函数被称为kernel，
// 在 GPU 上运行的代码通常称为设备代码（device code），而在 CPU 上运行的代码是主机代码（host code）。

__device__ void test() {
    // power_mod 998244353
    // 1e9+7
    long long P = 998244353;
    // a^b mod P
    long long a = 2;
    long long b = 1000000000000000000;
    long long ans = 1;
    while (b) {
        if (b & 1) {
            ans = ans * a % P;
        }
        a = a * a % P;
        b >>= 1;
    }
}


__global__ void cuda_add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
        test();
    }
}


#include <time.h>

int add_test(int N, int blockSize) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);

    float *x, *y;
    x = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = 3.0f;
        y[i] = 2.0f;
    }
    printf("start gpu test!\n");
    clock_t start = clock();
    float *gpu_x, *gpu_y;
    cudaMalloc((void **) &gpu_x, N * sizeof(float));
    cudaMalloc((void **) &gpu_y, N * sizeof(float));

    cudaMemcpy(gpu_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    // execution configuration, 执行配置
    // int blockSize = 2048;
    int numBlock = (N + blockSize - 1) / blockSize;

    cuda_add<<< numBlock, blockSize >>>(N, gpu_x, gpu_y);

    // Wait for GPU to finish before accessing on host
    // CPU需要等待cuda上的代码运行完毕，才能对数据进行读取
    //   cudaDeviceSynchronize();

    cudaMemcpy(y, gpu_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    double time = (double) (clock() - start) / (double) CLOCKS_PER_SEC;
    printf("gpu: %lf ms\n", time * 1000);
    //   float maxError = 0.0f;
    //   for (int i = 0; i < N; i++)
    //     maxError = fmax(maxError, fabs(y[i]-5.0f));
    //   std::cout << "Max error: " << maxError << std::endl;

    for (int i = 0; i < N; i++) {
        x[i] = 3.0f;
        y[i] = 2.0f;
    }
    printf("start cpu test!\n");
    start = clock();
    for (int i = 0; i < N; i++) {
        y[i] = x[i] + y[i];
    }
    time = (double) (clock() - start) / (double) CLOCKS_PER_SEC;
    printf("cpu: %lf ms\n", time * 1000);

    //   maxError = 0.0f;
    //   for (int i = 0; i < N; i++)
    //     maxError = fmax(maxError, fabs(y[i]-5.0f));
    //   std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    free(x);
    free(y);

    return 0;
}