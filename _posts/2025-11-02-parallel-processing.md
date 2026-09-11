---
layout: post
author: Rui F. David
title:  "GPU Parallel Processing"
date:   2026-01-28 05:27:00 -0400
usemathjax: true
published: false
categories: software engineering
toc: true
---

# Introduction

## Many-threaded GPU vs Multicore CPU

Processors started as a single core, and manufactures like Intel and AMD would
simply increase the clock speed to make sequential programs faster. Eventually,
hitting higher clock speed becomes impossible because chips would get too hot.
CPUs shifted to **multicore** to improve performance on parallel workloads and
multitasking, while single-thread speed relied more in per-core efficiency (IPC).
On the other hand, **many-thread** trajectory focuses more on the execution
throughput of parallel applications. It began with a large number of threads,
increasing with each generation.
For instance, Intel® Xeon® 6980P Processor [1] has 128 cores and is 6.1 TFLOPs<sup>1</sup> [2]
whereas a NVIDIA GB200 Grace Blackwell Superchip is 160 TFLOPs for 32-bit
(FP32) single precision, and 80 TFLOPs for 64-bit double precision (FP64).
The performance gap between these two lies in the fundamental design
philosophies between them. This difference makes CPUs optimized for latency,
known for task parallelism, while GPUs are optimized for throughput, known as data parallelism.

## Task parallelism vs Data parallelism

Figure 1 illustrates the difference between CPU and GPU architectures. CPUs
prioritizes minimizing latency for individual tasks, having large, fast caches
and a complex control logic to handle diverse workloads efficiently. In contrast,
GPUs are designed to maximize throughput by executing many threads in parallel.
GPUs use multiple independent DRAM channels to provide high memory bandwidth.


![large](/assets/images/cpu-gpu.png "CPU (latency-oriented design) vs GPU (throughput-oriented design)")
_Figure 1: CPU (latency-oriented design) vs GPU (throughput-oriented design) -
image from PMPP book._


[ how kids contributed to AI - video game stuff ]

[1] https://www.intel.com/content/www/us/en/products/sku/240777/intel-xeon-6980p-processor-504m-cache-2-00-ghz/specifications.html?wapkw=6980P
[2] https://cdrdv2-public.intel.com/840270/APP-for-Intel-Xeon-Processors.pdf

<small><sup>1</sup> The documentation does not specify the floating point
precision.</small>

# Fundamental Concepts

In data parallelism, the computation work is performed simultaneously across
multiple data points. A classical example is image processing, where the same operation
(e.g., applying a filter) is applied to each pixel independently. For example,
to convert an image to grayscale, each pixel's RBG values have to be processed
using the following formula:


$$
\begin{equation}
    Pixel[i] = r \times 0.21 + g \times 0.72 + b \times 0.07
\end{equation}
$$

[ put some grayscale conversion image example here ]
_Figure 2: Grayscale conversion using data parallelism - each pixel is processed independently._

As each pixel can be processed independently, and it is not I/O bounded,
this operation can be parallelized.

## Vector addition example

Starting from the following example, there are several fundamentals concepts
important to understand parallel processing. Here is the complete example to
start having an intuition of how parallel processing works in practice. Each
part will be explained afterwards:

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C) {
    int n = A.size();
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // allocate on GPU
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // copy data from host (CPU) to device (GPU)
    cudaMemcpy(A_d, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (int)ceil(n / 256.0);

    // this is a single kernel, has only one grid
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C.data(), C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    std::vector<float> A = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<float> B = {10.0, 20.0, 30.0, 40.0, 50.0};
    std::vector<float> C(5);

    vectorAdd(A, B, C);

    std::cout << "Result:\n";
    for (size_t i = 0; i < C.size(); i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << "\n";
    }

    return 0;
}
```

### Kernel function and function qualifiers

Consider the addition of two vectors A and B,
storing the output in C. The following example computes vector sum `C = A + B`:

```cpp
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

Note the keyword `__global__` represents a function execution space qualifier,
meaning the function is a **kernel** function called from the **host (CPU)**
and executed on **device (GPU)***. There are other two functions qualifiers:
`__device__`, function callable from the device (GPU) and executed on the device;
and `__host__`, function callable from the host (CPU) and executed on the host:

| Qualifier Keyword &nbsp;&nbsp;&nbsp;&nbsp; | Callable from &nbsp;&nbsp;&nbsp;&nbsp; | Executed on &nbsp;&nbsp;&nbsp;&nbsp; | Executed by &nbsp;&nbsp;&nbsp;&nbsp; |
| ----------------- | ------------- | ----------- | ------------- |
| __global__        | Host          | Device      | Many threads  |
| __host__          | Host          | Host        | CPU           |
| __device__        | Device        | Device      | Single thread |


### Thread hierarchy: grid, block, thread

In heterogeneous data parallel processing, threads are organized into a
two-level hierarchy: **grid** and **block**. A grid consists of one or more
blocks, and each block consists of one or more **threads**.

When a program's host code calls a kernel, the CUDA runtime system launches a
**grid of threads**. Each grid is organized as an array of **thread blocks**,
which are all of the same size.

### Built-in variables: blockDim, blockIdx, threadIdx

The kernel computes the
index `i` for each thread using the built-in variables `blockDim`, `blockIdx`, and `threadIdx`.

- `blockDim`: dimensions of each block (number of threads per block). This is a
  struct with three unsigned int members: `x`, `y`, and `z`.

- `blockIdx`: dimensions of the grid (number of blocks in the grid). This is a struct
  with three unsigned int members: `x`, `y`, and `z`.

- `threadIdx`: thread index within the block. This is a struct with three unsigned int members:
  `x`, `y`, and `z`.

Once `vecAddKernel` is called from a specific thread, it calculates the global index `i`
by using the built-in variables blockDim, blockIdx, and threadIdx.
Note that `vecAddKernel` is a single kernel, consisting of one grid. The grid has many blocks, and each block has many threads, and each thread in the grid executes the same kernel code.


This image illustrates the calculation of vector addition and how the variables
are defined across the grid, blocks and threads:

![medium](/assets/images/threads-and-blocks.png "Threads and Blocks")
_Figure 2: This example shows how the index is calculated for each block and
thread using blockIdx.x, blockDim.x and threadIdx.x_

Kernel code for vector addition:

```cpp
void vectorAdd(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // 1. Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // 2. Copy data from host to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // 3. Launch kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // 4. Copy result back to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

# Computer Architecture

Blocks are more like containers whereas warps are chunks the hardware executes.






# Parallel Patterns


# Advanced Patterns and Applications

# Advanced Practices

75-minutes chapter, except 11, 14 and 15
+ buffer
1 chapter / day
23 chapters total = 1 month
Deadline to finish: March 1st



## Data Parallelism

[good example of RGB]



https://harmanani.github.io/csc447.html


## Task parallelism


# TODO:

https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- TABLE comparison with Hopper, Ampere, Blackwell
- Explain hardware
- Blur kenerl example from chapter 3 and color to grayscale and flattening
  matrices
