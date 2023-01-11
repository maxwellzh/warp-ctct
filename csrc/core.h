#ifndef CTCT_CORE_H
#define CTCT_CORE_H

#include <cuda_runtime.h>
#define CHECK_KERNEL_STAT(s)                                                   \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, #s " error: %s\n", cudaGetErrorString(err));       \
            exit(-1);                                                          \
        }                                                                      \
    }

void run_warp_ctct(float *alphas, float *betas, const float *log_probs,
                   const int *ys, const int *lx, const int *ly, int *counts,
                   int N, int T, int U, int sT, bool beta_only);

void run_fill_grad(float *grads, const float *log_probs, const int *ys,
                   const float *alphas, const float *betas, const int *lx,
                   const int *ly, int N, int T, int U, int sT);
#endif