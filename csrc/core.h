#pragma once

#include <ATen/ATen.h>
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

void run_warp_ctct_simple(float *alphas, float *betas, const float *f,
                          const float *g, const int *ys, const int *lx,
                          const int *ly, int *counts, int N, int T, int U,
                          int sT, bool beta_only);
void run_warp_ctct_simple(float *alphas, float *betas, const float *f,
                          const float *g, const float *den, const int *ys,
                          const int *lx, const int *ly, int *counts, int N,
                          int T, int U, int sT, bool beta_only);

void run_fill_grad_simple(float *grads, const float *f, const float *g,
                          const int *ys, const float *alphas,
                          const float *betas, const int *lx, const int *ly,
                          int N, int T, int U, int sT);
void run_fill_grad_simple(float *grads, const float *f, const float *g,
                          const float *den, const int *ys, const float *alphas,
                          const float *betas, const int *lx, const int *ly,
                          int N, int T, int U, int sT);

void log_matmul_cuda_impl(const at::Tensor &lhs_, const at::Tensor &rhs_,
                          const at::Tensor &out);