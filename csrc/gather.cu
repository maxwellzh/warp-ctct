#include "core.h"

#include <algorithm>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define W 32
#define G 1024
#define B 256

#define IDX2(n, u, D1) ((n) * (D1) + (u))
#define IDX3(n, t, u, D1, D2) ((n) * (D1) * (D2) + (t) * (D2) + (u))

__forceinline__ __device__ static float logaddexpf(float a, float b) {
    float const tmp = a - b;

    if (a == b)
        return (float)(a + M_LN2);

    if (tmp > 0)
        return a + log1pf(expf(-tmp));
    else if (tmp <= 0)
        return b + log1pf(expf(tmp));
    // in case of overflow
    return tmp;
}

__forceinline__ __device__ static void logaddexpf_(volatile float &a, float b) {
    float const tmp = a - b;

    if (a == b)
        a += M_LN2;
    else if (tmp > 0)
        a += log1pf(expf(-tmp));
    else if (tmp <= 0)
        a = b + log1pf(expf(tmp));
    else
        // in case of overflow
        a = tmp;

    return;
}

__global__ void k_warp_alphas(volatile float *alphas, const float *log_probs,
                              const int *ys, const int *lx, const int *ly,
                              int *counts, int N, int T, int U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int n = blockIdx.z;

    // i >= T-U+1 or s >= 2*U+1
    if (i >= lx[n] - ly[n] + 1 || s >= 2 * ly[n] + 1)
        return;

    int sT = T - U + 1;
    int sU = 2 * U + 1;
    U += 1;
    int *lock = counts + IDX3(0, n, s, N, sU);

    volatile float *ptr_c = alphas + IDX3(n, i, s, sT, sU);
    if (s == 0) {
        /*
         *   s=0 |----|----|----|----|
         *         w0   w1   w2   w3
         *   each warp includes 32 threads (lanes)
         */

        if (i == 0) {
            /*
             *   s=0 |----|----|...
             *        ↑
             *   init the very first thread
             */
            *ptr_c = 0.0f;
        } else {
            // indexing log_probs is tricky, t = i + shift, here shift = -1
            // moreover, log_probs is 4-dim, but the final dim is of size 2,
            // log_probs[n, t, u, 0] to index blank; log_probs[n, t, u, 1] to
            // index label.
            *ptr_c = log_probs[IDX3(n, i - 1, 0, T, U) << 1];
        }
        // compute scan (inclusive prefix sum)
        float a;
#pragma unroll
        for (int i = 1; i < W; i *= 2) {
            a = __shfl_up_sync(0xffffffff, *ptr_c, i);
            if (i <= threadIdx.x)
                *ptr_c += a;
        }

        if (blockIdx.x > 0) {
            while (atomicAdd(lock, 0) < blockIdx.x)
                ;
            /*
             *   s=0 ...|----|----|...
             *              ↑ ↑
             *   add init value from last lane of last warp.
             */
            *ptr_c += alphas[IDX3(n, blockIdx.x * W - 1, 0, sT, sU)];
            __syncwarp();
        }
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(lock, 1);
        }
        return;
    }

    if (blockIdx.y > 0)
        while (atomicAdd(lock - 1, 0) <= blockIdx.x)
            ;
    int shift = (s - 1) / 2;
    // prob of emitting label at cur pos
    float emit =
        log_probs[(IDX3(n, i + shift, shift + 1, T, U) << 1) - (s % 2)];
    if (i == 0) {
        /*
         *  s=2 |----|----|...
         *       ↑
         *  s=1 |----|----|...
         *       ↑
         *  s=0 |----|----|...
         *  init the first column of alphas.
         */

        if (s % 2 == 0)
            // first column of blank output: -inf
            *ptr_c = -INFINITY;
        else if (s == 1)
            // t = i + shift, shift = 0 here, index is (n, 0, 0, 1)
            *ptr_c = log_probs[(IDX3(n, 0, 0, T, U) << 1) + 1];
        else {
            if (ys[IDX2(n, shift, U)] == ys[IDX2(n, shift - 1, U)])
                *ptr_c = -INFINITY;
            else
                *ptr_c = alphas[IDX3(n, 0, s - 2, sT, sU)] +
                         log_probs[(IDX3(n, shift, shift, T, U) << 1) + 1];
        }
    } else {
        // (s % 2) clarifies blank / label
        if (s % 2 == 0) {
            *ptr_c = emit + alphas[IDX3(n, i - 1, s - 1, sT, sU)];
        } else {
            *ptr_c = emit + alphas[IDX3(n, i, s - 1, sT, sU)];
            if (s > 1 && ys[IDX2(n, shift, U)] != ys[IDX2(n, shift - 1, U)]) {
                logaddexpf_(*ptr_c, emit + alphas[IDX3(n, i, s - 2, sT, sU)]);
            }
        }
    }

    // just ignore skip from last warp (0.0), we would add it back later.
    float skip;
#pragma unroll
    for (int i = 1; i < W; i++) {
        skip = __shfl_up_sync(1 << (W - i), *ptr_c, 1);
        if (i == threadIdx.x) {
            logaddexpf_(*ptr_c, emit + skip);
        }
    }

    if (blockIdx.x > 0) {
        while (atomicAdd(lock, 0) < blockIdx.x)
            ;
            // preivous skip prob is not added. scan algo.
#pragma unroll
        for (int i = 1; i < W; i *= 2) {
            skip = __shfl_up_sync(0xffffffff, emit, i);
            if (i <= threadIdx.x)
                emit += skip;
        }

        logaddexpf_(*ptr_c,
                    emit + alphas[IDX3(n, blockIdx.x * W - 1, s, sT, sU)]);
        __syncwarp();
    }

    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(lock, 1);
    }
}

void run_warp_ctct(float *alphas, float *betas, const float *log_probs,
                   const int *ys, const int *lx, const int *ly, int *counts,
                   int N, int T, int U) {
    dim3 threads(W);
    dim3 blocks((T - U + W) / W, 2 * U + 1, N);
    k_warp_alphas<<<blocks, threads>>>(alphas, log_probs, ys, lx, ly, counts, N,
                                       T, U);
    CHECK_KERNEL_STAT("ctc-t alphas computing")
    return;
}