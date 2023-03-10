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
// for indexing log probs of label tokens, v = 1, 2
// v = 0: blank
// v = 1: label at u
// v = 2: label at u+1
#define LOGPROBY(n, t, u, v, T, U)                                             \
    (f[IDX3(n, t, (u) + ((v)-1), T, U)] + g[IDX3(n, u, v, U, 3)] -             \
     den[IDX3(n, t, u, T, U)])

// for blank token
#define LOGPROBB(n, t, u, T, U)                                                \
    (f[IDX3(n, t, 0, T, U)] + g[IDX3(n, u, 0, U, 3)] - den[IDX3(n, t, u, T, U)])

__forceinline__ __device__ static void logaddexpf_(volatile float &a, float b) {
    float tmp = a - b;
    if (tmp < 0) {
        a = b + log1pf(expf(tmp));
    } else if (tmp >= 0) {
        a += log1pf(expf(b - a));
    }
    return;
}

__global__ void k_warp_alphas(volatile float *alphas, const float *f,
                              const float *g, const float *den, const int *ys,
                              const int *lx, const int *ly, int *counts, int N,
                              int T, int U, int sT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int n = blockIdx.z;

    // i >= T-U+1 or s >= 2*U+1
    if (i >= lx[n] - ly[n] + 1 || s >= 2 * ly[n] + 1)
        return;

    // this is for alpha only, t = i + s/2 + 1, assure t-1 <= T-1
    if (i + s / 2 >= lx[n])
        return;

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
             *        ???
             *   init the very first thread
             */
            *ptr_c = 0.0f;
        } else {
            // t = i + shift, where
            // shift = 0, for s = 0
            //       = s//2 + 1, for s > 0
            // u = (s-1)//2 is the index of ys label
            // therefore u+1, or (s+1)//2, is the index of log probs u-dim

            // (n, i-1, 0, 0)
            *ptr_c = LOGPROBB(n, i - 1, 0, T, U);
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
             *              ??? ???
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
    int shift = s / 2 + 1;
    int ul = (s - 1) / 2;

    if (i == 0) {
        /*
         *  s=2 |----|----|...
         *       ???
         *  s=1 |----|----|...
         *       ???
         *  s=0 |----|----|...
         *  init the first column of alphas.
         */
        if ((s & 0x00000001) == 0)
            *ptr_c = alphas[IDX3(n, 0, s - 1, sT, sU)] +
                     // (n, shift-1, s/2, 0)
                     LOGPROBB(n, shift - 1, s / 2, T, U);
        else if (s == 1)
            // t = i + shift, shift = 0 here
            *ptr_c = LOGPROBY(n, 0, 0, 2, T, U);
        else if (ys[IDX2(n, ul, U - 1)] == ys[IDX2(n, ul - 1, U - 1)])
            *ptr_c = -INFINITY;
        else
            *ptr_c = alphas[IDX3(n, 0, s - 2, sT, sU)] +
                     // (n, shift-1, ul, 2)
                     LOGPROBY(n, shift - 1, ul, 2, T, U);
    } else {
        // (s % 2) clarifies blank / label emitting
        if ((s & 0x00000001) == 0) {
            *ptr_c = alphas[IDX3(n, i, s - 1, sT, sU)] +
                     // (n, i + shift - 1, s / 2, 0)
                     LOGPROBB(n, i + shift - 1, s / 2, T, U);
        } else if (s == 1)
            *ptr_c = alphas[IDX3(n, i, 0, sT, sU)] + LOGPROBY(n, i, 0, 2, T, U);
        else {
            *ptr_c = alphas[IDX3(n, i - 1, s - 1, sT, sU)] +
                     LOGPROBY(n, i + shift - 1, s / 2, 2, T, U);
            if (ys[IDX2(n, ul, U - 1)] != ys[IDX2(n, ul - 1, U - 1)])
                logaddexpf_(*ptr_c,
                            alphas[IDX3(n, i, s - 2, sT, sU)] +
                                LOGPROBY(n, i + shift - 1, ul, 2, T, U));
        }
    }

    // just ignore skip from last warp (set it to 0), we would add it back
    // later.
    float skip;
    // prob of alpha(t-1, s) -> alpha(t, s)
    float emit;
    if ((s & 0x00000001) == 0)
        emit = LOGPROBB(n, i + shift - 1, ul + 1, T, U);
    else
        emit = LOGPROBY(n, i + shift - 1, ul + 1, 1, T, U);
    for (int i = 1; i < W; i++) {
        skip = __shfl_up_sync(1 << (W - i), *ptr_c, 1);
        if (i == threadIdx.x)
            logaddexpf_(*ptr_c, emit + skip);
    }

    if (blockIdx.x > 0) {
        // preivous skip prob is not added. scan algo.
#pragma unroll
        for (int i = 1; i < W; i *= 2) {
            skip = __shfl_up_sync(0xffffffff, emit, i);
            if (i <= threadIdx.x)
                emit += skip;
        }

        while (atomicAdd(lock, 0) < blockIdx.x)
            ;
        logaddexpf_(*ptr_c,
                    emit + alphas[IDX3(n, blockIdx.x * W - 1, s, sT, sU)]);
        __syncwarp();
    }

    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(lock, 1);
    }
}

__global__ void k_warp_betas(volatile float *betas, const float *f,
                             const float *g, const float *den, const int *ys,
                             const int *lx, const int *ly, int *counts, int N,
                             int T, int U, int sT) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int n = blockIdx.z;

    // indexing betas(sT1-i, sU1-s)
    const int sT1 = lx[n] - ly[n] + 1 - 1;
    const int sU1 = 2 * ly[n] + 1 - 1;

    // i >= T-U+1 or s >= 2*U+1
    if (i > sT1 || s > sU1)
        return;

    const int sU = 2 * U + 1;
    U += 1;
    int *lock = counts + IDX3(1, n, s, N, sU);

    volatile float *ptr_c = betas + IDX3(n, sT1 - i, sU1 - s, sT, sU);
    int t = (s == sU1) ? (sT1 - i) : (sT1 - i + (sU1 - s) / 2 + 1);
    int up = (s == sU1) ? 0 : (sU1 - s + 1) / 2;

    float emit;
    if (s == 0 && (i == 0 || i == 1)) {
        emit = 0.0f;
        *ptr_c = 0.0f;
    } else if ((s & 0x00000001) == 0) {
        emit = LOGPROBB(n, t, up, T, U);
    } else {
        emit = LOGPROBY(n, t, up, 1, T, U);
    }

    if (blockIdx.y > 0)
        while (atomicAdd(lock - 1, 0) <= blockIdx.x)
            ;

    if (s == sU1) {
        // specially dealing the row 0, shift = 0
        *ptr_c = betas[IDX3(n, sT1 - i, 1, sT, sU)] +
                 LOGPROBY(n, sT1 - i, 0, 2, T, U);
    } else if (i == 0) {
        // init betas in last col
        if (s == 1)
            *ptr_c = 0.0f;
        else if ((s & 0x00000001) == 1 &&
                 ys[IDX2(n, up - 1, U - 1)] != ys[IDX2(n, up, U - 1)])
            *ptr_c = betas[IDX3(n, sT1 - i, sU1 - s + 2, sT, sU)] +
                     LOGPROBY(n, t, up, 2, T, U);
        else
            *ptr_c = -INFINITY;
    } else if (s > 0) {
        // sum up probs come from upper rows
        if ((s & 0x00000001) == 0) {
            *ptr_c = betas[IDX3(n, sT1 - i + 1, sU1 - s + 1, sT, sU)] +
                     LOGPROBY(n, t, up, 2, T, U);
        } else {
            *ptr_c = betas[IDX3(n, sT1 - i, sU1 - s + 1, sT, sU)] +
                     LOGPROBB(n, t, up, T, U);
            if (s > 1 && ys[IDX2(n, up - 1, U - 1)] != ys[IDX2(n, up, U - 1)])
                logaddexpf_(*ptr_c,
                            betas[IDX3(n, sT1 - i, sU1 - s + 2, sT, sU)] +
                                LOGPROBY(n, t, up, 2, T, U));
        }
    }

    float skip;
    if (s == 0) {
        if (blockIdx.x == 0) {
#pragma unroll
            for (int i = 1; i < W; i *= 2) {
                skip = __shfl_up_sync(0xffffffff, emit, i);
                if (i <= threadIdx.x)
                    emit += skip;
            }
            *ptr_c = emit;
        } else
            *ptr_c = -INFINITY;

    } else {
        for (int i = 1; i < W; i++) {
            skip = __shfl_up_sync(1 << (W - i), *ptr_c, 1);
            if (i == threadIdx.x)
                logaddexpf_(*ptr_c, emit + skip);
        }
    }

    if (blockIdx.x > 0) {
        // scan algo.
#pragma unroll
        for (int i = 1; i < W; i *= 2) {
            skip = __shfl_up_sync(0xffffffff, emit, i);
            if (i <= threadIdx.x)
                emit += skip;
        }
        while (atomicAdd(lock, 0) < blockIdx.x)
            ;
        logaddexpf_(
            *ptr_c,
            emit + betas[IDX3(n, sT1 - (blockIdx.x * W - 1), sU1 - s, sT, sU)]);
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(lock, 1);
    }
}

void run_warp_ctct_simple(float *alphas, float *betas, const float *f,
                          const float *g, const float *den, const int *ys,
                          const int *lx, const int *ly, int *counts, int N,
                          int T, int U, int sT, bool beta_only) {
    dim3 threads(W);
    dim3 blocks((sT + W - 1) / W, 2 * U + 1, N);

    if (not beta_only) {
        k_warp_alphas<<<blocks, threads>>>(alphas, f, g, den, ys, lx, ly,
                                           counts, N, T, U, sT);
        CHECK_KERNEL_STAT("ctc-t simple forward alphas")
    }

    k_warp_betas<<<blocks, threads>>>(betas, f, g, den, ys, lx, ly, counts, N,
                                      T, U, sT);
    CHECK_KERNEL_STAT("ctc-t simple forward betas")
    return;
}

__global__ void k_fill_grad(float *grads, const float *f, const float *g,
                            const float *den, const int *ys,
                            const float *alphas, const float *betas,
                            const int *lx, const int *ly, int T, int U,
                            int sT) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    // s = 2u-1, s' = 2u
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (t >= lx[n] || u > ly[n])
        return;

    int sU = 2 * U + 1;
    U += 1;

    grads += 3 * IDX3(n, t, u, T, U);
    float cost = betas[IDX3(n, 0, 0, sT, sU)];

    // blank row
    int s = 2 * u;
    int shift = (u == 0) ? 0 : (s / 2 + 1);
    int i = t - shift;
    int t_end = lx[n] - (2 * ly[n] - s) / 2;
    if (t >= shift && t <= t_end) {
        if (t < t_end)
            grads[0] = -expf(alphas[IDX3(n, i, s, sT, sU)] +
                             betas[IDX3(n, i + 1, s, sT, sU)] +
                             LOGPROBB(n, t, u, T, U) - cost);

        if (u < ly[n]) {
            grads[2] = -expf(alphas[IDX3(n, i, s, sT, sU)] +
                             betas[IDX3(n, i + (u > 0), s + 1, sT, sU)] +
                             LOGPROBY(n, t, u, 2, T, U) - cost);
        }
    }
    if (u == 0)
        return;

    // label row
    s -= 1;
    shift = s / 2 + 1;
    i = t - shift;
    t_end = lx[n] - (2 * ly[n] - s) / 2;
    if (t < shift || t > t_end)
        return;

    if (t < t_end) {
        grads[0] += -expf(alphas[IDX3(n, i, s, sT, sU)] +
                          betas[IDX3(n, i, s + 1, sT, sU)] +
                          LOGPROBB(n, t, u, T, U) - cost);
        grads[1] = -expf(alphas[IDX3(n, i, s, sT, sU)] +
                         betas[IDX3(n, i + 1, s, sT, sU)] +
                         LOGPROBY(n, t, u, 1, T, U) - cost);
    }
    if (u == ly[n])
        return;
    if (ys[IDX2(n, (s - 1) / 2, U - 1)] != ys[IDX2(n, (s + 1) / 2, U - 1)]) {
        grads[2] += -expf(alphas[IDX3(n, i, s, sT, sU)] +
                          betas[IDX3(n, i, s + 2, sT, sU)] +
                          LOGPROBY(n, t, u, 2, T, U) - cost);
    }
}

void run_fill_grad_simple(float *grads, const float *f, const float *g,
                          const float *den, const int *ys, const float *alphas,
                          const float *betas, const int *lx, const int *ly,
                          int N, int T, int U, int sT) {
    dim3 threads(128, 8);
    dim3 blocks((T + 128 - 1) / 128, (U + 1 + 8 - 1) / 8, N);

    k_fill_grad<<<blocks, threads>>>(grads, f, g, den, ys, alphas, betas, lx,
                                     ly, T, U, sT);

    CHECK_KERNEL_STAT("ctc-t simple fill grads")
    return;
}

/*
__global__ void kernel_warp_grad_den(volatile float *grad_den,
                                     const float *alphas, const float *betas,
                                     const float *f, const float *g,
                                     const float *den, const int *lf,
                                     const int *lg, unsigned int N,
                                     unsigned int T, unsigned int U, int sT) {
    unsigned int n = blockIdx.z;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int u = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= T || u >= U)
        return;

    grad_den += IDX3(n, t, u, T, U);
    if (t >= lf[n] || u > lg[n]) {
        // zero the paddings
        *grad_den = 0.0f;
        return;
    }

    int sU = 2 * (U - 1) + 1;
    float llh = betas[IDX3(n, 0, 0, sT, sU)];
    *grad_den = 0.0f;

    // grad from blank row
    int s = 2 * u;
    int shift = (u == 0) ? 0 : (s / 2 + 1);
    int i = t - shift;
    int t_end = lf[n] - (2 * lg[n] - s) / 2;
    if (t >= shift && t <= t_end) {
        if (t < t_end)
            *grad_den += expf(alphas[IDX3(n, i, s, sT, sU)] +
                              betas[IDX3(n, i + 1, s, sT, sU)] +
                              LOGPROBB(n, t, u, T, U) - llh);
        if (t < lg[n])
            *grad_den += exp(alphas[IDX3(n, i, s, sT, sU)] +
                             betas[IDX3(n, i + (u > 0), s + 1, sT, sU)] +
                             LOGPROBY(n, t, u, 2, T, U) - llh);
    }

    float p_zero;
    float p_label;

    if (t == lf[n] - 1) {
        if (u == lg[n])
            p_zero = LOG_PROB_BLANK(n, t, u);
        else
            p_zero = -std::numeric_limits<float>::infinity();
    } else {
        p_zero = LOG_PROB_BLANK(n, t, u) + betas[IDX3(n, t + 1, u, T, U)];
    }

    if (u == lg[n]) {
        p_label = -std::numeric_limits<float>::infinity();
    } else {
        p_label = LOG_PROB_Y(n, t, u) + betas[IDX3(n, t, u + 1, T, U)];
    }

    *grad_den = expf(alphas[IDX3(n, t, u, T, U)] - betas[IDX3(n, 0, 0, T, U)] +
                     logaddexpf(p_zero, p_label));
}

void run_fill_grad_den(volatile float *grad_den, const float *alphas,
                       const float *betas, const float *f, const float *g,
                       const float *den, const int *lf, const int *lg,
                       unsigned int N, unsigned int T, unsigned int U) {
    dim3 threads(W, W);
    dim3 blocks((T + W - 1) / W, (U + W - 1) / W, N);
    kernel_warp_grad_den<<<blocks, threads>>>(grad_den, alphas, betas, f, g,
                                              den, lf, lg, N, T, U);
    CHECK_KERNEL_STAT("rnnt simple loss computing gradients for denonimator.")
}
*/