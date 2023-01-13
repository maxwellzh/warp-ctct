#include <string>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK((x).device().is_cuda(), #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                                         \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Float,                    \
                #x " must be a Float tensor")

#define CHECK_INT(x)                                                           \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Int,                      \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

std::tuple<torch::Tensor, torch::Tensor>
ctct_loss_fwd(const torch::Tensor &log_probs, const torch::Tensor &ys,
              const torch::Tensor &lx, const torch::Tensor &ly,
              const bool requires_grad) {
    // check contiguous
    CHECK_CONTIGUOUS(log_probs);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(lx);
    CHECK_CONTIGUOUS(ly);
    // check dtype
    CHECK_FLOAT(log_probs);
    CHECK_INT(ys);
    CHECK_INT(lx);
    CHECK_INT(ly);
    // check device
    CHECK_CUDA(log_probs);
    CHECK_CUDA(ys);
    CHECK_CUDA(lx);
    CHECK_CUDA(ly);
    // check shape
    TORCH_CHECK(log_probs.dim() == 4, "log probs must have 4 dims.")
    TORCH_CHECK(log_probs.size(3) == 3,
                "log probs should be first gathered to length 3 at last dim.")
    TORCH_CHECK(ys.dim() == 2, "targets must have 2 dims.")
    TORCH_CHECK(lx.dim() == 1, "lx must have only 1 dim.")
    TORCH_CHECK(ly.dim() == 1, "ly must have only 1 dim.")
    TORCH_CHECK(log_probs.size(0) == ys.size(0) && ys.size(0) == lx.size(0) &&
                    lx.size(0) == ly.size(0),
                "input args should match at dim 0 (batch N)")

    // setup device guard.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(log_probs));

    const int N = log_probs.size(0);
    const int T = log_probs.size(1);
    const int U = log_probs.size(2) - 1;
    // sT = T-U+1
    const int sT = (lx - ly).max().item<int>() + 1;

    auto betas = torch::empty({N, sT, 2 * U + 1}, log_probs.options());
    torch::Tensor alphas;
    if (requires_grad)
        alphas = torch::empty_like(betas);
    else
        // alpha won't be used.
        alphas = betas;
    auto counts = torch::zeros({2, N, 2 * U + 1}, lx.options());

    run_warp_ctct(
        (float *)alphas.data_ptr<float>(), (float *)betas.data_ptr<float>(),
        (const float *)log_probs.data_ptr<float>(),
        (const int *)ys.data_ptr<int>(), (const int *)lx.data_ptr<int>(),
        (const int *)ly.data_ptr<int>(), (int *)counts.data_ptr<int>(), N, T, U,
        sT, !requires_grad);

    auto costs = -betas.index({Slice(), 0, 0});
    if (requires_grad) {
        auto grads = torch::zeros_like(log_probs);
        run_fill_grad((float *)grads.data_ptr<float>(),
                      (const float *)log_probs.data_ptr<float>(),
                      (const int *)ys.data_ptr<int>(),
                      (const float *)alphas.data_ptr<float>(),
                      (const float *)betas.data_ptr<float>(),
                      (const int *)lx.data_ptr<int>(),
                      (const int *)ly.data_ptr<int>(), N, T, U, sT);
        return std::make_tuple(costs, grads);
    } else {
        return std::make_tuple(costs, costs);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ctct_loss_simple_fwd(const torch::Tensor &f, const torch::Tensor &g,
                     const torch::Tensor &ys, const torch::Tensor &lx,
                     const torch::Tensor &ly, const bool requires_grad) {
    // check contiguous
    CHECK_CONTIGUOUS(f);
    CHECK_CONTIGUOUS(g);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(lx);
    CHECK_CONTIGUOUS(ly);
    // check dtype
    CHECK_FLOAT(f);
    CHECK_FLOAT(g);
    CHECK_INT(ys);
    CHECK_INT(lx);
    CHECK_INT(ly);
    // check device
    CHECK_CUDA(f);
    CHECK_CUDA(g);
    CHECK_CUDA(ys);
    CHECK_CUDA(lx);
    CHECK_CUDA(ly);
    // check shape
    TORCH_CHECK(f.dim() == 3, "f must have 3 dims.")
    TORCH_CHECK(g.dim() == 3, "g must have 3 dims.")
    TORCH_CHECK(
        f.size(2) == g.size(1),
        "f should be gathered at last dim to length U matching the g at dim 1.")
    TORCH_CHECK(g.size(2) == 3,
                "g should be first gathered to length 3 at last dim.")
    TORCH_CHECK(ys.dim() == 2, "targets must have 2 dims.")
    TORCH_CHECK(lx.dim() == 1, "lx must have only 1 dim.")
    TORCH_CHECK(ly.dim() == 1, "ly must have only 1 dim.")
    TORCH_CHECK(f.size(0) == g.size(0) && g.size(0) == ys.size(0) &&
                    ys.size(0) == lx.size(0) && lx.size(0) == ly.size(0),
                "input args should match at dim 0 (batch N)")

    // setup device guard.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

    const int N = f.size(0);
    const int T = f.size(1);
    const int U = g.size(1) - 1;
    // sT = T-U+1
    const int sT = (lx - ly).max().item<int>() + 1;

    auto betas = torch::empty({N, sT, 2 * U + 1}, f.options());
    torch::Tensor alphas;
    if (requires_grad)
        alphas = torch::empty_like(betas);
    else
        // alpha won't be used.
        alphas = betas;
    auto counts = torch::zeros({2, N, 2 * U + 1}, lx.options());

    run_warp_ctct_simple(
        (float *)alphas.data_ptr<float>(), (float *)betas.data_ptr<float>(),
        (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
        (const int *)ys.data_ptr<int>(), (const int *)lx.data_ptr<int>(),
        (const int *)ly.data_ptr<int>(), (int *)counts.data_ptr<int>(), N, T, U,
        sT, !requires_grad);

    auto costs = -betas.index({Slice(), 0, 0});
    if (requires_grad) {
        auto grads = torch::zeros({N, T, U + 1, 3}, f.options());
        run_fill_grad_simple((float *)grads.data_ptr<float>(),
                             (const float *)f.data_ptr<float>(),
                             (const float *)g.data_ptr<float>(),
                             (const int *)ys.data_ptr<int>(),
                             (const float *)alphas.data_ptr<float>(),
                             (const float *)betas.data_ptr<float>(),
                             (const int *)lx.data_ptr<int>(),
                             (const int *)ly.data_ptr<int>(), N, T, U, sT);
        auto grad_g = grads.sum({1});
        auto grad_f = torch::empty_like(f);
        // grad_f[..., 0] = grads[..., 0].sum(dim=2)
        grad_f.index_put_({"...", 0}, (grads.index({"...", 0})).sum({2}));
        // grad_f[..., 1:] = grads[..., 1:, 1]
        grad_f.index_put_({"...", Slice(1, None)},
                          grads.index({"...", Slice(1, None), 1}));
        // grad_f[..., 1:] += grads[..., :-1, 2]
        grad_f.index_put_({"...", Slice(1, None)},
                          grad_f.index({"...", Slice(1, None)}) +
                              grads.index({"...", Slice(None, U), 2}));
        return std::make_tuple(costs, grad_f, grad_g);
    } else {
        return std::make_tuple(costs, costs, costs);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ctct_loss_fwd", &ctct_loss_fwd,
          "CUDA-Warp CTC-Transducer loss (forward).",
          pybind11::arg("log_probs"), pybind11::arg("ys"), pybind11::arg("lx"),
          pybind11::arg("ly"), pybind11::arg("requires_grad"));

    m.def("ctct_loss_simple_fwd", &ctct_loss_simple_fwd,
          "CUDA-Warp CTC-Transducer loss (forward).", pybind11::arg("f"),
          pybind11::arg("g"), pybind11::arg("ys"), pybind11::arg("lx"),
          pybind11::arg("ly"), pybind11::arg("requires_grad"));
}
