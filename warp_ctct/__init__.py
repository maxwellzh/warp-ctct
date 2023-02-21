"""CUDA-warp CTC-T loss compuatation.

Author: Huahuan Zheng (maxwellzh@outlook.com)

blank = 0
"""
import torch
import torch.nn.functional as F
import warp_ctct._C as core
from typing import Literal, Any
from pkg_resources import get_distribution
from torch.cuda.amp import autocast

__version__ = get_distribution('warp_ctct').version

T = torch.Tensor
__all__ = ["ctct_loss", "ctct_simple_loss"]


class CTCTLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: T, labels: T, lx: T, ly: T, requires_grad: bool):
        costs, grads = core.ctct_loss_fwd(
            log_probs, labels, lx, ly, requires_grad
        )
        if requires_grad:
            ctx.save_for_backward(grads)
        return costs

    @staticmethod
    def backward(ctx, grad_outputs: T):
        (grads,) = ctx.saved_tensors
        return grad_outputs.view(-1, 1, 1, 1)*grads, None, None, None, None


class LogMMExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, lhs: torch.Tensor, rhs: torch.Tensor, track_l: bool = True, track_r: bool = True) -> torch.Tensor:
        mm = core.log_matmul(lhs, rhs)

        if track_l or track_r:
            ctx.save_for_backward(lhs, rhs, mm)
            ctx.track = [track_l, track_r]
        return mm

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        lhs, rhs, res = ctx.saved_tensors

        grad_lhs, grad_rhs = core.log_matmul_backward(
            grad_outputs, lhs, rhs, res, ctx.track)
        return grad_lhs, grad_rhs, None, None


class SimpleCTCTLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: T, g: T, den: T, labels: T, lx: T, ly: T, rg_f: bool, rg_g: bool):
        if den is None:
            den = torch.empty(1)

        costs, grad_f, grad_g, grad_den = core.ctct_loss_simple_fwd(
            f, g, labels, den, lx, ly, (rg_f or rg_g)
        )

        if not rg_f:
            grad_f = None
        if not rg_g:
            grad_g = None
        if not (den.dim() == 3 and (rg_f or rg_g)):
            grad_den = None

        ctx.save_for_backward(grad_f, grad_g, grad_den)
        return costs

    @staticmethod
    def backward(ctx, grad_outputs: T):
        grad_f, grad_g, grad_den = ctx.saved_tensors
        grad_outputs = grad_outputs.view(-1, 1, 1)
        if grad_f is not None:
            grad_f *= grad_outputs
        if grad_g is not None:
            grad_g *= grad_outputs
        if grad_den is not None:
            grad_den *= grad_outputs
        return grad_f, grad_g, grad_den, None, None, None, None, None


def ctct_loss(log_probs: torch.FloatTensor,
              labels: torch.IntTensor,
              frames_lengths: torch.IntTensor,
              labels_lengths: torch.IntTensor,
              reduction: Literal['none', 'mean', 'sum'] = 'mean',
              average_frames: bool = False) -> torch.Tensor:
    """The CUDA-Warp CTC-Transducer loss.

    Args:
        log_probs (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels +1 and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
    """

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")

    assert log_probs.dim() == 4
    assert labels.dim() == 2
    assert log_probs.size(2) == labels.size(1) + 1
    assert log_probs.size(0) == labels.size(
        0) == frames_lengths.size(0) == labels_lengths.size(0)

    # gather blank and label
    N, T, U = log_probs.shape[:3]
    index = torch.full((N, U, 3), fill_value=0,
                       device=log_probs.device, dtype=torch.long)
    index[:, 1:, 1] = labels
    index[:, :U-1, 2] = labels
    index = index.unsqueeze(1).expand(-1, T, -1, -1)

    log_probs = log_probs.gather(dim=-1, index=index)

    enable_grad = (log_probs.requires_grad and torch.is_grad_enabled())
    with autocast(enabled=False):
        costs = CTCTLoss.apply(log_probs.float(), labels.to(torch.int), frames_lengths.to(
            torch.int), labels_lengths.to(torch.int), enable_grad)

    if average_frames:
        costs = costs / frames_lengths

    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    return costs


def ctct_simple_loss(f: torch.FloatTensor,
                     g: torch.FloatTensor,
                     labels: torch.IntTensor,
                     frames_lengths: torch.IntTensor,
                     labels_lengths: torch.IntTensor,
                     reduction: Literal['none', 'mean', 'sum'] = 'mean',
                     average_frames: bool = False,
                     normalized: bool = True) -> torch.Tensor:
    """The CUDA-Warp CTC-Transducer simple loss.

    Args:
        f (torch.FloatTensor): Input tensor with shape (N, T, V)
        g (torch.FloatTensor): Input tensor with shape (N, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels +1 and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
    """

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")

    assert f.dim() == 3 and g.dim() == 3
    assert labels.dim() == 2
    assert g.size(1) == labels.size(1) + 1
    assert f.size(0) == g.size(0) == labels.size(
        0) == frames_lengths.size(0) == labels_lengths.size(0)

    # gather blank and label
    N, T = f.shape[:2]
    U = g.size(1)

    track_f = f.requires_grad and torch.is_grad_enabled()
    track_g = g.requires_grad and torch.is_grad_enabled()
    # cal den if normalized
    if normalized:
        den = LogMMExp.apply(
            f.float(),
            g.float().transpose(1, 2),
            track_f, track_g
        )
    else:
        den = None

    # collect f: (N, T, V) -> (N, T, U), U = 1 + (U-1) = blank + labels
    index = torch.empty((N, T, U), device=f.device, dtype=torch.long)
    index[..., 0].fill_(0)
    index[..., 1:] = labels.unsqueeze(1).expand(-1, T, -1)
    f = f.gather(dim=-1, index=index)

    # collect g: (N, U, V) -> (N, U, 3), 3: 0 blank, 1 cur label, 2 next label
    index = torch.full((N, U, 3), fill_value=0,
                       device=f.device, dtype=torch.long)
    index[:, 1:, 1] = labels
    index[:, :U-1, 2] = labels
    g = g.gather(dim=-1, index=index)

    with autocast(enabled=False):
        costs = SimpleCTCTLoss.apply(
            f.float(), g.float(), den,
            labels.to(torch.int),
            frames_lengths.to(torch.int),
            labels_lengths.to(torch.int),
            track_f,
            track_g
        )

    if average_frames:
        costs = costs / frames_lengths

    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    return costs
