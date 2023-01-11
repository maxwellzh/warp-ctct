"""CUDA-warp CTC-T loss compuatation.

Author: Huahuan Zheng (maxwellzh@outlook.com)

blank = 0
"""
import torch
import torch.nn.functional as F
import warp_ctct._C as core
from typing import Literal
from pkg_resources import get_distribution
from torch.cuda.amp import autocast

__version__ = get_distribution('warp_ctct').version

T = torch.Tensor
__all__ = ["ctct_loss"]


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


def ctct_loss(log_probs: torch.FloatTensor,
              labels: torch.IntTensor,
              frames_lengths: torch.IntTensor,
              labels_lengths: torch.IntTensor,
              average_frames: bool = False,
              reduction: Literal['none', 'mean', 'sum'] = 'mean') -> torch.Tensor:
    """The CUDA-Warp RNN-Transducer loss.

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
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
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
