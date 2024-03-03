import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float, training: bool):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        assert len(input.shape) == 2, f'Unexpected ndim for shape {input.shape}'
        if not training:
            norm_input = (input - running_mean) / torch.sqrt(running_var + eps)
            return norm_input, running_mean, running_var

        bs_before = torch.tensor([len(input)], device=input.device)
        s = input.sum(dim=0)
        sq = torch.sum(input ** 2, dim=0)

        reduce_tns = torch.concatenate([bs_before, s, sq])
        dist.all_reduce(reduce_tns, op=dist.ReduceOp.SUM)

        bs_after = reduce_tns[0]
        s = reduce_tns[1:1 + len(s)]
        sq = reduce_tns[1 + len(s):]
        assert len(sq) == input.shape[1]

        mean = s / bs_after
        var = sq / bs_after - mean**2
        input_mean = input - mean

        sqrt_var = torch.sqrt(var + eps)
        inv_sqrt_var = 1 / sqrt_var

        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var * bs_after / (bs_after - 1)
        norm_input = input_mean * inv_sqrt_var

        ctx.save_for_backward(norm_input, inv_sqrt_var)

        return norm_input, running_mean, running_var

    @staticmethod
    def backward(ctx, grad_output, *args):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        norm_input, inv_sqrt_var = ctx.saved_tensors

        bs_before = torch.tensor([len(norm_input)], device=norm_input.device)
        grad_input_before = (grad_output * norm_input).sum(dim=0)
        grad_sum_before = grad_output.sum(dim=0)

        reduce_tns = torch.concatenate([bs_before, grad_input_before, grad_sum_before])
        dist.all_reduce(reduce_tns, op=dist.ReduceOp.SUM)

        bs_after = reduce_tns[0]
        grad_input_after = reduce_tns[1:1 + len(grad_input_before)]
        grad_sum_after = reduce_tns[1 + len(grad_input_before):]
        assert len(grad_sum_after) == len(grad_sum_before)

        grad_input = (grad_output * bs_after - grad_sum_after - norm_input * grad_input_after) * inv_sqrt_var / bs_after

        return grad_input, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        norm_input, running_mean, running_var = sync_batch_norm.apply(
            input,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training
        )
        if self.training:
            self.running_mean, self.running_var = running_mean.detach(), running_var.detach()
        return norm_input
