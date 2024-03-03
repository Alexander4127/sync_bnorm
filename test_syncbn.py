from functools import partial
import os
import pytest
import torch
import torch.distributed as dist

from syncbn import SyncBatchNorm


def init_process(rank, inputs, hid_dim, fn, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    return fn(rank, size, inputs, hid_dim)


def apply_default_version(inputs, hid_dim):
    bn = torch.nn.BatchNorm1d(num_features=hid_dim, momentum=0.1, eps=1e-5, affine=False)
    inputs.requires_grad = True
    outputs = bn(inputs)

    loss = outputs[:outputs.shape[0] // 2].sum()
    loss.backward()

    return outputs.detach(), inputs.grad.detach()


def apply_one_worker_sync(rank, size, inputs, hid_dim):
    bn = SyncBatchNorm(num_features=hid_dim, momentum=0.1, eps=1e-5)
    inputs.requires_grad = True
    outputs = bn(inputs)

    if rank * 2 < size - 1:
        loss = outputs.sum()
    elif rank * 2 == size - 1:
        loss = outputs[:outputs.shape[0] // 2].sum()
    else:
        loss = outputs[0, 0] * 0

    loss.backward()
    return outputs.detach(), inputs.grad.detach()


def apply_sync_version(inputs, num_workers, hid_dim, batch_size):
    ctx = torch.multiprocessing.get_context("spawn")
    fn = partial(init_process, fn=apply_one_worker_sync, size=num_workers, backend='gloo')

    lst_inputs = [
        inputs[start_pos:start_pos + batch_size]
        for start_pos in range(0, num_workers * batch_size, batch_size)
    ]
    lst_dims = [hid_dim] * num_workers

    with ctx.Pool(processes=num_workers) as pool:
        result = pool.starmap(fn, zip(range(num_workers), lst_inputs, lst_dims))

    outputs, grads = [el[0] for el in result], [el[1] for el in result]
    return torch.concatenate(outputs), torch.concatenate(grads)


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    torch.random.manual_seed(0)
    inputs = torch.randn([num_workers * batch_size, hid_dim], dtype=torch.float32)

    default_out, default_grad = apply_default_version(inputs.clone(), hid_dim)
    sync_out, sync_grad = apply_sync_version(inputs.clone(), num_workers, hid_dim, batch_size)

    assert torch.allclose(sync_out, default_out, atol=1e-3, rtol=0)
    assert torch.allclose(sync_grad, default_grad, atol=1e-3, rtol=0)
