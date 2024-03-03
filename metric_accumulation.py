import torch
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader


@torch.no_grad()
def accuracy(rank, size, model, dataset, args) -> float:
    model.eval()
    device = torch.device(rank)

    real_size = (len(dataset) + size - 1) // size
    tensor_size = real_size + 1
    ind_tensor = torch.empty(tensor_size, dtype=torch.int64).to(device)

    lst_idx = None
    if rank == 0:
        lst_idx = []
        for idx_tensor in range(size):
            cur_tensor = torch.arange(real_size * idx_tensor, real_size * idx_tensor + tensor_size, device=device)
            cur_tensor[-1] = min(real_size, len(dataset) - real_size * idx_tensor)
            lst_idx.append(cur_tensor)

    dist.scatter(tensor=ind_tensor, scatter_list=lst_idx)

    ind_tensor = ind_tensor[:ind_tensor[-1]]
    sub_dataset = Subset(dataset, ind_tensor)
    loader = DataLoader(sub_dataset, batch_size=args.batch_size, drop_last=False)

    predicted = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        predicted += (output.argmax(dim=1) == target).sum().item()

    reduce_tns = torch.tensor([predicted, len(sub_dataset)], dtype=torch.int64, device=device)
    dist.all_reduce(reduce_tns, op=dist.ReduceOp.SUM)

    return reduce_tns[0].item() / reduce_tns[1].item()
