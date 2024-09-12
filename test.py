import torch.distributed as dist
import torch

def all_gather_array(array, ws):
    local_size = array.shape[0]
    all_sizes = [torch.tensor(0, dtype=torch.int64, device=array.device) for _ in range(ws)]
    dist.all_gather(all_sizes, torch.tensor(local_size, dtype=torch.int64, device=array.device))
    max_size = max([s.item() for s in all_sizes])

    size_diff = max_size - local_size
    if size_diff:
        padding_shape = list(array.shape)
        padding_shape[0] = size_diff
        padded = torch.cat((array, torch.zeros(torch.Size(padding_shape), device=array.device, dtype=array.dtype)), dim=0)
    else:
        padded = array

    all_arrays_padded = [torch.zeros_like(padded) for _ in range(ws)]
    dist.all_gather(all_arrays_padded, padded)

    all_arrays = []
    for a, size in zip(all_arrays_padded, all_sizes):
        all_arrays.append(a[:size])

    return torch.cat(all_arrays, dim=0)

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    ws = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    array = torch.randn(3 + rank, device=device)

    print(f"rank: {rank}\n {array}")

    all_array = all_gather_array(array, ws)

    print(f"rank: {rank}\n {all_array}")

    dist.destroy_process_group()
