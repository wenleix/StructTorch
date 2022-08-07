import torch

def _get_arrow_buffer_from_tensor(tensor: torch.Tensor):
    import pyarrow as pa

    return pa.foreign_buffer(
        address=tensor.data_ptr(),
        size=tensor.element_size() * tensor.numel(),
        base=tensor
    )
