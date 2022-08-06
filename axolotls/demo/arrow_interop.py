import axolotls as ax
import torch

col = ax.NumericColumn(
    torch.tensor([1, 0, 3, 4, 0, 6, 7, 8]), 
    presence=torch.tensor([True, False, True, True, False, True, True, True]))

print(col.to_arrow().__repr__())


non_null_col = ax.NumericColumn(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
print(non_null_col.to_arrow().__repr__())
