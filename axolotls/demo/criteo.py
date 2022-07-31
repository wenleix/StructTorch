import axolotls as ax
import torch

dense1 = ax.NumericColumn(
    torch.tensor([1382, 2, 2, -100, 3]), 
    presence=torch.tensor([True, True, True, False, True]))
dense2 = ax.NumericColumn(
    torch.tensor([-100, 0, 0, 893, -1]), 
    presence=torch.tensor([False, True, True, True, True]))
dense3 = ax.NumericColumn(
    torch.tensor([2, 44, 1, -100, -100]), 
    presence=torch.tensor([True, True, True, False, False]))

print(f"dense1\n{dense1}\n")
print(f"dense2\n{dense2}\n")
print(f"dense3\n{dense3}\n")

df = ax.StructColumn({
    "dense1": dense1,
    "dense2": dense2,
    "dense3": dense3,
})
print(f"df\n{df}\n")

print(f'df["dense1"] + df["dense2"]\n{df["dense1"] + df["dense2"]}\n')
print(f'df["dense1"] + df["dense3"]\n{df["dense1"] + df["dense3"]}\n')

df["dense1"] = df["dense1"].fill_null(0)
df["dense2"].fill_null_(0)
print(f"df\n{df}\n")

df["dense1"] = (df["dense1"] + 3).log()
df["dense2"] = (df["dense2"] + 3).log()
print(f"df\n{df}\n")
