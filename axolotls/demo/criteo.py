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
dense_grp = df.clone()
df["dense_grp"] = dense_grp
print(f"df\n{df.__repr__()}\n")

print(f"df\n{df.__repr__()}\n")

df["dense1"] = df["dense1"].fill_null(0)
df["dense2"].fill_null_(0)
print(f"df\n{df.__repr__()}\n")

df["dense1"] = (df["dense1"] + 3).log()
df["dense2"] = (df["dense2"] + 3).log()
print(f"df\n{df.__repr__()}\n")

df["dense_grp"].fill_null_(0)
df["dense_grp"] = (df["dense_grp"] + 3).log()

print(f"df\n{df.__repr__()}\n")

print(str(df["dense1"]))
print(str(df["dense3"]))

col = ax.NumericColumn(torch.tensor([1, -100, 3, 4, -100, 6, 7, 8]), presence=torch.tensor([True, False, True, True, False, True, True, True]))
list_col = ax.ListColumn(col, offsets=torch.tensor([0, 1, 3, 6, 8]))

print(str(list_col))