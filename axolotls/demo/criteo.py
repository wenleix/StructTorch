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

all_dense = ax.NumericColumn(
    torch.tensor(
        [1382, -100, 2,
        2, 0, 44,
        2, 0, 1,
        -100, 893, -100,
        3, -1, -100]),
    presence=torch.tensor([
        True, False, True,
        True, True, True,
        True, True, True,
        False, True, False,
        True, True, False
    ])
)
dense_grp = ax.ListColumn(
    values=all_dense,
    offsets=torch.arange(6) * 3,
)

print(f"dense1\n{dense1}\n")
print(f"dense2\n{dense2}\n")
print(f"dense3\n{dense3}\n")

df = ax.StructColumn({
    "dense1": dense1,
    "dense2": dense2,
    "dense3": dense3,
    "dense_grp": dense_grp,
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

import axolotls.functional.velox as VF
VF.transform_(df["dense_grp"], lambda col: col.fill_null_(0))
print(f"df\n{df}\n")

df["dense_grp"] = VF.transform(df["dense_grp"], lambda col: (col + 3).log())
print(f"df\n{df}\n")
