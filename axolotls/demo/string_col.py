import pyarrow as pa
import axolotls as ax
import torch

arr1 = pa.array(["abc", "de", "XYZ", "abcXYZ", "123"])
col1 = ax.StringColumn.from_arrow(arr1)

print(col1.__repr__() + '\n')
print(col1[2:].__repr__() + '\n')

print(str(col1) + "\n")

print(col1.to_arrow().__repr__())
print(str(col1[2:]) + "\n")
print(col1[2:].to_arrow().__repr__())


print("\n------------------------------------\n")

# List[String]
list_col = ax.ListColumn(
    values=col1,
    offsets=torch.tensor([0, 2, 3, 5])
)

print(list_col.__repr__() + '\n')
print(str(list_col))

print("\n------------------------------------\n")

arr2 = pa.array(["abc", "de", "XYZ", "不只是ascii"])
col2 = ax.StringColumn.from_arrow(arr2)

print(col2.__repr__() + '\n')
print(str(col2))

print(col2.to_arrow().__repr__())
