from .column_base import ColumnBase
from . import dtypes as dt
from typing import Optional
import torch

class ListColumn(ColumnBase):
    def __init__(
        self, 
        values: ColumnBase, 
        offsets: torch.Tensor, 
        presence: Optional[torch.BoolTensor] = None
    ) -> None:
        super().__init__(dtype=dt.List(values.dtype, nullable=presence is not None))

        if not isinstance(values, ColumnBase):
            raise ValueError("ListColumn expects values to be ColumnBase")
        if offsets.dim() != 1:
            raise ValueError("ListColumn expects 1D offsets Tensor")

        self._values = values
        self._offsets = offsets
        self._presence = presence

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.presence is None or self.presence[key]:
                return list(self.values[self.offsets[key] : self.offsets[key + 1]])
            return None
        elif isinstance(key, slice):
            # convert offsets to length for selection
            lengths = self.offsets[1:] - self.offsets[:-1]
            lengths = lengths[key]
            # convert selected lengths back to offsets
            offsets = torch.cat((torch.tensor([0]), torch.cumsum(lengths, dim=0)))

            values = self.values[key]
            presence = self.presence[key] if self.presence is not None else None
            return ListColumn(values=values, offsets=offsets, presence=presence)
        else:
            raise ValueError(f"Unsupported key for __getitem__: f{key}")

    @property
    def values(self) -> ColumnBase:
        return self._values

    @property
    def offsets(self) -> torch.Tensor:
        return self._offsets

    @property
    def presence(self) -> Optional[torch.BoolTensor]:
        return self._presence

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def to_arrow(self):
        if self.presence is not None:
            # pyarrow.ListArray.from_arrays doesn't take mask array
            # null value is marked in offsets array: https://arrow.apache.org/docs/python/generated/pyarrow.ListArray.html#pyarrow.ListArray.from_arrays
            # TODO: We should still be able to construct the desired offsets array with the null buffer?
            raise NotImplementedError

        # TODO: Check whether PyArrow is available 
        import pyarrow as pa
        values_array = self.values.to_arrow()
        # TODO: enforce offset to be int32 dtype
        offsets_array = pa.array(self.offsets.to(torch.int32).numpy())

        return pa.ListArray.from_arrays(values=values_array, offsets=offsets_array)
    