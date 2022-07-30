from .column_base import ColumnBase
from . import dtypes as dt
from typing import Optional
import torch

class NumericColumn(ColumnBase):
    def __init__(self, values: torch.Tensor, presence: Optional[torch.BoolTensor] = None) -> None:
        super().__init__(dtype=dt._dtype_from_pytorch_dtype(dtype=values.dtype, nullable=presence is not None))

        if values.dim() != 1:
            raise ValueError("NumericCollumn expects 1D values Tensor")
        self._values = values
        self._presence = presence

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.presence is None or self.presence[key]:
                return self.values[key].item()
            return None
        elif isinstance(key, slice):
            values = self.values[key]
            presence = self.presence[key] if self.presence is not None else None
            return NumericColumn(values=values, presence=presence)
        else:
            raise ValueError(f"Unsupported key for __getitem__: f{key}")

    def fill_null(self, val):
        if self.presence is None:
            # TODO: should we return a copy here?
            return self

        values = self.values.clone()
        values[~self.presence] = val
        return NumericColumn(values=values)

    def fill_null_(self, val):
        if self.presence is None:
            return self
        
        self.values[~self.presence] = val
        self._presence = None
        self._dtype = self._dtype.with_null(nullable=False)

        return self

    # Common PyTorch ops
    def logit(self, eps=None):
        return NumericColumn(
            values=self.values.logit(eps),
            presence=self.presence,
        )

    @property
    def values(self) -> torch.Tensor:
        return self._values
    
    @property
    def presence(self) -> Optional[torch.BoolTensor]:
        return self._presence

    def __len__(self) -> int:
        return len(self._values)

    def to_arrow(self):
        # TODO: Check whether PyArrow is available 
        import pyarrow as pa

        values = self.values.numpy()
        mask = ~self.presence.numpy() if self.presence is not None else None

        return pa.array(values, mask=mask)
