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
        pass

    def __repr__(self) -> str:
        return f"NumericaColumn(\n\tvalues={self._values}\n\tdtype={self._dtype}\n)"

    def __str__(self) -> str:
        return self.__repr__()

    def values(self) -> torch.Tensor:
        return self._data
    
    def presence(self) -> Optional[torch.BoolTensor]:
        return self._presence

    def __len__(self) -> int:
        return self._data.size(0)
