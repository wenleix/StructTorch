import string
from .column_base import ColumnBase
from . import dtypes as dt
from typing import Optional
import torch

class StringColumn(ColumnBase):
    def __init__(
        self, 
        values: torch.ByteTensor, 
        offsets: torch.Tensor,
    ) -> None:
        # String Column doesn't support null value for now
        # It is straightforward to support it (with extra optional), but hold on for simplicity, 
        #   and only add it once there is clear use cases.
        super().__init__(dtype=dt.string)

        if not isinstance(values, torch.ByteTensor) or values.dim() != 1:
            raise ValueError("StringColumn expects values to be 1D ByteTensor")
        if not isinstance(offsets, torch.Tensor) or offsets.dim() != 1 or offsets.dtype != torch.int32:
            raise ValueError("StringColumn expects 1D offsets int32 Tensor")

        self._values = values
        self._offsets = offsets

    def clone(self) -> "StringColumn":
        return StringColumn(
            values=self.values.detach().clone(),
            offsets=self.offsets.detach().clone(),
        )

    def __getitem__(self, key) -> str:
        if isinstance(key, int):
            bytes: torch.ByteTensor = self.values[self.offsets[key] : self.offsets[key + 1]]
            return bytearray(bytes).decode('utf-8')

        if isinstance(key, slice):
            # non-continugous slice is not supported yet
            assert key.step is None
            start = key.start or 0
            stop = key.stop or len(self)

            values = self.values[self.offsets[start] : self.offsets[stop]]

            # convert offsets to length for selection
            lengths: torch.Tensor = self.offsets[1:] - self.offsets[:-1]
            lengths = lengths[key]
            # convert selected lengths back to offsets
            offsets = torch.cat((
                torch.tensor([0], dtype=torch.int32),
                torch.cumsum(lengths, dtype=torch.int32, dim=0)
            ))

            return StringColumn(values=values, offsets=offsets)

        raise ValueError(f"Unsupported key for __getitem__: f{key}")

    def __str__(self) -> str:
        return f"""StringColumn(
    values={self.values},
    offsets={self.offsets},
    dtype={self.dtype},
)"""

    @property
    def values(self) -> torch.ByteTensor:
        return self._values

    @property
    def offsets(self) -> torch.Tensor:
        return self._offsets

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def to_arrow(self):
        import pyarrow as pa
        from .utils import _get_arrow_buffer_from_tensor

        values_buffer = _get_arrow_buffer_from_tensor(self.values)
        offsets_buffer = _get_arrow_buffer_from_tensor(self.offsets)

        return pa.Array.from_buffers(
            type=pa.string(),
            length=len(self),
            buffers=[None, offsets_buffer, values_buffer],  # validity buffer is None
        )

    @staticmethod
    def from_arrow(array):
        import pyarrow as pa
        
        if not isinstance(array, pa.Array) or array.type != pa.string():
            raise ValueError(f"Expect array to be a StringArray")
        if array.null_count != 0:
            raise ValueError(f"Expect array contains no NULL")

        buffers = array.buffers()
        return StringColumn(
            values=torch.frombuffer(buffers[2], dtype=torch.uint8),
            offsets=torch.frombuffer(buffers[1], dtype=torch.int32),
        )
