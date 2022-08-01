from .column_base import ColumnBase
from . import dtypes as dt
from typing import Dict
from tabulate import tabulate
import torch

class StructColumn(ColumnBase):
    """
    In axoltls, Column designed to be both a data container implements Arrow memory layout,
    and a user interfacing class.

    As a result, StructColumn provides DataFrame-like API.
    """
    def __init__(self, field_columns: Dict[str, ColumnBase]) -> None:
        super().__init__(dtype=dt.Struct(
            [dt.Field(name, col.dtype) for (name, col) in field_columns.items()]
        ))

        self._field_columns = field_columns

    @property
    def columns(self):
        return list(self.field_columns.keys())

    def clone(self) -> "StructColumn":
        return StructColumn(
            field_columns={
                name: col.clone()
                for (name, col) in self.field_columns.items()
            }
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.field_columns[key]

        if isinstance(key, int):
            return tuple(col[key] for col in self.field_columns.values())

        raise ValueError(f"Unsupported key for __getitem__: f{key}")

    def __setitem__(self, key: str, value: ColumnBase):
        self._field_columns[key] = value
        self._dtype=dt.Struct(
            [dt.Field(name, col.dtype) for (name, col) in self._field_columns.items()]
        )
        
    def __len__(self):
        first_col = next(iter(self.field_columns.values()))
        return len(first_col)

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self):
        rows = []
        for idx in range(len(self)):
            row = self[idx]
            row = [val if val is not None else "None" for val in row]
            rows.append(row)

        tab = tabulate(
            rows, headers=["index"] + self.columns, tablefmt="simple", showindex=True
        )
        typ = f"dtype: {self._dtype}, length: {len(self)}"
        return tab + "\n" + typ    

    @property
    def field_columns(self) -> Dict[str, ColumnBase]:
        return self._field_columns

    # Data cleaning ops
    def fill_null(self, val) -> "StructColumn":
        return StructColumn(
            field_columns={
                name: col.fill_null(val)
                for (name, col) in self.field_columns.items()
            }
        )

    def fill_null_(self, val) -> "StructColumn":
        for col in self.field_columns.values():
            col.fill_null_(val)

        return self

    # Common Arithmatic / PyTorch ops
    def __add__(self, other) -> "StructColumn":
        if isinstance(other, (float, int, torch.Tensor)):
            return StructColumn(
                field_columns={
                    name: col + other
                    for (name, col) in self.field_columns.items()
                }
            )

        raise ValueError(f"Unsupported value {other}")

    def log(self) -> "StructColumn":
        return StructColumn(
            field_columns={
                name: col.log()
                for (name, col) in self.field_columns.items()
            }
        )

    def to_arrow(self):
        raise NotImplementedError
