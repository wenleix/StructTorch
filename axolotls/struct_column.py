from .column_base import ColumnBase
from . import dtypes as dt
from typing import Dict
from tabulate import tabulate

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
