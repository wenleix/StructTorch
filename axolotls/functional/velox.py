from typing import Callable
from ..column_base import ColumnBase
from ..list_column import ListColumn

# TODO: lambda capture is not supported yet
def transform_(col: ListColumn, func: Callable[[ColumnBase], None]):
    func(col.values)
    return col

# TODO: lambda capture is not supported yet
def transform(col: ListColumn, func: Callable[[ColumnBase], ColumnBase]):
    values = func(col.values)

    return ListColumn(
        values=values,
        offsets=col.offsets,
        presence=col.presence,
    )
