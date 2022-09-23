from .numeric_column import NumericColumn
from .string_column import StringColumn
from .list_column import ListColumn
from .struct_column import StructColumn
from .column_base import ColumnBase, TRACE_MODE, TraceMode
from typing import Any, List, Tuple


if TRACE_MODE == TraceMode.JAX_MIMIC:
    import axolotls.pytree as pytree
    def _struct_column_flatten(col: StructColumn) -> Tuple[List[ColumnBase], pytree.Context]:
        return pytree._dict_flatten(col.field_columns)

    def _struct_column_unflatten(values: List[Any], context: pytree.Context) -> StructColumn:
        return StructColumn(
            field_columns=pytree._dict_unflatten(values, context),
        )
    pytree._register_pytree_node(StructColumn, _struct_column_flatten, _struct_column_unflatten)
