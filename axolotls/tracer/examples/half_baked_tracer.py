from typing import Callable, Dict, Any, List, Tuple, Optional, Union, Set, NamedTuple
import axolotls as ax
import torch
from types import ModuleType
import axolotls.proxy.half_baked_proxy as px
from axolotls.tracer.tracer_base import Tracer, wrap

@wrap
def udf_normalize(col: ax.NumericColumn, shift: float, scale: float) -> ax.NumericColumn:
    result = (col.values + shift) * scale
    return ax.NumericColumn(values=result)

@wrap
def torch_arange(start: int, end: int, step: int = 1):
    return torch.arange(start, end, step)

def nonudf_bucketize(col: ax.NumericColumn, boundaries: List[float]) -> ax.ListColumn:
    boundary_tensor = torch.tensor(boundaries)
    result = torch.bucketize(col.values, boundary_tensor)
    result_col = ax.NumericColumn(values=result)
    offsets = torch_arange(0, result.numel() + 1)
    return ax.ListColumn(values=result_col, offsets=offsets)

def preproc(row_batch: ax.StructColumn):
    row_batch["float_features"]["f1"] = udf_normalize(row_batch["float_features"]["f1"], 1.0, 0.1)
    row_batch["id_list_features"] = ax.StructColumn(
        field_columns={
            "derived_id2": nonudf_bucketize(row_batch["float_features"]["f2"], [4., 5., 7.1, 9.])
        },
    )
    return row_batch

def main():
    row_batch = ax.StructColumn(
        field_columns={
            "float_features": ax.StructColumn(
                field_columns={
                    "f1": ax.NumericColumn(
                        values=torch.FloatTensor([1., 2., 3., 4.]),
                    ),
                    "f2": ax.NumericColumn(
                        values=torch.FloatTensor([5., 6., 7., 8.]),
                    ),
                },
            ),
        },
    )
    row_batch = preproc(row_batch)
    print(row_batch)

    tracer = Tracer()
    graph = tracer.trace(preproc)
    print(graph)
    
if __name__ == "__main__":
    main()