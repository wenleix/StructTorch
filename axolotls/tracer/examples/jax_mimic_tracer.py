from axolotls.pytree import tree_flatten, flatten_func, tree_unflatten, Store
import axolotls as ax
import torch
import functools 
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from axolotls.tracer.tracer_base import Tracer, wrap
from copy import deepcopy
import axolotls.proxy.proxy_base as px

@functools.lru_cache()
def compile_func(fn: Callable[..., Any], ) -> Callable:
    store = Store()

    @functools.wraps(fn)
    def traced_fn(*args, **kwargs) -> Any:
        flat_args, in_tree = tree_flatten(args)   
        flat_fn, out_tree = flatten_func(fn, in_tree)

        tracer = Tracer()
        graph = tracer.trace(flat_fn)
        store.set(graph)

        # TODO: allow graph to evaluate on the real data and produce results
        flat_output = flat_fn(*flat_args)
        outs = tree_unflatten(flat_output, out_tree())
        return outs

    traced_fn.graph = store

    return traced_fn
      
    
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

def preproc(row_batch: ax.StructColumn) -> ax.StructColumn:
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
    
    out = preproc(deepcopy(row_batch))
    print(f"\n===Orig Output===")
    print(out)

    compiled_fn = compile_func(preproc)

    output = compiled_fn(deepcopy(row_batch))
    print(f"\n===Graph===")
    print(compiled_fn.graph())

    print(f"\n===Output===")
    print(output)
    
    
if __name__ == "__main__":
    main()