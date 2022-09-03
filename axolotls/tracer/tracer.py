from typing import Callable, Dict, Any, List, Tuple, Optional, Union, Set, NamedTuple
import functools
import math
import axolotls as ax
from itertools import chain
import torch
from types import ModuleType
import inspect
import builtins

BaseArgumentTypes = Union[str, int, float, bool, torch.Tensor]

Argument = Optional[
    Union[
        Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
        List[Any],  # actually Argument
        Dict[str, Any],  # actually Argument
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        'Node',
        BaseArgumentTypes
    ]
]

_WRAPPED_FNS_TO_PATCH: List[Tuple[dict, str]] = []

class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError()


class _PatchedFnSetItem(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _Patcher:
    def __init__(self):
        self.patches_made: List[_PatchedFn] = []
        self.visited: Set[int] = set()

    def patch(
        self,
        frame_dict: Dict[str, Any],
        name: str,
        new_fn: Callable,
        deduplicate: bool = True,
    ):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItem(frame_dict, name, frame_dict[name])
            )
        frame_dict[name] = new_fn

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True
    ):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false"""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()


def wrap(fn_or_name: Union[str, Callable]):
    if hasattr(fn_or_name, "__code__"):
        assert not isinstance(fn_or_name, str)  # to make mypy happy
        fn_name = fn_or_name.__code__.co_name
    else:
        assert isinstance(
            fn_or_name, str
        ), "fn_or_name must be a global function or string name"
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    assert currentframe is not None
    # Caller's frame
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != "<module>":
        raise NotImplementedError("wrap must be called at the top level of a module")

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    _WRAPPED_FNS_TO_PATCH.append((f.f_globals, fn_name))
    return fn_or_name


def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        proxy = None
        for arg in args:
            if isinstance(arg, ax.proxy.Proxy):
                proxy = arg
                break
        for arg in kwargs.values():
            if isinstance(arg, ax.proxy.Proxy):
                proxy = arg
                break
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy(
                "call_function", orig_fn, args, kwargs
            )
            # return_proxy.node.meta["is_wrapped"] = True
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped


def _patch_wrapped_functions(patcher: _Patcher):
    """
    Go through ``_WRAPPED_FNS_TO_PATCH`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for frame_dict, name in _WRAPPED_FNS_TO_PATCH:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))


# aka _autowrap_check()
def _patch_autowrap_functions(
    patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]
):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (
                not name.startswith("_")
                and callable(value)
                and id(value) in function_ids
            ):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


class SimpleTracer:
    graph: ax.proxy.Graph

    def __init__(
        self, 
        autowrap_modules: Tuple[ModuleType] = (math,),
        autowrap_functions: Tuple[Callable, ...] = (),
    ) -> None:
        self._autowrap_function_ids: Set[int] = {
            id(value)
            for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))
        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_modules: List[ModuleType] = list(autowrap_modules)

    def proxy(self, node: ax.proxy.Node) -> ax.proxy.Proxy:
        return ax.proxy.Proxy(node, self)

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.create_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore[arg-type]
        elif isinstance(a, (tuple, list)):
            return type(a)(self.create_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # Check for invalid dict keys. We do not want a Proxy to appear
                # anywhere within the key. Since keys can be collection types,
                # we iterate through the key with map_aggregate
                k = self.create_arg(k)
                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))
        elif isinstance(a, torch.Tensor):
            qualname: Optional[str] = self.tensor_attrs.get(a)
            if not qualname: 
                i = 0
                while True:
                    qualname = f"_tensor_constant{i}"
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)
            return self.create_node("get_attr", qualname, a, {})
        elif isinstance(a, ax.proxy.Proxy):
            # base case: we unwrap the Proxy object
            return a.node
        # TODO: Should we anatomate Columns or keep them as a generate node component? 
        elif isinstance(a, ax.ColumnBase):
            i = 0
            while True:
                qualname = f"_{a.__class__.__name__}_constant_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
            return self.create_node("get_attr", qualname, (), {})

        return a

    def create_node(self, kind: str, target: ax.proxy.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> ax.proxy.Node: 
        return self.graph.create_node(kind, target, args, kwargs)

    # Proxy is created on top of nodes
    # proxy represents a variable or an operation, 
    def create_proxy(self, kind: str, target: ax.proxy.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> ax.proxy.Proxy:
        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_)
        proxy = self.proxy(node)
        return proxy

    def trace(self, root: Callable[..., Any], ) -> ax.proxy.Graph:
        self.root = root
        self.graph = ax.proxy.Graph()
        self.tensor_attrs: Dict[torch.Tensor, str] = {}

        co = root.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        names_iter = iter(co.co_varnames)

        sig = inspect.signature(root)
        arg_names = [next(names_iter) for idx in range(total_args)]
        args = [self.create_proxy("placeholder", name, (), {}) for name in arg_names]

        with _Patcher() as patcher:
            _patch_wrapped_functions(patcher)
            _patch_autowrap_functions(patcher, root.__globals__, self._autowrap_function_ids)
            for module in self._autowrap_modules:
                _patch_autowrap_functions(
                    patcher, module.__dict__, self._autowrap_function_ids
                )
            result = root(*args)
            self.create_proxy("output", "output", (result,), {})

        return self.graph


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

    tracer = SimpleTracer()
    graph = tracer.trace(preproc)
    print(graph)
    
if __name__ == "__main__":
    main()