from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type
from collections import namedtuple

from ._pytree_utils import _register_pytree_node, tree_flatten, tree_unflatten, flatten_func, Context, Store

def _tuple_flatten(t: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(t), None

def _tuple_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)

def _list_flatten(l: List[Any]) -> Tuple[List[Any], Context]:
    return l, None

def _list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return list(values)

def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}

def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)

def _namedtuple_unflatten(values: List[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))


_register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten)
_register_pytree_node(list,  _list_flatten, _list_unflatten)
_register_pytree_node(dict, _dict_flatten, _dict_unflatten)
_register_pytree_node(namedtuple, _namedtuple_flatten, _namedtuple_unflatten)