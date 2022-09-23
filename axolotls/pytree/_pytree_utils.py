from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type
from collections import namedtuple
import functools
# Refer to https://jax.readthedocs.io/en/latest/autodidax.html#pytrees-and-flattening-user-functions-inputs-and-outputs
# Jax: https://jax.readthedocs.io/en/latest/pytrees.html
# Very close to https://github.com/pytorch/pytorch/blob/master/torch/utils/_pytree.py
Context = Any

@dataclass
class NodeDef:
  ty: Type
  flatten_fn: Callable
  unflatten_fn: Callable

print("init SUPPORTED_NODES")
SUPPORTED_NODES: Dict[Type, NodeDef] = {}


def _register_pytree_node(ty: Type, flatten_fn: Callable, unflatten_fn: Callable) -> None:
  SUPPORTED_NODES[ty] = NodeDef(ty, flatten_fn, unflatten_fn)

# TreeSpec
class TreeDef:
    def __init__(self, typ: Any, context: Context, children_specs: List['TreeDef']) -> None:
        self.type = typ
        self.context = context
        self.children_specs = children_specs
        self.num_leaves: int = sum([spec.num_leaves for spec in children_specs])

    def __repr__(self) -> str:
        return (
            f"TreeDef(\n  typ={self.type},\n  context={self.context},"
            f"\n  children_specs=(\n    {self.children_specs}\n  ),"
            f"\n  num_leaves={self.num_leaves}\n)"
        )

    __str__ = __repr__


# A tree can be a single leaf 
class LeafDef(TreeDef): 
    def __init__(self) -> None:
        super().__init__(None, None, [])
        self.num_leaves = 1

def _is_namedtuple_instance(pytree: Any) -> bool:
    typ = type(pytree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)

def _get_node_type(pytree: Any) -> Any:
    if _is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)

# A leaf is defined as anything that is not a Node.
def _is_leaf(pytree: TreeDef) -> bool:
    global SUPPORTED_NODES
    print(f"_is_leaf?: {_get_node_type(pytree)}, {SUPPORTED_NODES.keys()}")
    return _get_node_type(pytree) not in SUPPORTED_NODES.keys()

# postorder
def tree_flatten(x: Any) -> Tuple[List[Any], TreeDef]:
    # NOTE: Does not flatten Proxy now. Need something to handle Proxy with particular properties like https://fburl.com/code/by5ubu66
    if _is_leaf(x):
        print(f"XX2: is_leaf {x}")
        return [x], LeafDef()

    node_type = _get_node_type(x)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(x)

    print(f"XX: {child_pytrees} {context}")

    result : List[Any] = []
    children_specs : List['TreeDef'] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten(child)
        result += flat
        children_specs.append(child_spec)
    return result, TreeDef(node_type, context, children_specs)


def tree_unflatten(xs: List[Any], treedef: TreeDef) -> Any:
    assert isinstance(treedef, TreeDef), f"{type(treedef)}"
    assert len(xs) == treedef.num_leaves, f"{xs} len: {len(xs)} <> {treedef} num_leaves: {treedef.num_leaves}"
    if isinstance(treedef, LeafDef):
        return xs[0]

    unflatten_fn = SUPPORTED_NODES[treedef.type].unflatten_fn

    start = 0
    end = 0
    child_pytrees = []
    for child_spec in treedef.children_specs:
        end += child_spec.num_leaves
        child_pytrees.append(tree_unflatten(xs[start:end], child_spec))
        start = end

    return unflatten_fn(child_pytrees, treedef.context)


class Empty: 
    pass
empty = Empty()

class Store:
  val = empty

  def set(self, val):
    self.val = empty
    self.val = val

  def __call__(self):
    return self.val

def fake_signature(fn, nargs):
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})

def flatten_func(f: Callable, in_tree: TreeDef) -> Tuple[Callable, TreeDef]:
    store = Store()

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(args_flat, in_tree)
        out = f(*pytree_args)
        print(f"PP1: {out}")
        out_flat, out_tree = tree_flatten(out)
        print(f"PP2: {out_flat}")
        store.set(out_tree)
        return out_flat

    return fake_signature(flat_fun, in_tree.num_leaves), store