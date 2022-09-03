from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
import operator
from dataclasses import dataclass, field, fields
import torch.fx
import torch
from tabulate import tabulate

Target = Union[Callable[..., Any], str]


def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)


@dataclass
class Node:
    name: str
    op: str
    target: Target
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def __repr__(self):
        return self.name

    __str__ = __repr__

@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    node_names: Dict[str, int] = field(default_factory=dict)

    def create_node(self, op: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], name: Optional[str] = None,) -> Node:
        candidate = name if name is not None else self._target_to_str(target)
        
        cand_cnt = self.node_names.setdefault(candidate, 0)
        cand_name = f"{candidate}_{cand_cnt}"
        
        node = Node(cand_name, op, target, args, kwargs)
        self.nodes.append(node)
        self.node_names[candidate] += 1
        return node

    def __len__(self) -> int:
        return len(self.nodes)

    def _target_to_str(self, target : Target) -> str:
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            if op.startswith('__') and op.endswith('__'):
                op = op[2:-2]
        op = _snake_case(op)
        return op

    def __repr__(self):
        rows = []
        header = []
        for node in self.nodes:
            node_fields = fields(node)
            header = [fld.name for fld in node_fields]
            row = [getattr(node, name) for name in header]
            rows.append(row)

        tab = tabulate(
            rows, headers=["index"] + header, tablefmt="simple", showindex=True
        )
        return tab


        return "\n".join([str(node) for node in self.nodes])

    __str__ = __repr__

@dataclass
class Proxy:
    node: Node
    tracer: "SimpleTracer"

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    __str__ = __repr__

    def __getitem__(self, key) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.getitem, (self, key), {})

    def __setitem__(self, key, val) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.setitem, (self, key,  val), {})

    def __add__(self, other) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.add, (self, other), {})

    def __mul__(self, other) -> "Proxy":
        return self.tracer.create_proxy("call_function", operator.mul, (self, other), {})

    def __getattr__(self, key: str) -> "Attribute":
        return Attribute(self, key)

    def __call__(self, *args, **kwargs) -> 'Proxy':
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers : Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            return tracer.create_proxy('call_function', orig_method, args, kwargs)
    

class Attribute(Proxy):
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)
