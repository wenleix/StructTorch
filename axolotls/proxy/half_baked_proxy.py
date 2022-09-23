from typing import Any, Dict, Optional 
import operator
from dataclasses import dataclass
import torch.fx
import torch
from tabulate import tabulate
from .graph import Graph, Node, Target


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
