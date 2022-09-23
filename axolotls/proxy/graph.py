from typing import Any, Callable, Dict, List, Optional, Tuple, Union 
from dataclasses import dataclass, field, fields
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
