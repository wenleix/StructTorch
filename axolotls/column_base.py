from abc import ABC, abstractmethod, ABCMeta
from . import dtypes as dt
from tabulate import tabulate
import torch.fx
from enum import Enum
from .proxy.proxy_base import Proxy

class TraceMode(Enum):
    HALF_BAKED = "half_baked"
    TORCH_FX = "torch_fx"
    JAX_MIMIC = "jax_mimic"

TRACE_MODE = TraceMode.JAX_MIMIC

class ColumnMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        torch.fx.node.map_aggregate(args, check_proxy)
        torch.fx.node.map_aggregate(kwargs, check_proxy)
        
        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy('call_function', cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance


class ColumnBase(ABC, metaclass=ColumnMeta):
    @abstractmethod
    def __init__(self, dtype: dt.DType) -> None:
        self._dtype = dtype

    @property
    def dtype(self) -> dt.DType:
        return self._dtype
    
    @abstractmethod
    def clone(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> None:
        rows = [[self[idx] if self[idx] is not None else "None"] for idx in range(len(self))]
        tab = tabulate(
            rows,
            tablefmt="plain",
            showindex=True,
        )
        typ = (
            f"dtype: {self._dtype}, length: {len(self)}"
        )
        return tab + "\n" + typ

    @abstractmethod
    def to_arrow(self):
        """
        Experimental interop API with soft dependency on PyArrow
        """
        raise NotImplementedError
