from abc import ABC, abstractmethod
from . import dtypes as dt

class ColumnBase(ABC):
    @abstractmethod
    def __init__(self, dtype: dt.DType) -> None:
        self._dtype = dtype

    @property
    def dtype(self) -> dt.DType:
        return self._dtype
    
    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
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

    def __str__(self) -> str:
        return self.__repr__()


    @abstractmethod
    def to_arrow(self):
        """
        Experimental interop API with soft dependency on PyArrow
        """
        raise NotImplementedError

    