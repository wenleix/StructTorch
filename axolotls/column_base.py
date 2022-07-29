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
        pass

    