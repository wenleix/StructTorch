# Based on https://github.com/pytorch/torcharrow/blob/ba822da72825150e971c3ee7b6dcb81668eb79f2/torcharrow/dtypes.py
# Changes may apply

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import re
import typing as ty
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass, replace
import torch
import sys

# -----------------------------------------------------------------------------
# Aux

# Pretty printing constants; reused everywhere
OPEN = "{"
CLOSE = "}"
NL = "\n"

# Handy Type abbreviations; reused everywhere
ScalarTypes = ty.Union[int, float, bool, str]


# -----------------------------------------------------------------------------
# Schema and Field

MetaData = ty.Dict[str, str]


@dataclass(frozen=True)
class Field:
    name: str
    dtype: "DType"
    metadata: ty.Optional[MetaData] = None

    def __str__(self):
        meta = ""
        if self.metadata is not None:
            meta = (
                f"meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
            )
        return f"Field('{self.name}', {str(self.dtype)}{meta})"


# -----------------------------------------------------------------------------
# Immutable Types with structural equality...


@dataclass(frozen=True)  # type: ignore
class DType(ABC):
    fields: ty.ClassVar[ty.Any] = NotImplementedError

    typecode: ty.ClassVar[str] = "__TO_BE_DEFINED_IN_SUBCLASS__"
    arraycode: ty.ClassVar[str] = "__TO_BE_DEFINED_IN_SUBCLASS__"

    @property
    @abstractmethod
    def nullable(self):
        return False

    @property
    def py_type(self):
        return type(self.default_value())

    def __str__(self):
        if self.nullable:
            return f"{self.name.title()}(nullable=True)"
        else:
            return self.name

    @abstractmethod
    def constructor(self, nullable):
        pass

    def with_null(self, nullable=True):
        return self.constructor(nullable)

    def default_value(self):
        # must be overridden by all non primitive types!
        return type(self).default

    def __qualstr__(self):
        return "torcharrow.dtypes"


# for now: no float16, and all date and time stuff, no categorical, (and Null is called Void)


@dataclass(frozen=True)
class Void(DType):
    nullable: bool = True
    typecode: ty.ClassVar[str] = "n"
    arraycode: ty.ClassVar[str] = "b"
    name: ty.ClassVar[str] = "void"
    default: ty.ClassVar[ty.Optional[bool]] = None

    def constructor(self, nullable):
        return Void(nullable)


@dataclass(frozen=True)  # type: ignore
class Numeric(DType):
    pass


@dataclass(frozen=True)
class Boolean(DType):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "b"
    arraycode: ty.ClassVar[str] = "b"
    name: ty.ClassVar[str] = "boolean"
    default: ty.ClassVar[bool] = False

    def constructor(self, nullable):
        return Boolean(nullable)


@dataclass(frozen=True)
class Int8(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "c"
    arraycode: ty.ClassVar[str] = "b"
    name: ty.ClassVar[str] = "int8"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int8(nullable)


@dataclass(frozen=True)
class Int16(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "s"
    arraycode: ty.ClassVar[str] = "h"
    name: ty.ClassVar[str] = "int16"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int16(nullable)


@dataclass(frozen=True)
class Int32(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "i"
    arraycode: ty.ClassVar[str] = "i"
    name: ty.ClassVar[str] = "int32"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int32(nullable)


@dataclass(frozen=True)
class Int64(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "l"
    arraycode: ty.ClassVar[str] = "l"
    name: ty.ClassVar[str] = "int64"
    default: ty.ClassVar[int] = 0

    def constructor(self, nullable):
        return Int64(nullable)


# Not all Arrow types are supported. We don't have a backend to support unsigned
# integer types right now so they are removed to not confuse users. Feel free to
# add unsigned int types when we have a supporting backend.


@dataclass(frozen=True)
class Float32(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "f"
    arraycode: ty.ClassVar[str] = "f"
    name: ty.ClassVar[str] = "float32"
    default: ty.ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float32(nullable)


@dataclass(frozen=True)
class Float64(Numeric):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "g"
    arraycode: ty.ClassVar[str] = "d"
    name: ty.ClassVar[str] = "float64"
    default: ty.ClassVar[float] = 0.0

    def constructor(self, nullable):
        return Float64(nullable)


@dataclass(frozen=True)
class String(DType):
    nullable: bool = False
    typecode: ty.ClassVar[str] = "u"  # utf8 string (n byte)
    arraycode: ty.ClassVar[str] = "w"  # wchar_t (2 byte)
    name: ty.ClassVar[str] = "string"
    default: ty.ClassVar[str] = ""

    def constructor(self, nullable):
        return String(nullable)


@dataclass(frozen=True)
class Map(DType):
    key_dtype: DType
    item_dtype: DType
    nullable: bool = False
    keys_sorted: bool = False
    name: ty.ClassVar[str] = "Map"
    typecode: ty.ClassVar[str] = "+m"
    arraycode: ty.ClassVar[str] = ""

    @property
    def py_type(self):
        return ty.Dict[self.key_dtype.py_type, self.item_dtype.py_type]

    def constructor(self, nullable):
        return Map(self.key_dtype, self.item_dtype, nullable)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        return f"Map({self.key_dtype}, {self.item_dtype}{nullable})"

    def default_value(self):
        return {}


@dataclass(frozen=True)
class List(DType):
    item_dtype: DType
    nullable: bool = False
    fixed_size: int = -1
    name: ty.ClassVar[str] = "List"
    typecode: ty.ClassVar[str] = "+l"
    arraycode: ty.ClassVar[str] = ""

    @property
    def py_type(self):
        return ty.List[self.item_dtype.py_type]

    def constructor(self, nullable, fixed_size=-1):
        return List(self.item_dtype, nullable, fixed_size)

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        fixed_size = (
            ", fixed_size=" + str(self.fixed_size) if self.fixed_size >= 0 else ""
        )
        return f"List({self.item_dtype}{nullable}{fixed_size})"

    def default_value(self):
        return []


@dataclass(frozen=True)
class Struct(DType):
    fields: ty.List[Field]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: ty.Optional[MetaData] = None
    name: ty.ClassVar[str] = "Struct"
    typecode: ty.ClassVar[str] = "+s"
    arraycode: ty.ClassVar[str] = ""

    # For generating NamedTuple class name for cached _py_type (done in __post__init__)
    _global_py_type_id: ty.ClassVar[int] = 0
    _local_py_type_id: int = dataclasses.field(compare=False, default=-1)

    # TODO: perhaps this should be a private method
    def get_index(self, name: str) -> int:
        for idx, field in enumerate(self.fields):
            if field.name == name:
                return idx
        # pyre-fixme[7]: Expected `int` but got `None`.
        return None

    def __getstate__(self):
        # _py_type is NamedTuple which is not pickle-able, skip it
        return (self.fields, self.nullable, self.is_dataframe, self.metadata)

    def __setstate__(self, state):
        # Restore state, __setattr__ hack is needed due to the frozen dataclass
        object.__setattr__(self, "fields", state[0])
        object.__setattr__(self, "nullable", state[1])
        object.__setattr__(self, "is_dataframe", state[2])
        object.__setattr__(self, "metadata", state[3])

        # reconstruct _py_type
        self.__post_init__()

    def __post_init__(self):
        if self.nullable:
            for f in self.fields:
                if not f.dtype.nullable:
                    raise TypeError(
                        f"nullable structs require each field (like {f.name}) to be nullable as well."
                    )
        object.__setattr__(self, "_local_py_type_id", type(self)._global_py_type_id)
        type(self)._global_py_type_id += 1

    def _set_py_type(self):
        # cache the type instance, __setattr__ hack is needed due to the frozen dataclass
        # the _py_type is not listed above to avoid participation in equality check

        def fix_name(name, idx):
            # Anonomous Row
            if name == "":
                return "f_" + str(idx)

            # Remove invalid character for NamedTuple
            # TODO: this might cause name duplicates, do disambiguation
            name = re.sub("[^a-zA-Z0-9_]", "_", name)
            if name == "" or name[0].isdigit() or name[0] == "_":
                name = "f_" + name
            return name

        object.__setattr__(
            self,
            "_py_type",
            ty.NamedTuple(
                "TorchArrowGeneratedStruct_" + str(self._local_py_type_id),
                [
                    (fix_name(f.name, idx), f.dtype.py_type)
                    for (idx, f) in enumerate(self.fields)
                ],
            ),
        )

    @property
    def py_type(self):
        if not hasattr(self, "_py_type"):
            # this call is expensive due to the namedtuple creation, so
            # do it lazily
            self._set_py_type()
        return self._py_type

    def constructor(self, nullable):
        return Struct(self.fields, nullable)

    def get(self, name):
        for f in self.fields:
            if f.name == name:
                return f.dtype
        raise KeyError(f"{name} not among fields")

    def __str__(self):
        nullable = ", nullable=" + str(self.nullable) if self.nullable else ""
        fields = f"[{', '.join(str(f) for f in self.fields)}]"
        meta = ""
        if self.metadata is not None:
            meta = f", meta = {OPEN}{', '.join(f'{k}: {v}' for k,v in self.metadata)}{CLOSE}"
        else:
            return f"Struct({fields}{nullable}{meta})"

    def default_value(self):
        return tuple(f.dtype.default_value() for f in self.fields)


# only used internally for type inference -------------------------------------


@dataclass(frozen=True)
class Tuple(DType):
    fields: ty.List[DType]
    nullable: bool = False
    is_dataframe: bool = False
    metadata: ty.Optional[MetaData] = None
    name: ty.ClassVar[str] = "Tuple"
    typecode: ty.ClassVar[str] = "+t"
    arraycode: ty.ClassVar[str] = ""

    @property
    def py_type(self):
        return tuple

    def constructor(self, nullable):
        return Tuple(self.fields, nullable)

    def default_value(self):
        return tuple(f.dtype.default_value() for f in self.fields)


@dataclass(frozen=True)
class Any(DType):
    nullable: bool = True
    typecode: ty.ClassVar[str] = "?"
    arraycode: ty.ClassVar[str] = "?"
    name: ty.ClassVar[str] = "any"
    default: ty.ClassVar[ty.Optional[bool]] = None

    @property
    def size(self):
        # currently 1 byte per bit
        raise ValueError("Shouldn't be called")

    @property
    def py_type(self):
        raise ValueError("Shouldn't be called")

    def constructor(self, nullable=True):
        assert nullable
        return Any()


# TorchArrow does not yet support these types ---------------------------------
Tag = str

# abstract


@dataclass(frozen=True)  # type: ignore
class Union_(DType):
    pass


@dataclass(frozen=True)  # type: ignore
class DenseUnion(DType):
    tags: ty.List[Tag]
    name: ty.ClassVar[str] = "DenseUnion"
    typecode: ty.ClassVar[str] = "+ud"
    arraycode: ty.ClassVar[str] = ""


@dataclass(frozen=True)  # type: ignore
class SparseUnion(DType):
    tags: ty.List[Tag]
    name: ty.ClassVar[str] = "SparseUnion"
    typecode: ty.ClassVar[str] = "+us"
    arraycode: ty.ClassVar[str] = ""


boolean = Boolean()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()
string = String()

# Type test -------------------------------------------------------------------
# can be deleted once TorchArrow is implemented over velox...


def is_void(t):
    """
    Return True if value is an instance of a void type.
    """
    # print('is_boolean', t.typecode)
    return t.typecode == "n"


def is_boolean(t):
    """
    Return True if value is an instance of a boolean type.
    """
    # print('is_boolean', t.typecode)
    return t.typecode == "b"


def is_boolean_or_numerical(t):
    return is_boolean(t) or is_numerical(t)


def is_numerical(t):
    return is_integer(t) or is_floating(t)


def is_integer(t):
    """
    Return True if value is an instance of any integer type.
    """
    return t.typecode in "csilCSIL"


def is_signed_integer(t):
    """
    Return True if value is an instance of any signed integer type.
    """
    return t.typecode in "csil"


def is_int8(t):
    """
    Return True if value is an instance of an int8 type.
    """
    return t.typecode == "c"


def is_int16(t):
    """
    Return True if value is an instance of an int16 type.
    """
    return t.typecode == "s"


def is_int32(t):
    """
    Return True if value is an instance of an int32 type.
    """
    return t.typecode == "i"


def is_int64(t):
    """
    Return True if value is an instance of an int64 type.
    """
    return t.typecode == "l"


def is_floating(t):
    """
    Return True if value is an instance of a floating point numeric type.
    """
    return t.typecode in "fg"


def is_float32(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode == "f"


def is_string(t):
    return t.typecode == "u"


def is_float64(t):
    """
    Return True if value is an instance of a float32 (single precision) type.
    """
    return t.typecode == "g"


def is_list(t):
    return t.typecode.startswith("+l")


def is_map(t):
    return t.typecode.startswith("+m")


def is_struct(t):
    return t.typecode.startswith("+s")


def is_primitive(t):
    return t.typecode[0] != "+"


def is_tuple(t):
    return t.typecode.startswith("+t")


def contains_tuple(t: DType):
    if is_tuple(t):
        return True
    if is_list(t):
        # pyre-fixme[16]: `DType` has no attribute `item_dtype`.
        return contains_tuple(t.item_dtype)
    if is_map(t):
        # pyre-fixme[16]: `DType` has no attribute `key_dtype`.
        return contains_tuple(t.key_dtype) or contains_tuple(t.item_dtype)
    if is_struct(t):
        return any(contains_tuple(f.dtype) for f in t.fields)

    return False


def is_any(t):
    return t.typecode == "?"


def _dtype_from_pytorch_dtype(dtype: torch.dtype, nullable: bool = False) -> DType:
    if dtype == torch.int32:
        return Int32(nullable=nullable)
    if dtype == torch.int64:
        return Int64(nullable=nullable)

    raise ValueError(f"Unsupported PyTorch dtype: {dtype}")
