import collections

# option specific ones
from .state.options import types as options


def make_optional(type):
    return lambda x: None if x is None else type(x)


def make_typeslist(*types):
    types_str = ', '.join(map(repr, types))

    def converter(val):
        orig_val = val
        if isinstance(val, str):
            val = list(val.split(','))
        if isinstance(val, collections.abc.Sequence):
            if len(val) != len(types):
                raise RuntimeError("Expected sequence of length {} for type List[{}], but got: {}".format(
                    len(types), types_str, orig_val))
            return list(ty(elem) for ty, elem in zip(types, val))
        raise RuntimeError("Unsupported value for type List[{}]: {}".format(types_str, orig_val))

    return converter


def make_typelist(type):
    def converter(val):
        orig_val = val
        if isinstance(val, str):
            val = list(val.split(','))
        if isinstance(val, collections.abc.Sequence):
            return list(type(elem) for elem in val)
        raise RuntimeError("Unsupported value for type List[{}]: {}".format(type, orig_val))

    return converter


intlist = make_typelist(int)
floatlist = make_typelist(float)
strlist = make_typelist(str)

__all__ = [options, make_optional, make_typeslist, make_typelist, intlist, floatlist, strlist]
