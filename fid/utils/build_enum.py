import enum
import inspect
import types
import typing
from collections import OrderedDict


class FunctionWrapper(object):
    r"""Allow to wrap a function as an object.

    See https://stackoverflow.com/questions/31907060/python-3-enums-with-function-values
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# NB: Not inheriting from enum.EnumMeta.
#     This `cls` is really an regular class (rather than an enum class), whose
#     certain members are taken as the enum elements.
class BuildEnum(type):
    # This metaclass is entirely here to give the resulting enum class a custom
    # __getitem__ that supports looking up None.
    class enum_base_metacls(enum.EnumMeta):
        def __getitem__(cls, index):
            if index is None and cls.__None_member_actual_index__ is not None:
                index = cls.__None_member_actual_index__
            return super().__getitem__(index)

    class enum_base_cls(enum.Enum, metaclass=enum_base_metacls):
        pass

    enum_cls_made_this_way = []

    # decorator
    class None_indexed_member(object):
        def __init__(self, attr):
            self.attr = attr

        def __get__(self, obj, klass=None):
            return self.attr

    def call_enum_value(enum, *args, **kwargs):
        return enum.value(*args, **kwargs)

    @classmethod
    def process_enum_value(metacls, k, v):
        if isinstance(v, (types.FunctionType, types.MethodType)):
            return FunctionWrapper(v)
        return v

    @classmethod
    def auto(metacls):
        return enum.auto()

    # Remember that __new__ is implicitly a staticmethod for allowing handling
    # subclass creation.
    #
    # https://docs.python.org/3/reference/datamodel.html#object.__new__
    #
    # Context/debate on classmethod vs. staticmethod:
    #   https://stackoverflow.com/questions/9092072/why-isnt-new-in-python-new-style-classes-a-class-method
    def __new__(metacls, name, bases, dict):
        # NB: remove the metaclass dependency here by using `type` not `metacls`
        #     o.w. we can't use it as the mixin type for constructing the enum
        #     class because the enum base class does not have a metaclass that
        #     is subclass of `metacls`.
        cls = type.__new__(type, name, bases, dict)

        if '__members__' in dict:
            member_names = dict['__members__']
        else:
            member_names = tuple(k for k in dict.keys() if not k.startswith('_'))

        members_dict = OrderedDict()
        None_member_actual_index = None
        for k in member_names:
            # Why not just v = dict[k]?
            # We want to support things like classmethod and staticmethod.
            # However, those things are only bound through the descriptor
            # protocol, and are otherwise objects of type classmethod or
            # staticmethod.  Hence, we use getattr to access it via the
            # descriptor protocol.
            v = getattr(cls, k)
            # test None indexed member
            if isinstance(dict[k], metacls.None_indexed_member):
                assert None_member_actual_index is None, \
                    "Cannot have more than one member indexed by None: {} and {}".format(k, None_member_actual_index)
                None_member_actual_index = k
            members_dict[k] = metacls.process_enum_value(k, v)
        return metacls.make_enum_cls(name, members_dict, mixin_type=cls,  # mixin the original cls
                                     None_member_actual_index=None_member_actual_index)  # mixin the original cls

    @classmethod
    def make_enum_cls(metacls, name, members_dict, mixin_type=None, None_member_actual_index=None):
        if not isinstance(members_dict, typing.Mapping):
            members_dict = OrderedDict([(k, enum.auto()) for k in members_dict])
        enum_cls = metacls.enum_base_cls(name, names=members_dict, type=mixin_type, module=__name__)
        enum_cls.__call__ = metacls.call_enum_value
        enum_cls.__None_member_actual_index__ = None_member_actual_index
        BuildEnum.enum_cls_made_this_way.append(enum_cls)  # store all in BuildEnum, so not metacls
        return enum_cls

    @staticmethod
    def is_derived(cls):
        return inspect.isclass(cls) and issubclass(cls, tuple(BuildEnum.enum_cls_made_this_way))
