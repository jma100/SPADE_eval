import argparse
import collections
import inspect
import typing

from ...build_enum import BuildEnum
from .options import Option


class DictOptionType(type):
    def __init__(cls, *suboptions):
        assert all(isinstance(opt, Option) for opt in suboptions)
        cls.suboptions = suboptions

    def __call__(cls, yaml_val):
        assert isinstance(yaml_val, collections.abc.Mapping)
        assert set(yaml_val.keys()).issubset(set(subopt.name for subopt in cls.suboptions))
        res = argparse.Namespace()
        for subopt in cls.suboptions:
            val = yaml_val.get(subopt.name, subopt.default)
            val = subopt.from_yaml(val)
            setattr(res, subopt.name, val)
        return res

    def additional_constraint(cls, value):
        for subopt in cls.suboptions:
            if subopt.constraint is not None:
                if not subopt.constraint(getattr(value, subopt.name)):
                    return False
        return True


class BuildCompositeOptionEnum(BuildEnum):
    CompositeEnumElement = collections.namedtuple('CompositeEnumElement', ['value', 'suboptions'], module=__name__)

    # Value users get from parsing
    class CompositeParseResult(collections.namedtuple('CompositeParseResult', ['enum_obj', 'suboption_actuals'],
                                                      module=__name__)):
        # enum_obj: enum  # enum object
        # value: typing.Any
        # suboption_actuals: typing.Mapping[str, typing.Any]

        @property
        def actual_obj(self):
            return self.enum_obj.value.value  # first .value gives a CompositeEnumElement

        def __call__(self, *args, **kwargs):
            # TODO: check signature here
            for key, default in self.suboption_actuals.items():
                kwargs.setdefault(key, default)
            return self.actual_obj(*args, **kwargs)

        def __eq__(self, other):
            return self.enum_obj == other or self.actual_obj == other or super().__eq__(other)

        def __getattr__(self, key):
            if key in self.suboption_actuals:
                return self.suboption_actuals[key]
            return getattr(self.actual_obj, key)

        def __repr__(self):
            return repr(self.actual_obj)

    # type
    class enum_base_metacls(BuildEnum.enum_base_metacls):
        # signature from https://github.com/python/cpython/blob/888f37bc2826d9ab2cbec6f153e7f58a34785c4a/Lib/enum.py#L278
        def __call__(enum_cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
            if names is None:  # simple value lookup
                return enum_cls.from_yaml(value)
            return super().__call__(value, names, module=module,
                                    qualname=qualname, type=type, start=start)

        def expand_into_full_yaml(enum_cls, yaml_val):
            # This method returns a full yaml with all suboptions for the
            # selected value filled.
            #
            # Note that users can pass in a simple value if the default
            # options for that value are intended to use.

            simple = not isinstance(yaml_val, (typing.Mapping, tuple, list))

            if not simple:
                keys = set(yaml_val.keys())
                if keys != {'name', 'options'}:
                    raise argparse.ArgumentTypeError(
                        '{}: is composite and expects keys {{"name", "options"}}, but got keys {}'.format(
                            enum_cls.__name__, keys))

            user_name = yaml_val if simple else yaml_val['name']
            user_suboption_vals = {} if simple else yaml_val['options']

            result = collections.OrderedDict()

            result['name'] = user_name
            suboptions = enum_cls[user_name].value[1]

            expect_key_set = set(opt.name for opt in suboptions)
            actual_key_set = set(user_suboption_vals.keys())
            if not actual_key_set.issubset(expect_key_set):
                raise argparse.ArgumentTypeError(
                    'choice "{}": expects suboptions with keys in {}, but got keys {}'.format(
                        user_name, expect_key_set, actual_key_set))

            suboption_vals = collections.OrderedDict()
            for subopt in suboptions:
                val = subopt.expand_into_full_yaml(
                    user_suboption_vals.get(subopt.name, subopt.default),
                    [enum_cls.__name__, user_name, subopt.name])
                suboption_vals[subopt.name] = val

            result['options'] = suboption_vals
            return result

        def from_yaml(enum_cls, yaml_val):
            yaml_val = enum_cls.expand_into_full_yaml(yaml_val)

            user_name = yaml_val['name']
            yaml_options = yaml_val['options']

            enum_obj = enum_cls[user_name]
            _, suboptions = enum_obj.value

            suboption_actuals = collections.OrderedDict()
            for subopt in suboptions:  # stay true to the order in spec
                suboption_actuals[subopt.name] = subopt.from_yaml(
                    yaml_val['options'][subopt.name],
                    [enum_cls.__name__, user_name, subopt.name])

            return BuildCompositeOptionEnum.CompositeParseResult(enum_obj, suboption_actuals)

        def additional_constraint(enum_cls, parse_result):
            for subopt in parse_result.enum_obj.value.suboptions:
                if subopt.constraint is not None:
                    if not subopt.constraint(parse_result.suboption_actuals[subopt.name]):
                        return False
            return True

    class enum_base_cls(BuildEnum.enum_base_cls, metaclass=enum_base_metacls):
        def __getattr__(self, name):
            if name == 'options':  # give user a way to access suboptions from option_types
                return self.value[1]
            else:
                return super().__getattr__(name)

    @classmethod
    def process_enum_value(cls, k, v):
        # extract options since the super().process_enum_value call may
        # return a new object.
        if not isinstance(v, collections.abc.Sequence):
            if hasattr(v, 'options'):
                options = v.options
            else:
                options = []
        else:
            assert len(v) == 2
            v, options = v
            assert not hasattr(v, 'options')
        if isinstance(options, Option):
            options = [options]
        assert isinstance(options, collections.abc.Sequence)
        for opt in options:
            assert isinstance(opt, Option)
        v = super().process_enum_value(k, v)
        return BuildCompositeOptionEnum.CompositeEnumElement(v, options)

    # util decorator for users to add options
    @staticmethod
    def add_options(*options):
        def decorator(obj):
            assert not hasattr(obj, 'options')
            obj.options = options
            return obj
        return decorator

    def __new__(cls, name, bases, dict):
        enum_cls = super().__new__(cls, name, bases, dict)
        for name, enum_item in enum_cls.__members__.items():
            (value, options) = enum_item.value
            if not callable(value):
                continue
            sig = inspect.signature(value, follow_wrapped=True)
            try:
                # check that given suboptions partially match
                sig.bind_partial(**{opt.name: None for opt in options})
            except TypeError as e:
                err_msg = "{}: expect {} (value for '{}') to accept given suboption names {} as kwargs".format(
                    enum_cls, value, name, tuple(opt.name for opt in options))
                raise TypeError(err_msg) from e
        return enum_cls
