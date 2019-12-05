import os
import sys

# TODO: someday fix this path hack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import utils
from utils import state


class Test(object, metaclass=utils.types.options.BuildCompositeOptionEnum):
    @utils.types.options.BuildCompositeOptionEnum.None_indexed_member
    def none(input):
        pass

    @utils.types.options.BuildCompositeOptionEnum.add_options(
        state.Option('int_arg', type=int, default=4, desc='int'),
    )
    def a(input, int_arg):
        return 'a', input, int_arg

    def __b(input, str_arg):
        return 'b', input, str_arg

    b = [__b, [state.Option('str_arg', type=str, default='thisisatr', desc='str')]]


opt = state.Option('test', Test)

# FIXME: figure out a way to not directly use `from_yaml`
assert opt.from_yaml(None, [])(3) is None
assert opt.from_yaml(dict(name='a', options=dict()), [])(3) == ('a', 3, 4)
assert opt.from_yaml(dict(name='a', options=dict(int_arg=5)), [])(3) == ('a', 3, 5)
assert opt.from_yaml(dict(name='b', options=dict()), [])(6) == ('b', 6, 'thisisatr')
assert opt.from_yaml(dict(name='b', options=dict(str_arg='x')), [])(6) == ('b', 6, 'x')
