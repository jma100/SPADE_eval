import argparse
import collections
import contextlib
import enum
import copy
import inspect
import traceback
import functools
import oyaml as yaml
import typing

from .. import namespace


############################
# Options
############################

# General switches
# ----------------

_ignore_option_related = False
_options_parsed = False


@contextlib.contextmanager
def ignore_option_related(value=True):
    global _ignore_option_related
    orig = _ignore_option_related
    _ignore_option_related = value
    yield
    _ignore_option_related = orig


def _assert_options_not_parsed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        assert _ignore_option_related or not _options_parsed
        return fn(*args, **kwargs)
    return wrapper


def _skip_if_ignoring_option_related(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _ignore_option_related:
            return
        return fn(*args, **kwargs)
    return wrapper


def options_parsed():
    return _options_parsed

# Option, OptionType, and type
# ----------------------------

required = object()


class _OptionKind(enum.Enum):
    ARGPARSE_POSITIONAL_ONLY = enum.auto()
    ARGPARSE_KEYWORD_ONLY = enum.auto()
    YAML_AND_ARGPARSE_KEYWORD_ONLY = enum.auto()


class Option(collections.namedtuple('Option', ['name', 'type', 'constraint', 'default', 'desc', 'kind'],
                                    module=__name__)):
    ARGPARSE_POSITIONAL_ONLY = _OptionKind.ARGPARSE_POSITIONAL_ONLY
    ARGPARSE_KEYWORD_ONLY = _OptionKind.ARGPARSE_KEYWORD_ONLY
    YAML_AND_ARGPARSE_KEYWORD_ONLY = _OptionKind.YAML_AND_ARGPARSE_KEYWORD_ONLY

    def __new__(cls, name, type, constraint=None, default=required, desc="",
                kind=YAML_AND_ARGPARSE_KEYWORD_ONLY):
        return super().__new__(cls, name, type, constraint, default, desc, kind)

    def expand_into_full_yaml(self, yaml_val, namespace_list):
        if hasattr(self.type, 'expand_into_full_yaml'):
            try:
                return self.type.expand_into_full_yaml(yaml_val)
            except Exception as e:
                key_str = '.'.join(namespace_list + [self.name])
                raise RuntimeError('option "{}" errors in expand_into_full_yaml'.format(key_str)) from e
        return yaml_val

    def from_yaml(self, yaml_val, namespace_list):
        try:
            if yaml_val is required:
                raise RuntimeError("Option is required, but not set")
            if hasattr(self.type, 'from_yaml'):
                return self.type.from_yaml(yaml_val)
            elif inspect.isclass(self.type) and issubclass(self.type, enum.Enum):
                return self.type[yaml_val]
            elif isinstance(self.type, collections.abc.Mapping):
                return self.type[yaml_val]
            else:
                return self.type(yaml_val)
        except Exception as e:
            key_str = '.'.join(namespace_list + [self.name])
            raise RuntimeError('option "{}" errors in from_yaml'.format(key_str)) from e

    def verify_constraint(self, value, namespace_list):
        try:
            if hasattr(self.type, "additional_constraint"):  # for constraints like suboptions
                assert self.type.additional_constraint(value)
            if self.constraint is not None:
                assert self.constraint(value)
        except Exception as e:
            key_str = '.'.join(namespace_list + [self.name])
            raise RuntimeError('option "{}" with value {} failed to satisfy constraints'.format(key_str, value)) from e

# add_option
# ----------

options = collections.OrderedDict()
# used for constraint verification, and thus should be real enum classes
# TODO: freeze it in user space
option_types = argparse.Namespace()

current_namespace = []


@contextlib.contextmanager
def option_namespace(name):
    current_namespace.append(name)
    yield
    assert current_namespace.pop() == name


@_skip_if_ignoring_option_related
@_assert_options_not_parsed
def add_option(*args, **kwargs):
    option = Option(*args, **kwargs)

    assert len(current_namespace) == 0 or option.kind == Option.YAML_AND_ARGPARSE_KEYWORD_ONLY, \
        "argparse option doesn't support nested namespacing"

    opt_store = options
    option_type_store = option_types
    for i, ns in enumerate(current_namespace):
        if ns not in opt_store:
            opt_store[ns] = collections.OrderedDict()
            setattr(option_type_store, ns, argparse.Namespace())
        opt_store = opt_store[ns]
        option_type_store = getattr(option_type_store, ns)
        assert isinstance(opt_store, collections.OrderedDict), \
            "{} overwrites existing option {}".format(
                '.'.join(current_namespace + [name]),
                '.'.join(current_namespace[:(i + 1)]))

    def fmt(name, x):
        if isinstance(x, Option):
            return 'option {}'.format(name)
        else:
            return 'namespace {}'.format(name)

    assert option.name not in opt_store, "{} overwrites existing {}".format(
        '.'.join(current_namespace + [option.name]), fmt(opt_store[option.name]))

    opt_store[option.name] = option
    setattr(option_type_store, option.name, option.type)

# option parsing
# --------------

parse_hooks = []


@_skip_if_ignoring_option_related
@_assert_options_not_parsed
def register_parse_hook(hook):
    # Doesn't support removing
    parse_hooks.append(hook)


@_assert_options_not_parsed
def parse_options(state):
    parser = argparse.ArgumentParser(description=state.get_desc(),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ############################
    # deal with argparse options
    ############################

    # argparse options are only allowed at top-level namespace for now

    # FIXME: allow customizing argparse str => object, maybe `opt.type.from_str`?

    # new type returns (string_val, val, use_this_val=(specified or argparse_only))
    def make_str_type(opt, argparse_only):
        return lambda x, is_default=False: (x, opt.from_yaml(x, []), argparse_only or not is_default)

    for name, opt in options.items():
        if isinstance(opt, typing.Mapping):
            continue

        new_ty = make_str_type(opt, opt.kind != Option.YAML_AND_ARGPARSE_KEYWORD_ONLY)
        kwargs = {}

        if opt.default is not required:
            kwargs['default'] = new_ty(opt.default, is_default=True)
            if opt.kind == Option.ARGPARSE_POSITIONAL_ONLY:
                kwargs['nargs'] = '?'
        else:
            kwargs['default'] = (None, None, False)  # dummy default that will never be used

        if opt.kind == Option.ARGPARSE_POSITIONAL_ONLY:
            new_name = name
        else:
            new_name = '--' + name

        parser.add_argument(new_name, type=new_ty, help=opt.desc, **kwargs)

    argparse_ns = parser.parse_args()

    argparse_strs = collections.OrderedDict(
        [(k, s) for k, (s, v, use_this_val) in vars(argparse_ns).items() if use_this_val])
    argparse_actuals = collections.OrderedDict(
        [(k, v) for k, (s, v, use_this_val) in vars(argparse_ns).items() if use_this_val])

    def in_argparse(key):
        return key in argparse_strs

    user_config_path = argparse_actuals['config']
    if user_config_path is not None:
        with open(user_config_path, 'r') as config_f:
            yaml_user_config = yaml.load(config_f, Loader=yaml.SafeLoader) or {}  # None/[] -> {}
    else:
        yaml_user_config = collections.OrderedDict()
    yaml_user_config.update(argparse_strs)

    # helper
    namespace_list = []

    @contextlib.contextmanager
    def recurse_namespace(subnamespace_name):
        namespace_list.append(subnamespace_name)
        yield
        assert namespace_list.pop() == subnamespace_name

    ####################
    # add default config
    ####################

    # why new dst?  because yaml_user_config may not be ordered properly.
    def add_default(src, dst, options=options):
        assert isinstance(src, typing.Mapping), \
            "user yaml config for namespace [{}] not respecting spec, got: {}".format('.'.join(namespace_list), src)
        assert set(src.keys()).issubset(set(options.keys())), \
            "expected keys under [{}] to be subset of {}, got: {}".format('.'.join(namespace_list), set(options.keys()),
                                                                          set(src.keys()))
        for k, opt in options.items():  # iterate by options
            if isinstance(opt, typing.Mapping):
                with recurse_namespace(k):
                    dst[k] = add_default(src.get(k, collections.OrderedDict()),
                                         dst=collections.OrderedDict(), options=opt)
            elif k not in src:
                dst[k] = opt.default
            else:
                dst[k] = opt.expand_into_full_yaml(src[k], namespace_list)
        return dst

    yaml_config = add_default(yaml_user_config, dst=collections.OrderedDict())

    ###########################
    # convert to actual objects
    ###########################

    def convert(src, dst, options=options):
        for k, v in src.items():
            full_key = '.'.join(namespace_list + [k])
            if isinstance(options[k], typing.Mapping):  # decide ns using spec
                with recurse_namespace(k):
                    v = convert(v, dst=collections.OrderedDict(), options=options[k])
            elif in_argparse(full_key):  # FIXME: unify yaml and argparse results better
                v = argparse_actuals[full_key]
            elif v is required:
                raise argparse.ArgumentTypeError('required option [{}] is missing'.format(full_key))
            else:
                v = options[k].from_yaml(v, namespace_list)
            dst[k] = v
        return dst

    obj_config = convert(yaml_config, dst=collections.OrderedDict())

    #######################
    # generate yaml configs
    #######################

    yaml_full_config = yaml_config
    yaml_display_config = copy.deepcopy(yaml_config)
    yaml_user_config = yaml_user_config

    def set_as_attr(src, dst=state, display_config=yaml_display_config,
                    options=options):
        for k, v in src.items():
            if isinstance(options[k], typing.Mapping):  # decide ns using spec
                sub_display_config = display_config[k]
                v = set_as_attr(v, dst=namespace.Namespace(sub_display_config),
                                display_config=sub_display_config,
                                options=options[k])
            assert not hasattr(dst, k)
            setattr(dst, k, v)
        return dst

    obj_config.update(argparse_actuals)
    set_as_attr(obj_config)

    ####################
    # assign yamls
    ####################

    assert not hasattr(state, 'yaml_full_config')
    assert not hasattr(state, 'yaml_display_config')
    assert not hasattr(state, 'yaml_user_config')
    state.yaml_full_config = yaml_full_config
    state.yaml_display_config = yaml_display_config
    state.yaml_user_config = yaml_user_config

    ####################
    # verify constraints
    ####################

    def verify(owner=state, options=options):
        for k, opt in options.items():
            if isinstance(opt, typing.Mapping):  # decide ns using spec
                with recurse_namespace(k):
                    verify(getattr(owner, k), options=opt)
            else:
                opt.verify_constraint(getattr(owner, k), namespace_list)

    verify()

    #####################
    # execute parse hooks
    #####################

    # Some hooks rely on others to be executed first and setting required
    # attributes.
    # We employ a simple mechanism and just try the hooks sequentially N
    # times, and if any hook fails, just retry in the next time.
    # Of course this is assuming that the hooks are re-entrant...
    N = 10
    exception_strs = []
    for n in range(N):
        for idx in reversed(range(len(parse_hooks))):
            try:
                parse_hooks[idx]()
                del parse_hooks[idx]
            except AttributeError:
                if n == N - 1:
                    exception_strs.append(traceback.format_exc())

    if len(exception_strs) > 0:
        error_str = '{} exception(s) raised during last iteration of parse hooks:\n'.format(len(exception_strs))
        for s in exception_strs:
            error_str += '\n{}'.format(s).replace('\n', '\n\t')
        raise RuntimeError(error_str)

    global _options_parsed
    _options_parsed = True
