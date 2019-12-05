from . import options
from . import namespace


class State(object):
    def __init__(self, wrapped_module):
        # You have to retain the old module, otherwise it will
        # get GC'ed and a lot of things will break.  See:
        # https://stackoverflow.com/questions/47540722/how-do-i-use-the-sys-modules-replacement-trick-in-init-py-on-python-2
        self.__wrapped_module = wrapped_module
        # add this first
        self.add_option('config', type=lambda x: None if x is None else str(x),
                        default=None, desc="Path to the yaml config file",
                        kind=self.Option.ARGPARSE_POSITIONAL_ONLY)

    __desc = None

    @staticmethod
    def set_desc(desc):
        assert State.__desc is None
        State.__desc = desc

    @staticmethod
    def get_desc():
        assert State.__desc is not None
        return State.__desc

    def __repr__(self):
        if hasattr(self, 'yaml_display_config'):
            return namespace.display_yaml(self.yaml_display_config)
        else:
            return super().__repr__()

    ignore_option_related = staticmethod(options.ignore_option_related)
    overwrite = staticmethod(namespace.overwrite)
    option_namespace = staticmethod(options.option_namespace)
    add_option = staticmethod(options.add_option)
    register_parse_hook = staticmethod(options.register_parse_hook)
    parse_options = options.parse_options
    options_parsed = staticmethod(options.options_parsed)
    Option = options.Option

    @property
    def option_types(self):
        return options.option_types

    def __setattr__(self, key, value):
        return namespace.Namespace._setattr_to_namespace(self, 'yaml_display_config', key, value)
