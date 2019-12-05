import os
import typing


__all__ = ['mkdirs', 'mkdir']


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def mkdirs(paths):
    if not isinstance(paths, typing.Iterable) or isinstance(paths, str):  # we py3
        paths = (paths,)
    for path in paths:
        mkdir(path)
