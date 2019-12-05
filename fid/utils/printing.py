import torch


# TODO: add zero padded int
def format_str_one(v, float_prec=6):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        v = v.item()
    if isinstance(v, float):
        return ('{:.' + str(float_prec) + 'f}').format(v)
    return str(v)


def format_str(*args, format_opts={}, **kwargs):
    ss = [format_str_one(arg, **format_opts) for arg in args]
    for k, v in kwargs.items():
        ss.append('{}: {}'.format(k, format_str_one(v, **format_opts)))
    return '\t'.join(ss)
