import os
import shutil
import base64 as b64
import logging
import contextlib
import collections
from io import BytesIO

import PIL
import numpy as np
import scipy.ndimage
import torch
import torchvision
import visdom
import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br, pre

from . import file as file_utils

__all__ = ['DummyVisualizer', 'Visualizer']


class DummyVisualizer(object):
    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return self


class HTML(object):
    r"""This HTML class allows us to save images and write texts into a single HTML file.
    It consists of functions such as <add_header> (add a text header to the HTML file),
    <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
    It is based on Python library 'dominate', a Python library for creating and manipulating
    HTML documents using a DOM API.
    """

    class Data(object):
        def html(self):
            return NotImplemented

    class Image(Data):
        def __init__(self, data, name, desc=None, display_height=256):
            self.data = data
            self.name = name
            self.desc = desc
            self.display_height = display_height
            self.relative_save_path = None

        def save(self, root_dir):
            if self.relative_save_path is not None:
                return
            relative_save_path = os.path.join('images', self.name + '.png')
            full_save_path = os.path.join(root_dir, relative_save_path)
            file_utils.mkdir(os.path.dirname(full_save_path))

            if os.path.exists(full_save_path):
                raise RuntimeError('{} already exists'.format(full_save_path))

            try:
                if isinstance(self.data, BytesIO):  # assume png
                    self.data.seek(0)
                    with open(full_save_path, 'wb') as f:
                        shutil.copyfileobj(self.data, f, length=131702)  # 128k
                    return
                elif isinstance(self.data, (torch.Tensor, np.ndarray)):
                    self.data = torchvision.transforms.functional.to_pil_image(self.data)
                if isinstance(self.data, PIL.Image.Image):
                    self.data.save(full_save_path, format='PNG')
                    return
                else:
                    return NotImplemented
            finally:
                self.relative_save_path = relative_save_path

        def html(self, root_dir):
            self.save(root_dir)
            with p():
                with a(href=self.relative_save_path):
                    img_style = "image-rendering:pixelated;"
                    if self.display_height is not None:
                        img_style += 'height:{}px;'.format(self.display_height)
                    img(style=img_style, src=self.relative_save_path)
                if self.desc is not None:
                    br()
                    pre(self.desc)

    class Text(Data):
        def __init__(self, data):
            self.data = data

        def html(self, root_dir):
            p(self.data)

    def __init__(self, web_dir, web_subdir, title, refresh=0):
        self.title = title
        if web_subdir is not None:
            self.web_dir = os.path.join(web_dir, web_subdir)
        else:
            self.web_dir = web_dir

        if os.path.exists(self.web_dir):
            logging.warning("Web directory already exists, removing: {}".format(self.web_dir))
            shutil.rmtree(self.web_dir)

        file_utils.mkdir(self.web_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_table(self, data_coll, header=None):
        """add images to the HTML file
        Parameters:
            data_coll: a 1/2D list of HTML.Data.
        """
        if not isinstance(data_coll[0], collections.Sequence):
            data_coll_2d = [data_coll]
        else:
            data_coll_2d = data_coll

        if header is not None:
            self.add_header(header)

        t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(t)
        with t:
            for data_coll_1d in data_coll_2d:
                with tr():
                    for data in data_coll_1d:
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            data.html(self.web_dir)
        self.save()

    def save(self):
        """save the current content to the HTML file"""
        html_file = os.path.join(self.web_dir, 'index.html')
        with open(html_file, 'wt') as f:
            f.write(self.doc.render())

    def __del__(self):
        if hasattr(self, 'web_dir') and hasattr(self, 'doc'):
            self.save()


class Visualizer(visdom.Visdom):
    def __init__(self, *args, env='main', html_opts=None, **kwargs):
        env = env.replace('_', '-').replace('/', '-')  # envrionments are hierarchically organized by first _ or /
        super().__init__(*args, env=env, **kwargs)
        self.html_opts = html_opts
        self.htmls = {}
        self.subenv_name = None

    @contextlib.contextmanager
    def subenv(self, subenv):
        assert self.subenv_name is None, "nested subenv not supported"
        old_env = self.env
        self.subenv_name = subenv
        self.env = '{}_{}'.format(self.env, subenv)
        yield
        if self.subenv_name in self.htmls:
            self.htmls[self.subenv_name].save()  # save when leaving subenv
        self.subenv_name = None
        self.env = old_env

    @property
    def html(self):
        if self.subenv_name not in self.htmls:
            if self.html_opts is not None:
                self.htmls[self.subenv_name] = HTML(web_subdir=self.subenv_name, **self.html_opts)
            else:
                self.htmls[self.subenv_name] = DummyVisualizer()
        return self.htmls[self.subenv_name]

    def state(self, state, **kwargs):
        yaml_repr = repr(state)  # TODO: highlight
        return self.text('<pre><code class="yaml">{}</code></pre>'.format(yaml_repr), **kwargs)

    def images_hwc_grid(self, tensors, ncols=8, padding=2,
                        special_pad=set(),     # tensor indices to apply special padding (rather than black)
                        special_pad_val=None,  # special padding val (Default: None => white)
                        interp_shape=None, interp_order=2,
                        **kwargs):
        N = len(tensors)
        assert N > 0

        if interp_shape is not None:
            orig_im_shape = tensors[0].shape
            if not isinstance(interp_shape, collections.abc.Sequence):
                interp_shape = (interp_shape, interp_shape)
            else:
                assert len(interp_shape) == 2

            if tensors.shape[1:3] != interp_shape:
                interp_ratio = (1, interp_shape[0] / float(orig_im_shape[0]),
                                interp_shape[1] / float(orig_im_shape[1]), 1)

                tensors = scipy.ndimage.zoom(np.asarray(tensors), interp_ratio, order=interp_order)

        tensors = [torch.as_tensor(t) for t in tensors]
        im_shape = tensors[0].shape
        channel_shape = tuple(im_shape[2:])
        padded_shape = (im_shape[0] + padding * 2, im_shape[1] + padding * 2) + channel_shape
        dtype = tensors[0].dtype

        special_pad = set(special_pad)

        if len(special_pad) > 0:
            if special_pad_val is None:
                special_pad_val = torch.full(size=(), fill_value=255 if dtype == torch.uint8 else 1, dtype=dtype)
            else:
                special_pad_val = torch.as_tensor(special_pad_val)
            special = special_pad_val.expand(padded_shape)
        black = torch.zeros(size=(), dtype=dtype).expand(padded_shape)

        nrows = int(np.ceil(N / ncols))
        rows = []
        for i in range(nrows):
            row = []
            for j in range(ncols):
                idx = i * ncols + j
                row.append(special if idx in special_pad else black)
            rows.append(torch.cat(row, dim=1))
        full = torch.cat(rows, dim=0)

        image_grid_view = full.view((nrows, padded_shape[0], ncols, padded_shape[1]) + channel_shape)
        image_grid_view = image_grid_view[:, padding:-padding, :, padding:-padding].transpose(1, 2)

        for n, t in enumerate(tensors):
            i = n // ncols
            j = n % ncols
            image_grid_view[i, j].copy_(torch.as_tensor(t))

        return self.image_hwc(full, **kwargs)

    def image_hwc(self, img, win=None, env=None, opts=None,
                  interp_shape=None, interp_order=2, get_buffer=False):
        r"""
        This function draws an img. It takes as input an `HxWxC` or `HxW` tensor
        `img` that contains the image. The array values can be float in [0,1] or
        uint8 in [0, 255].
        """
        opts = {} if opts is None else opts
        visdom._title2str(opts)
        visdom._assert_opts(opts)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if isinstance(img, PIL.Image.Image):
            im = img
        else:
            im = torchvision.transforms.functional.to_pil_image(img)

        if interp_shape is not None:
            if not isinstance(interp_shape, collections.abc.Sequence):
                interp_shape = (interp_shape, interp_shape)
            im = im.resize(interp_shape, interp_order)

        # PIL uses (width, height)
        opts['width'] = opts.get('width', im.size[0])
        opts['height'] = opts.get('height', im.size[1])

        buf = BytesIO()
        im.save(buf, format='PNG')
        b64encoded = b64.b64encode(buf.getvalue()).decode('utf-8')

        data = [{
            'content': {
                'src': 'data:image/png;base64,' + b64encoded,
                'caption': opts.get('caption'),
            },
            'type': 'image',
        }]

        response = self._send({
            'data': data,
            'win': win,
            'eid': env,
            'opts': opts,
        })

        if get_buffer:
            return response, buf
        else:
            return response
