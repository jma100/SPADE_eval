''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

Derived from:
https://github.com/bioinf-jku/TTUR/blob/8eca1abd808aba8d8cf90208887453744bb53190/fid.py

See --help to see further details.
'''

import numpy as np
import os
import sys
import gzip
import pickle

from scipy import linalg
import tempfile
import re
import time
import contextlib
import pathlib
import logging
import urllib
import warnings
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # in case you want to run this in a subdir
import utils
from utils import state


# ------------------------------------------------------------------------------

# max center crop
# from biggan
# https://github.com/ajbrock/BigGAN-PyTorch/blob/65ade92981e9f44e3b7aea895e20886219a85a25/utils.py#L434
class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return torchvision.transforms.functional.center_crop(img, min(img.size))


class UnsupervisedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, max_size=None):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.symlink(root, os.path.join(self.temp_dir.name, 'dummy'))
        root = self.temp_dir.name
        super().__init__(root, transform=transform)
        self.perm = None
        if max_size is not None:
            actual_size = super().__len__()
            if actual_size > max_size:
                self.perm = torch.randperm(actual_size)[:max_size].clone()
                logging.info(f"{root} has {actual_size} images, downsample to {max_size}")
            else:
                logging.info(f"{root} has {actual_size} images <= max_size={max_size}")

    def _find_classes(self, dir):
        return ['./dummy'], {'./dummy': 0}

    def __getitem__(self, key):
        if self.perm is not None:
            key = self.perm[key].item()
        return super().__getitem__(key)[0]

    def __len__(self):
        if self.perm is not None:
            return self.perm.size(0)
        else:
            return super().__len__()

def get_image_loader(path, resize_size=299, max_size=None, **dataloader_kwargs):
    dataset = UnsupervisedImageFolder(
        path,
        torchvision.transforms.Compose([
            CenterCropLongEdge(),
            # inception takes in 299 x 299 tensor:
            # https://github.com/tensorflow/models/blob/d11aa330decc34e9433351eb75ba59a325660e0f/research/inception/inception/image_processing.py#L49
            # can verify by looking at 'ResizeBilinear/size' node
            torchvision.transforms.Resize(resize_size),
            torchvision.transforms.Lambda(lambda x: torch.as_tensor(np.asarray(x.convert('RGB')))),
        ]),
        max_size=max_size)
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# ------------------------------------------------------------------------------


class GeneratorLoader(object):
    class DummyDataset(object):
        def __init__(self, len):
            self.len = len

        def __len__(self):
            return self.len

    def __init__(self, generator, batch_size, num_images, device):
        torch.backends.cudnn.benchmark = True
        self.generator = generator
        self.batch_size = batch_size
        self.num_images = num_images
        self.device = device
        self.dataset = GeneratorLoader.DummyDataset(self.num_images)

    def __iter__(self):
        yielded = 0
        while yielded < self.num_images:
            batch_size = min(self.batch_size, self.num_images - yielded)
            with torch.no_grad():
                z = torch.randn(batch_size, self.generator.z_dim, device=self.device)
                outputs = self.generator(z).detach()
            yield utils.output_to_image(outputs, numpy=False)
            del z, outputs
            yielded += batch_size

    def __len__(self):
        return int(np.ceil(self.num_images / self.batch_size))


def get_generator_loader(checkpoint, batch_size, num_images, device):
    # FIXME: use your code to load generator
    # generator = ...
    raise NotImplementedError
    return GeneratorLoader(generator, batch_size, num_images, device)


# ------------------------------------------------------------------------------

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


# ------------------------------------------------------------------------------

def get_activations(images, sess, batch_size=50):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        logging.warning("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        logging.info("Propagating batch %d/%d" % (i + 1, n_batches))
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        # Put the image in a node after decoding and cropping, but before
        # normalization, i.e.,
        # https://github.com/tensorflow/models/blob/d11aa330decc34e9433351eb75ba59a325660e0f/research/inception/inception/image_processing.py#L297
        #
        # Technically this is after convert to floating point, i.e., normalizing
        # to [0, 1]. But the normalization part is apparently fused with the
        # later scaling to [-1, 1], and thus this takes in [0, 255] batched
        # image tensors. Finally, this accepts both uint8 and float32.
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    logging.info("done")
    return pred_arr


# ------------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
# ------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# -----------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
# -----------------


def get_activations_from_loader(loader, sess):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- loader      : image loader
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """

    logging.info('Dataset size: {}'.format(len(loader.dataset)))
    logging.info('Batch size: {}'.format(loader.batch_size))

    inception_layer = _get_inception_layer(sess)
    pred_arr = np.empty((len(loader.dataset), 2048))
    start = end = 0
    for i, batch in enumerate(loader):
        if i % 10 == 0:
            logging.info("Propagating batch %d/%d" % (i + 1, len(loader)))
        start = end
        end = start + batch.size(0)
        # Put the image in a node after decoding and cropping, but before
        # normalization, i.e.,
        # https://github.com/tensorflow/models/blob/d11aa330decc34e9433351eb75ba59a325660e0f/research/inception/inception/image_processing.py#L297
        #
        # Technically this is after convert to floating point, i.e., normalizing
        # to [0, 1]. But the normalization part is apparently fused with the
        # later scaling to [-1, 1], and thus this takes in [0, 255] batched
        # image tensors. Finally, this accepts both uint8 and float32.
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch.numpy()})
        pred_arr[start:end] = pred.reshape(batch.size(0), -1)
    logging.info("done")
    return pred_arr


def calculate_activation_statistics_from_loader(loader, sess):
    """Calculation of the statistics used by the FID.
    Params:
    -- loader      : image loader
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_loader(loader, sess)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# ------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# ------------------------------------------------------------------------------

def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        logging.info("Downloading Inception model...")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
        logging.info('done!')
    return str(model_file)


def calculate_fid_given_paths(datasets, inception_path, config):
    ''' Calculates the FID of many datasets. '''
    inception_path = check_or_download_inception(inception_path)

    create_inception_graph(str(inception_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        stats = [d(sess) for d in datasets]
        if len(stats) != 2:
            raise RuntimeError(f"Expected two datasets, but got {len(datasets)}")
        fid_value = calculate_frechet_distance(*stats[0], *stats[1])
        return fid_value


# ------------------------------------------------------------------------------
# Set configs

class DatasetType(object, metaclass=utils.types.options.BuildCompositeOptionEnum):
    @utils.types.options.BuildCompositeOptionEnum.add_options(
        state.Option('path', utils.types.make_optional(str),
                     default=None,
                     desc='Generator checkpoint path. None means original weights. '),
        state.Option('state_dict_key', utils.types.make_optional(str),
                     default=None,
                     desc=('Key to index the checkpoint for state dict. '
                           'None means using the entire checkpoint as statedict. ')),
        state.Option('resolution', int, desc='Output resolution'),
        state.Option('batch_size', int, default=50, desc='Batch size'),
        state.Option('num_images', int, default=50000, desc='Total number of images'),
        state.Option('device', torch.device, default='cuda',
                     desc='Which device to use for this generator'),
        state.Option('stats_save_path', utils.types.make_optional(str),
                     desc='Path to save stats'),
    )
    def generator(sess, path, batch_size, num_images, device, stats_save_path):
        if path is not None:
            checkpoint = torch.load(path, map_location='cpu')
            if state_dict_key is not None:
                checkpoint = checkpoint[state_dict_key]
        else:
            checkpoint = None
        loader = get_generator_loader(checkpoint, batch_size, num_images, device)  # implement this
        del checkpoint
        torch.cuda.empty_cache()
        logging.info('Built generator from {}{}'.format(
            path, '' if state_dict_key is None else '["{}"]'.format(state_dict_key)))
        m, s = calculate_activation_statistics_from_loader(loader, sess)
        if stats_save_path:
            utils.mkdir(os.path.dirname(os.path.abspath(stats_save_path)))
            np.savez(stats_save_path, mu=m, sigma=s)
            logging.info('Saved inception stats to {}'.format(stats_save_path))
        return m, s

    @utils.types.options.BuildCompositeOptionEnum.add_options(
        state.Option('path', str, desc='Images path'),
        state.Option('resize_size', int, default=299,
                     desc=('Resize the images. 299 (default) is the Inception '
                           'v3 input size, but you should match the generator '
                           'output size here.')),
        state.Option('batch_size', int, default=50, desc='Batch size'),
        state.Option('max_size', utils.types.make_optional(int), default=None),
        state.Option('num_workers', int, default=12, desc='Data loader workers'),
        state.Option('stats_save_path', utils.types.make_optional(str),
                     desc='Path to save stats'),
    )
    def images(sess, path, resize_size, batch_size, max_size, num_workers, stats_save_path):
        loader = get_image_loader(path, resize_size, batch_size=batch_size,
                                  max_size=max_size, num_workers=num_workers)
        logging.info(f'Built image loader (dataset len={len(loader.dataset)}) for {path}')
        m, s = calculate_activation_statistics_from_loader(loader, sess)
        if stats_save_path:
            utils.mkdir(os.path.dirname(os.path.abspath(stats_save_path)))
            np.savez(stats_save_path, mu=m, sigma=s)
            logging.info('Saved inception stats to {}'.format(stats_save_path))
        return m, s

    @utils.types.options.BuildCompositeOptionEnum.add_options(
        state.Option('path', str, desc='Stats path'),
    )
    def stats(sess, path):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        if path.endswith('.npz'):
            f.close()
        logging.info('Loaded inception stats from {}'.format(path))
        return m, s

state.add_option('datasets', utils.types.make_typelist(DatasetType),
                 desc='Datasets')

state.add_option('device', torch.device, default='cuda', desc='Which device to use')

state.add_option('inception_path', str, default='/tmp',
                 desc='Path to Inception model (will be downloaded if not existent)')


def set_start_time():
    state.start_time = time.strftime(r"%Y-%m-%d %H:%M:%S")

state.register_parse_hook(set_start_time)


state.add_option('output_base_dir', type=str,
                 default='./results/', desc='Base directory to store outputs')

state.add_option('output_folder', type=utils.types.make_optional(str),
                 default=None, desc='Folder to store outputs')


def set_output_folder():
    if state.output_folder is None:
        with state.overwrite():
            config_base = os.path.splitext(os.path.basename(state.config))[0]
            time_suffix = re.sub('[^0-9a-zA-Z]+', '_', state.start_time)
            state.output_folder = '{}_{}'.format(config_base, time_suffix)
    state.output_dir = os.path.join(state.output_base_dir, state.output_folder)
    utils.mkdir(state.output_dir)

state.register_parse_hook(set_output_folder)


state.add_option('logging_file', type=utils.types.make_optional(str),
                 default='output.log', desc='Filename to log outputs')


def config_logging():
    logging_file = None
    if state.logging_file is not None:
        logging_file = os.path.join(state.output_dir, state.logging_file)
    utils.logging.configure(logging_file)
    state.logging_configured = True

state.register_parse_hook(config_logging)

state.set_desc('FID computation')

if __name__ == "__main__":
    state.parse_options()
    logging.info('')
    logging.info(state)
    logging.info('')

    if state.device.type == 'cpu':
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        # FIXME: the following code doesn't actually prevent context being
        #        created by tf on other devices.
        assert state.device.type == 'cuda'
        device_index = state.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        config = tf.ConfigProto(device_count={'GPU': 1})
        config.gpu_options.visible_device_list = str(device_index)

    fid_value = calculate_fid_given_paths(state.datasets, state.inception_path, config)
    logging.info("FID: {}".format(fid_value))
