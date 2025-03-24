# Benchmarks for image operations under nilearn.image module.
# ===========================================================
from nilearn.image import mean_img
from .common import Benchmark, load


class Loading(Benchmark):
    """
    An example benchmark that measures the performance of loading images from
    disk using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_loading(self, loader):
        load(loader)

    def peakmem_loading(self, loader):
        load(loader)


class Mean(Benchmark):
    """
    An example benchmark that measures the performance of computing the mean
    of a 4D image using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_mean(self, loader):
        img = load(loader)[1]
        mean_img(img, copy_header=True)

    def peakmem_mean(self, loader):
        img = load(loader)[1]
        mean_img(img, copy_header=True)


class Slicing(Benchmark):
    """
    An example benchmark that measures the performance of slicing a 4D image
    using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_slicing(self, loader):
        img = load(loader)[1]
        img.dataobj[..., 0]

    def peakmem_slicing(self, loader):
        img = load(loader)[1]
        img.dataobj[..., 0]
