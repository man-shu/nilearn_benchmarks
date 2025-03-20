# Benchmarks for loading data from disk
# =====================================
from nilearn.image import (
    load_img,
    mean_img,
)
import nibabel as nib
from nilearn.maskers import NiftiMasker
import numpy as np
from .common import Benchmark


class Loading(Benchmark):
    """
    An example benchmark that measures the performance of loading images from
    disk using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_loading(self, loader):
        if loader == "nilearn":
            load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            nib.load("fmri.nii.gz")

    def peakmem_loading(self, loader):
        if loader == "nilearn":
            load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            nib.load("fmri.nii.gz")


class Mean(Benchmark):
    """
    An example benchmark that measures the performance of computing the mean
    of a 4D image using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_mean(self, loader):
        if loader == "nilearn":
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            img = nib.load("fmri.nii.gz")

        mean_img(img, copy_header=True)

    def peakmem_mean(self, loader):
        if loader == "nilearn":
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            img = nib.load("fmri.nii.gz")

        mean_img(img, copy_header=True)


class Slicing(Benchmark):
    """
    An example benchmark that measures the performance of slicing a 4D image
    using nibabel and nilearn.
    """

    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_slicing(self, loader):
        if loader == "nilearn":
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            img = nib.load("fmri.nii.gz")

        img.dataobj[..., 0]

    def peakmem_slicing(self, loader):
        if loader == "nilearn":
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            img = nib.load("fmri.nii.gz")

        img.dataobj[..., 0]


class Masking(Benchmark):
    """
    An example benchmark that measures the performance of applying a mask to
    an image using numpy and nilearn.
    """

    param_names = ["implementation", "loader"]
    params = (["nilearn", "numpy (ref)"], ["nilearn", "nibabel (ref)"])

    def time_masker(self, implementation, loader):
        if loader == "nilearn":
            mask = load_img("mask.nii.gz")
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            mask = nib.load("mask.nii.gz")
            img = nib.load("fmri.nii.gz")

        if implementation == "nilearn":
            NiftiMasker(mask_img=mask).fit_transform(img)
        elif implementation == "numpy (ref)":
            mask = np.asarray(mask.dataobj).astype(bool)
            img = np.asarray(img.dataobj)
            img[mask]

    def peakmem_masker(self, implementation, loader):
        if loader == "nilearn":
            mask = load_img("mask.nii.gz")
            img = load_img("fmri.nii.gz")
        elif loader == "nibabel (ref)":
            mask = nib.load("mask.nii.gz")
            img = nib.load("fmri.nii.gz")

        if implementation == "nilearn":
            NiftiMasker(mask_img=mask).fit_transform(img)
        elif implementation == "numpy (ref)":
            mask = np.asarray(mask.dataobj).astype(bool)
            img = np.asarray(img.dataobj)
            img[mask]
