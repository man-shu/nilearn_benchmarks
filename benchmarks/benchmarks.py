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

    def time_nilearn_load_img(self):
        load_img("fmri.nii.gz")

    def time_nib_load(self):
        nib.load("fmri.nii.gz")

    def peakmem_nilearn_load_img(self):
        load_img("fmri.nii.gz")

    def peakmem_nib_load(self):
        nib.load("fmri.nii.gz")


class Mean(Benchmark):
    """
    An example benchmark that measures the performance of computing the mean
    of a 4D image using nibabel and nilearn.
    """

    def time_nilearn_mean_img(self):
        img = load_img("fmri.nii.gz")
        mean_img(img, copy_header=True)

    def time_nib_mean(self):
        img = nib.load("fmri.nii.gz")
        mean_img(img, copy_header=True)

    def peakmem_nilearn_mean_img(self):
        img = load_img("fmri.nii.gz")
        mean_img(img, copy_header=True)

    def peakmem_nib_mean(self):
        img = nib.load("fmri.nii.gz")
        mean_img(img, copy_header=True)


class Slicing(Benchmark):
    """
    An example benchmark that measures the performance of slicing a 4D image
    using nibabel and nilearn.
    """

    def time_nilearn_slice_img(self):
        img = load_img("fmri.nii.gz")
        img.dataobj[..., 0]

    def time_nib_slice(self):
        img = nib.load("fmri.nii.gz")
        img.dataobj[..., 0]

    def peakmem_nilearn_slice_img(self):
        img = load_img("fmri.nii.gz")
        img.dataobj[..., 0]

    def peakmem_nib_slice(self):
        img = nib.load("fmri.nii.gz")
        img.dataobj[..., 0]


class NiftiMasking(Benchmark):
    """
    An example benchmark that measures the performance of applying a mask to
    an image using nilearn.
    """

    def time_path_nifti_masker(self):
        NiftiMasker(mask_img="mask.nii.gz").fit_transform("fmri.nii.gz")

    def time_nilearn_nifti_masker(self):
        mask_img = load_img("mask.nii.gz")
        img = load_img("fmri.nii.gz")
        NiftiMasker(mask_img=mask_img).fit_transform(img)

    def time_nib_nifti_masker(self):
        mask_img = nib.load("mask.nii.gz")
        img = nib.load("fmri.nii.gz")
        NiftiMasker(mask_img=mask_img).fit_transform(img)

    def peakmem_path_nifti_masker(self):
        NiftiMasker(mask_img="mask.nii.gz").fit_transform("fmri.nii.gz")

    def peakmem_nilearn_nifti_masker(self):
        mask_img = load_img("mask.nii.gz")
        img = load_img("fmri.nii.gz")
        NiftiMasker(mask_img=mask_img).fit_transform(img)

    def peakmem_nib_nifti_masker(self):
        mask_img = nib.load("mask.nii.gz")
        img = nib.load("fmri.nii.gz")
        NiftiMasker(mask_img=mask_img).fit_transform(img)


class NumpyMasking(Benchmark):
    """
    An example benchmark that measures the performance of applying a mask to
    an image using numpy.
    """

    def time_nilearn_numpy_masker(self):
        mask = np.asarray(load_img("mask.nii.gz").dataobj).astype(bool)
        img = np.asarray(load_img("fmri.nii.gz").dataobj)
        img[mask]

    def time_nib_numpy_masker(self):
        mask = np.asarray(nib.load("mask.nii.gz").dataobj).astype(bool)
        img = np.asarray(nib.load("fmri.nii.gz").dataobj)
        img[mask]

    def peakmem_nilearn_numpy_masker(self):
        mask = np.asarray(load_img("mask.nii.gz").dataobj).astype(bool)
        img = np.asarray(load_img("fmri.nii.gz").dataobj)
        img[mask]

    def peakmem_nib_numpy_masker(self):
        mask = np.asarray(nib.load("mask.nii.gz").dataobj).astype(bool)
        img = np.asarray(nib.load("fmri.nii.gz").dataobj)
        img[mask]
