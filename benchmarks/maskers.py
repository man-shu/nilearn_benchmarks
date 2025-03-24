# Benchmarks for masker objects under nilearn.maskers module.
# ===========================================================

from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
import nibabel as nib
import numpy as np
from .common import Benchmark


class NiftiMaskingVsReference(Benchmark):
    """
    Comparison between the performance of applying a mask to an image using
    numpy vs. nilearn.
    """

    param_names = [
        "implementation",
        "loader",
    ]
    params = (
        ["nilearn", "numpy (ref)"],
        ["nilearn", "nibabel (ref)"],
    )

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


class NiftiMasking(Benchmark):
    """
    Benchmark for applying a mask to an image using nilearn with different
    parameters.
    """

    param_names = ["smoothing_fwhm", "standardize", "detrend"]
    params = (
        [None, 6],
        [False, "zscore_sample", "zscore", "psc"],
        [False, True],
    )

    def time_masker(
        self,
        implementation,
        loader,
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        mask = load_img("mask.nii.gz")
        img = load_img("fmri.nii.gz")
        NiftiMasker(
            mask_img=mask,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
        ).fit_transform(img)

    def peakmem_masker(
        self,
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        mask = load_img("mask.nii.gz")
        img = load_img("fmri.nii.gz")

        NiftiMasker(
            mask_img=mask,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
        ).fit_transform(img)
