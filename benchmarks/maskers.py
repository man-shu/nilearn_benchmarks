# Benchmarks for masker objects under nilearn.maskers module.
# ===========================================================

from nilearn.maskers import NiftiMasker
import numpy as np
from .common import Benchmark, load


def apply_mask(mask, img, implementation, nifti_masker_params=None):
    if implementation == "nilearn":
        if nifti_masker_params is None:
            NiftiMasker(mask_img=mask).fit_transform(img)
        else:
            masker = NiftiMasker(mask_img=mask)
            masker.set_params(**nifti_masker_params)
            masker.fit_transform(img)
    elif implementation == "numpy":
        mask = np.asarray(mask.dataobj).astype(bool)
        img = np.asarray(img.dataobj)
        img[mask]


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
        mask, img = load(loader)
        apply_mask(mask, img, implementation)

    def peakmem_masker(self, implementation, loader):
        mask, img = load(loader)
        apply_mask(mask, img, implementation)


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
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        mask, img = load("nilearn")
        apply_mask(
            mask,
            img,
            "nilearn",
            nifti_masker_params={
                "smoothing_fwhm": smoothing_fwhm,
                "standardize": standardize,
                "detrend": detrend,
            },
        )

    def peakmem_masker(
        self,
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        mask, img = load("nilearn")
        apply_mask(
            mask,
            img,
            "nilearn",
            nifti_masker_params={
                "smoothing_fwhm": smoothing_fwhm,
                "standardize": standardize,
                "detrend": detrend,
            },
        )
