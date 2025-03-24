# Benchmarks for masker objects under nilearn.maskers module.
# ===========================================================

from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
import nibabel as nib
import numpy as np
from .common import Benchmark
from joblib import Parallel, delayed


def loader(loader, n_masks=1, n_subjects=10):
    loader_to_func = {
        "nilearn": load_img,
        "nibabel (ref)": nib.load,
    }
    loading_func = loader_to_func[loader]
    if n_masks < 1:
        raise ValueError("Number of masks must be at least 1.")
    elif n_masks == 1:
        return loading_func("mask.nii.gz"), loading_func(
            f"fmri_{n_subjects}.nii.gz"
        )
    else:
        return [
            loading_func(f"mask_{idx}.nii.gz") for idx in range(1, n_masks + 1)
        ], loading_func(f"fmri_{n_subjects}.nii.gz")


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
        mask, img = loader(loader)
        apply_mask(mask, img, implementation)

    def peakmem_masker(self, implementation, loader):
        mask, img = loader(loader)
        apply_mask(mask, img, implementation)


class ParallelNiftiMaskingVsReference(Benchmark):
    """
    Comparison between the performance of applying several masks to an
    image by parallelizing nilearn masker objects vs. using numpy
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
        masks, img = loader(loader, n_masks=10)
        Parallel(n_jobs=10)(
            delayed(apply_mask)(mask, img, implementation) for mask in masks
        )

    def peakmem_masker(self, implementation, loader):
        masks, img = loader(loader, n_masks=10)
        Parallel(n_jobs=10)(
            delayed(apply_mask)(mask, img, implementation) for mask in masks
        )


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
        mask, img = loader("nilearn")
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
        mask, img = loader("nilearn")
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
