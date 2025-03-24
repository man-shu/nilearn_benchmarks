from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
import nibabel as nib
import numpy as np
from .common import Benchmark
from joblib import Parallel, delayed


def load(loader, n_masks=1, n_subjects=10):
    loader_to_func = {
        "nilearn": load_img,
        "nibabel (ref)": nib.load,
    }
    loading_func = loader_to_func[loader]
    if n_masks < 1:
        raise ValueError("Number of masks must be at least 1.")
    elif n_masks == 1:
        return loading_func("mask_1.nii.gz"), loading_func(
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
            print(masker.get_params())
    elif implementation == "numpy":
        mask = np.asarray(mask.dataobj).astype(bool)
        img = np.asarray(img.dataobj)
        img[mask]


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

    def setup_cache(self):
        Benchmark.setup_cache(self, n_subjects=10, n_masks=10)

    def time_masker(self, implementation, loader):
        masks, img = load(loader, n_masks=10)
        Parallel(n_jobs=10)(
            delayed(apply_mask)(mask, img, implementation) for mask in masks
        )

    def peakmem_masker(self, implementation, loader):
        masks, img = load(loader, n_masks=10)
        Parallel(n_jobs=10)(
            delayed(apply_mask)(mask, img, implementation) for mask in masks
        )
