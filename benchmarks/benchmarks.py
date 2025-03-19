# Benchmarks for loading data from disk
# =====================================
from nilearn.datasets import fetch_adhd, fetch_atlas_basc_multiscale_2015
from nilearn.image import (
    concat_imgs,
    load_img,
    mean_img,
    resample_to_img,
    new_img_like,
)
import nibabel as nib
from nilearn.maskers import NiftiMasker
import numpy as np


class LoadSuite:
    """
    An example benchmark that measures the performance of loading images from
    disk using nibabel and nilearn.
    """

    def setup_cache(self):
        fmri_data = fetch_adhd(n_subjects=10)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename("fmri.nii.gz")

    def time_nilearn_load_img(self):
        load_img("fmri.nii.gz")

    def time_nib_load(self):
        nib.load("fmri.nii.gz")

    def peakmem_nilearn_load_img(self):
        load_img("fmri.nii.gz")

    def peakmem_nib_load(self):
        nib.load("fmri.nii.gz")


class MeanSuite:
    """
    An example benchmark that measures the performance of computing the mean
    of a 4D image using nibabel and nilearn.
    """

    def setup_cache(self):
        fmri_data = fetch_adhd(n_subjects=10)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename("fmri.nii.gz")

    def time_nilearn_mean_img(self):
        img = load_img("fmri.nii.gz")
        mean_img(img)

    def time_nib_mean(self):
        img = nib.load("fmri.nii.gz")
        mean_img(img)

    def peakmem_nilearn_mean_img(self):
        img = load_img("fmri.nii.gz")
        mean_img(img)

    def peakmem_nib_mean(self):
        img = nib.load("fmri.nii.gz")
        mean_img(img)


class SliceSuite:
    """
    An example benchmark that measures the performance of slicing a 4D image
    using nibabel and nilearn.
    """

    def setup_cache(self):
        fmri_data = fetch_adhd(n_subjects=10)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename("fmri.nii.gz")

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


class NiftiMaskerSuite:
    """
    An example benchmark that measures the performance of applying a mask to
    an image using nilearn.
    """

    def setup_cache(self):
        # get an image
        fmri_data = fetch_adhd(n_subjects=10)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename("fmri.nii.gz")

        # get a mask
        atlas_path = fetch_atlas_basc_multiscale_2015(resolution=7).maps
        resampled_atlas = resample_to_img(
            atlas_path,
            concat,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )
        mask = resampled_atlas.get_fdata() == 1
        mask_img = new_img_like(
            resampled_atlas,
            mask,
            affine=resampled_atlas.affine,
            copy_header=True,
        )
        mask_img.to_filename("mask.nii.gz")

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


class NumpyMaskerSuite:
    """
    An example benchmark that measures the performance of applying a mask to
    an image using numpy.
    """

    def setup_cache(self):
        # get an image
        fmri_data = fetch_adhd(n_subjects=10)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename("fmri.nii.gz")

        # get a mask
        atlas_path = fetch_atlas_basc_multiscale_2015(resolution=7).maps
        resampled_atlas = resample_to_img(
            atlas_path,
            concat,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )
        mask = resampled_atlas.get_fdata() == 1
        mask_img = new_img_like(
            resampled_atlas,
            mask,
            affine=resampled_atlas.affine,
            copy_header=True,
        )
        mask_img.to_filename("mask.nii.gz")

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
