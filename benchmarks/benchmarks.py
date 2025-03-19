# Benchmarks for loading data from disk
# =====================================
from nilearn.datasets import fetch_adhd
from nilearn.image import concat_imgs, load_img, mean_img
import nibabel as nib


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
