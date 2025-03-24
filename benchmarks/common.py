"""
Common utilities for the benchmarks.
"""

from nilearn.datasets import fetch_adhd, fetch_atlas_basc_multiscale_2015
from nilearn.image import concat_imgs, new_img_like, resample_to_img


class Benchmark:

    def setup_cache(self, n_subjects=10, n_masks=1):
        # get an image
        fmri_data = fetch_adhd(n_subjects=n_subjects)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename(f"fmri_{n_subjects}.nii.gz")

        # get a mask
        atlas_path = fetch_atlas_basc_multiscale_2015(resolution=64).maps
        resampled_atlas = resample_to_img(
            atlas_path,
            concat,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )
        for idx in range(1, n_masks + 1):
            mask = resampled_atlas.get_fdata() == idx
            mask_img = new_img_like(
                resampled_atlas,
                mask,
                affine=resampled_atlas.affine,
                copy_header=True,
            )
            mask_img.to_filename(f"mask_{idx}.nii.gz")
