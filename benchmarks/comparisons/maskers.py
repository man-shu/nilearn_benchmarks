"""Benchmarks for masker objects under nilearn.maskers module."""

# ruff: noqa: RUF012

from ..common import Benchmark
from ..utils import apply_mask, load, apply_mask_parallel


class CompareMask(Benchmark):
    """
    Comparison between the performance of applying a mask to an image using
    nilearn vs. numpy.
    """

    # here we vary both the implementation and the loader
    # so masking can be done using nilearn or numpy (implementation)
    # and the mask and image can be loaded using nilearn or nibabel (loader)
    param_names = ["implementation", "loader"]
    params = (["nilearn", "numpy (ref)"], ["nilearn", "nibabel (ref)"])

    def time_compare_mask(self, implementation, loader):
        """Time the loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)

    def peakmem_compare_mask(self, implementation, loader):
        """Peak memory of loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)


class CompareParallelMask(Benchmark):
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
        Benchmark.setup_cache(self, n_subjects=10, n_masks=4)

    def time_masker(
        self,
        implementation,
        loader,
    ):
        masks, img = load(loader, n_masks=4)
        apply_mask_parallel(masks, img, implementation, n_jobs=4)

    def peakmem_masker(
        self,
        implementation,
        loader,
    ):
        masks, img = load(loader, n_masks=4)
        apply_mask_parallel(masks, img, implementation, n_jobs=4)
