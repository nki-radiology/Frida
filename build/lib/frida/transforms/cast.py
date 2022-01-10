import SimpleITK as sitk
from .base import Cast


class ToNumpyArray(Cast):

    def __init__(self, add_batch_dim=False, add_singleton_dim=False):
        self.add_batch_dim = add_batch_dim
        self.add_singleton_dim = add_singleton_dim
        super(ToNumpyArray, self).__init__()

    def __call__(self, image):
        image = sitk.GetArrayFromImage(image)
        if self.add_batch_dim:
            image = image[None]
        if self.add_singleton_dim:
            image = image[..., None]
        return image
