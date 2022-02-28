import SimpleITK as sitk
from .base import Transform

class Cast(Transform):

    def __init__(self):
        self.random_seed = 0
        super(Cast, self).__init__()

    def __call__(self, image, *args):
        pass

class ToNumpyArray(Cast):
    """Cast the SimpleITK.Image to a numpy ndarray.

    :Example:
    >>> ppl = Pipeline(ReadVolume(), Resample(2))
    >>> type(ppl('path/to/image'))
    >>> SimpleITK.Image
    >>> ppl = Pipeline(ReadVolume(), Resample(2) ToNumpyArray())
    >>> type(ppl('path/to/image'))
    >>> numpy.ndarray
    """
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
