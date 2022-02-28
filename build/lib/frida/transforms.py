import SimpleITK as sitk
from numpy import floor, ceil

from frida.base import Transform


# !- Base transforms, functioning as wrappers for ITK or Numpy functions


class ITKFilter(Transform):
    """Wraps a SimpleITK filter into a transform operation.
    The Execute method has to be exposed.

    :Example:

    >>> flt = SimpleITK.DiscreteGaussianImageFilter()
    >>> flt.SetVariance(0.65)
    >>> blurring = ITKFilter(flt)
    >>> pipl = Pipeline(ReadVolume(), blurring, ToNumpyArray())

    :param itk_filter: filter to apply
    :type itk_filter: object of type SimpleITK.ImageFilter
    """
    def __init__(self, itk_filter):
        self.flt = itk_filter
        super(ITKFilter, self).__init__()

    def __call__(self, image):
        return self.flt.Execute(image)


class NumpyFunction(Transform):
    """Wraps a function that operates on numpy array into a transform operation.
    The function can have only one input, corresponding to the numpy array.

    :Example:

    >>> def my_numpy_log_filter(image_array):
    >>>     non_neg_arr = np.clip(image_array, 1e-9, image_array.max())
    >>>     return np.log(non_neg_arr)
    >>> log_filter = NumpyFunction(my_numpy_log_filter)
    >>> pipl = Pipeline(ReadVolume(), log_filter, ToNumpyArray())

    :param np_function: function that operates on a numpy array
    :type np_function: python function
    """
    def __init__(self, np_function):
        self.flt = np_function
        super(NumpyFunction, self).__init__()

    def __call__(self, image):
        t = sitk.GetArrayFromImage(image)
        t = self.flt(t)
        t = sitk.GetImageFromArray(t)
        image = t.CopyInformation(image)
        return image


# !- Common transforms


class ZeroOneScaling(Transform):
    """Scales intensities between 0 and 1

    :Example:

    >>> ppl = Pipeline(ReadVolume(), ZeroOneScaling(), ToNumpyArray())
    >>> arr = ppl('path/to/image')
    >>> print('max: ' + str(arr.max()) + ' min: ' + str(arr.min()))
    >>> max: 1.0 min: 0.0

    """
    def __init__(self):
        self.minmax_flt = sitk.MinimumMaximumImageFilter()
        self.cast_flt = sitk.CastImageFilter()
        self.cast_flt.SetOutputPixelType(sitk.sitkFloat32)
        super(ZeroOneScaling, self).__init__()

    def __call__(self, image):
        image = self.cast_flt.Execute(image)
        # get min and max
        self.minmax_flt.Execute(image)
        minimum = self.minmax_flt.GetMinimum()
        maximum = self.minmax_flt.GetMaximum()
        image = (image - minimum) / (maximum - minimum)
        return image


class PadAndCropTo(Transform):
    """Sets the image to a specific size by pad and crop operations.
    Both pad and crop are applied from the center.
    Pad is applied first.

    :Example:

    >>> # without PadAndCropTo
    >>> my_pipeline = Pipeline(ReadVoume(), ToNumpyArray())
    >>> my_pipeline('my_image.nrrd').shape
    >>> (52, 79, 89)
    >>> # with PadAndCropTo
    >>> my_pipeline = Pipeline(ReadVoume(), PadAndCropTo((64, None, 64)), ToNumpyArray())
    >>> my_pipeline('my_image.nrrd').shape
    >>> (64, 79, 64)

    :param target_shape: output size
    :type target_shape: tuple of ints.
        If entry is None, the shape along that dim is left untouched.
    :param cval: filling value for the padding operation, default to 0.
    :type cval: int or float
    """
    def __init__(self, target_shape, cval=0.):
        self.target_shape = target_shape
        self.cval = cval
        super(PadAndCropTo, self).__init__()

    def __call__(self, image):
        # padding
        shape = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        pad = [max(s - t, 0) for t, s in zip(shape, target_shape)]
        lo_bound = [int(floor(p / 2)) for p in pad]
        up_bound = [int(ceil(p / 2)) for p in pad]
        image = sitk.ConstantPad(image, lo_bound, up_bound, self.cval)

        # cropping
        shape = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        crop = [max(t - s, 0) for t, s in zip(shape, target_shape)]
        lo_bound = [int(floor(c / 2)) for c in crop]
        up_bound = [int(ceil(c / 2)) for c in crop]
        image = sitk.Crop(image, lo_bound, up_bound)

        return image


class Resample(Transform):
    """Resamples the image to a specified voxel size.

    :Example:
    >>> # with PadAndCropTo
    >>> my_pipeline = Pipeline(ReadVoume(), Resample(2.), ToNumpyArray())

    :param spacing: output spacing
    :type spacing: int, tuple of ints or Nones.
        If entry is None, the shape along that dim is left untouched.
    :param interpolator: interpolator function
    :type

    """
    def __init__(self, spacing=1., interpolator=sitk.sitkLinear):
        self.spacing = spacing
        self.interpolator = interpolator
        self.flt = sitk.ResampleImageFilter()
        super(Resample, self).__init__()

    def __call__(self, image):
        spacing = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3
        self.flt.SetReferenceImage(image)
        self.flt.SetOutputSpacing(spacing)
        self.flt.SetInterpolator(self.interpolator)
        s0 = int(round((image.GetSize()[0] * image.GetSpacing()[0]) / spacing[0], 0))
        s1 = int(round((image.GetSize()[1] * image.GetSpacing()[1]) / spacing[1], 0))
        s2 = int(round((image.GetSize()[2] * image.GetSpacing()[2]) / spacing[2], 0))
        self.flt.SetSize([s0, s1, s2])
        return self.flt.Execute(image)


class ResampleAndOrient(Transform):
    """Same as the Resample transform, but with the options to correct for different orientations.
    This is frequently used for MRIs, where acquisition planes can be often not along the axial plane.
    """
    def __init__(self, spacing=1., interpolator=sitk.sitkLinear):
        self.spacing = spacing
        self.interpolator = interpolator
        self.flt = sitk.ResampleImageFilter()
        super(ResampleAndOrient, self).__init__()

    def __call__(self, image):
        spacing = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3

        w, h, d = image.GetWidth(), image.GetHeight(), image.GetDepth()

        extreme_points = [
            image.TransformIndexToPhysicalPoint((0, 0, 0)),
            image.TransformIndexToPhysicalPoint((w, 0, 0)),
            image.TransformIndexToPhysicalPoint((w, h, 0)),
            image.TransformIndexToPhysicalPoint((0, h, 0)),
            image.TransformIndexToPhysicalPoint((0, 0, d)),
            image.TransformIndexToPhysicalPoint((w, 0, d)),
            image.TransformIndexToPhysicalPoint((w, h, d)),
            image.TransformIndexToPhysicalPoint((0, h, d))
        ]

        points = [sz / 2 for sz in image.GetSize()]
        physical_points = image.TransformContinuousIndexToPhysicalPoint(points)
        affine = sitk.Euler3DTransform(physical_points)
        inv_affine = affine.GetInverse()

        extreme_points_transformed = [inv_affine.TransformPoint(pnt) for pnt in extreme_points]
        min_x = min(extreme_points_transformed)[0]
        min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
        min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
        max_x = max(extreme_points_transformed)[0]
        max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
        max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

        origin = [min_x, min_y, min_z]

        size = [
            int(ceil((max_x - min_x) / spacing[0])),
            int(ceil((max_y - min_y) / spacing[1])),
            int(ceil((max_z - min_z) / spacing[2]))
        ]

        self.flt.SetReferenceImage(image)
        self.flt.SetSize(size)
        self.flt.SetOutputSpacing(spacing)
        self.flt.SetOutputOrigin(origin)
        self.flt.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.flt.SetInterpolator(self.interpolator)

        return self.flt.Execute(image)


class RunningDatasetStandardization(Transform):
    """Uses the Welford's method to compute running mean and running variance (stored internally).
    Single image standardization is performed via classic (x - mu)/sigma.

    :Example:
    >>> stats = RunningDatasetStandardization()
    >>> ppl = Pipeline(ReadVolume(), stats, ToNumpyArray())
    >>> for ct_path in ct_dataset:
    >>>     ppl(ct_path)
    >>> print(stats.var)
    >>> 378.34 # variance of the CT scans estimated over the dataset

    :param max_iters: number maximum iterations, default to 1000
    :type max_iters: int
    """
    def __init__(self, max_iters=1000):
        self.n = -1.
        self.max_iters = max_iters
        self.avg = self.var = None
        self.init = []
        super(RunningDatasetStandardization, self).__init__()

    def __init_stats(self, image):
        if self.n < 2:
            self.init.append(image)
        if self.n == 2:
            self.avg = (self.init[0] + self.init[1]) / 2.
            self.var = (self.init[0] - self.avg) ** 2 + (self.init[1] - self.avg) ** 2
            self.var = self.var / 2

    def __update_stats(self, image):
        new_avg = self.avg + (1 - self.avg) / self.n
        self.var = self.var + (image - new_avg) * (image - self.avg)
        self.avg = new_avg

    def standardize(self, image):
        std = sitk.Sqrt(sitk.Maximum(self.var, 1e-5))
        return (image - self.avg) / std

    def __call__(self, image):
        self.n += 1
        if self.n < 3:
            self.__init_stats(image)
        if self.n < 2:
            return image
        if self.n < self.max_iters:
            self.__update_stats(image)
        image = self.standardize(image)
        return image


class RunningDatasetStandardizationListener(Transform):
    """Related to the RunningDatasetStandarization transform.
    This is useful for the validation set, where you don't want to collect statistics, but just run the standardization
    according to the ones you have collected so far.

    :Example
    >>> stats = RunningDatasetStandardization(max_iters=1000)
    >>> pipeline_train = Pipeline(ReadVolume(), stats, ToNumpyArray())
    >>> stats_list = RunningDatasetStandardizationListener(stats)
    >>> pipeline_valid = Pipeline(ReadVolume(), stats_list, ToNumpyArray())
    """
    def __init__(self, running_stats):
        self.running_stats = running_stats
        super(RunningDatasetStandardizationListener, self).__init__()

    def __call__(self, image):
        return self.running_stats.standardize(image)
