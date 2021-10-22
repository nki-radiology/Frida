import SimpleITK as sitk
from numpy import floor, ceil
from numpy.random import uniform

from .base import Transform

# !- Base transforms, functioning as wrappers for ITK or Numpy functions

class ITKFilter(Transform):

    def __init__( self, itk_filter ):
        self.flt = itk_filter
        super(ITKFilter, self).__init__()

    def __call__( self, image ):
        return self.flt.Execute(image)


class NumpyFunction(Transform):

    def __init__( self, np_function ):
        self.flt = np_function
        super(NumpyFunction, self).__init__()

    def __call__( self, image ):
        t = sitk.GetArrayFromImage(image)
        t = self.flt(t)
        t = sitk.GetImageFromArray(t)
        image = t.CopyInformation(image)
        return image

# !- Common transforms 

class ZeroOneScaling(Transform):

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
        image = (image - minimum)/(maximum - minimum)
        return image


class PadAndCropTo(Transform):

    def __init__( self, target_shape, cval=0. ):
        self.target_shape = target_shape
        self.cval = cval
        super(PadAndCropTo, self).__init__()

    def __call__( self, image ):

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

    def __init__( self, spacing=1., orient=True, interpolator=sitk.sitkLinear ):
        self.spacing = spacing
        self.interpolator = interpolator
        self.orient = orient
        self.flt = sitk.ResampleImageFilter()
        super(Resample, self).__init__()

    def __call__( self, image ):
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

    def __init__( self, spacing=1., interpolator=sitk.sitkLinear ):
        self.spacing = spacing
        self.interpolator = interpolator
        self.flt = sitk.ResampleImageFilter()
        super(ResampleAndOrient, self).__init__()

    def __call__( self, image ):
        
        spacing = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3
            
        w, h, d = image.GetWidth(), image.GetHeight(), image.GetDepth()

        extreme_points = [
              image.TransformIndexToPhysicalPoint((0,0,0)), 
              image.TransformIndexToPhysicalPoint((w,0,0)),
              image.TransformIndexToPhysicalPoint((w,h,0)),
              image.TransformIndexToPhysicalPoint((0,h,0)),
              image.TransformIndexToPhysicalPoint((0,0,d)), 
              image.TransformIndexToPhysicalPoint((w,0,d)),
              image.TransformIndexToPhysicalPoint((w,h,d)),
              image.TransformIndexToPhysicalPoint((0,h,d))
        ]

        points = [sz/2 for sz in image.GetSize()]
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
            int(ceil((max_x-min_x)/spacing[0])), 
            int(ceil((max_y-min_y)/spacing[1])), 
            int(ceil((max_z-min_z)/spacing[2]))
        ]

        self.flt.SetReferenceImage(image)
        self.flt.SetSize(size)
        self.flt.SetOutputSpacing(spacing)
        self.flt.SetOutputOrigin(origin)
        self.flt.SetOutputDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.flt.SetInterpolator(self.interpolator)
        
        return self.flt.Execute(image)


class ToNumpyArray(Transform):

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


class RunningDatasetStandardization(Transform):
    """" RunningDatasetStandardization
    Uses the Welford's method to compute running mean and running variance (stored internally).
    Single image standardization is performed via classic (x - mu)/sigma.

    Params
    ------
        * max_iters: maximum iterations up until which the internal stats have to been updated
    """

    def __init__( self, max_iters=1000 ):
        self.n = -1.
        self.max_iters = max_iters
        self.avg = self.var = None
        self.init = []
        super(RunningDatasetStandardization, self).__init__()

    def __init_stats( self, image ):
        if self.n < 2:
            self.init.append(image)
        if self.n == 2:
            self.avg = (self.init[0] + self.init[1]) / 2.
            self.var = (self.init[0] - self.avg) ** 2 + (self.init[1] - self.avg) ** 2
            self.var = self.var / 2

    def __update_stats( self, image ):
        new_avg = self.avg + (1 - self.avg) / self.n
        self.var = self.var + (image - new_avg) * (image - self.avg)
        self.avg = new_avg

    def standardize( self, image ):
        std = sitk.Sqrt(sitk.Maximum(self.var, 1e-5))
        return (image - self.avg) / std

    def __call__( self, image ):
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

    def __init__( self, running_stats ):
        self.running_stats = running_stats
        super(RunningDatasetStandardizationListener, self).__init__()

    def __call__( self, image ):
        return self.running_stats.standardize(image)
