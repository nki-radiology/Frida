from .base import Augmentation

import SimpleITK as sitk
import numpy as np

from numpy.random import uniform
from scipy.ndimage.filters import gaussian_filter


class ApplyTransformAtRandom(Augmentation):

    def __init__( self, probability, transform ):
        self.probability = probability
        self.transform = transform
        super(ApplyTransformAtRandom, self).__init__()

    def __call__( self, image ):
        if self.probability < uniform(0., 1., 1):
            image = self.transform(image)
        return image


class RandomLinearDisplacement(Augmentation):

    def __init__( self,
                  rotation_range=None,
                  shear_range=None,
                  zoom_range=None,
                  shift_range=None,
                  random_axis_flip=False,
                  interpolator=sitk.sitkLinear,
                  cval=0. ):

        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.random_axis_flip = random_axis_flip
        self.interpolator = interpolator
        self.cval = cval
        super(RandomLinearDisplacement, self).__init__()

    def __call__( self, image, *args ):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        transform_matrix = args[0]

        if len(args) > 1:
            for t in args:
                transform_matrix = transform_matrix.dot(t)

        flt = sitk.AffineTransform(3)
        flt.SetTranslation(tuple(transform_matrix[0:3, -1].squeeze()))
        flt.SetMatrix(transform_matrix[:3, :3].ravel())
        image = sitk.Resample(image, image, flt, self.interpolator, self.cval)
        return image

    def get_random_params( self, image ):

        np.random.seed(self.random_seed)

        transform_matrix = np.eye(4)
        if self.rotation_range is not None:
            R = self._random_rotation()
            transform_matrix = np.dot(transform_matrix, R)
        if self.shear_range is not None:
            S = self._random_shear()
            transform_matrix = np.dot(transform_matrix, S)
        if self.shift_range is not None:
            S = self._random_shift()
            transform_matrix = np.dot(transform_matrix, S)
        if self.zoom_range is not None:
            Z = self._random_zoom()
            transform_matrix = np.dot(transform_matrix, Z)
        if self.random_axis_flip:
            AF = self._random_axis_flip()
            transform_matrix = np.dot(transform_matrix, AF)

        return transform_matrix

    def _random_rotation( self ):

        rg = self.rotation_range

        axis_of_rotation = np.random.permutation([0, 1, 2])
        axis_of_rotation = axis_of_rotation[:np.random.randint(low=1, high=4)]

        thetas = [np.pi / 180 * np.random.uniform(-rg, rg) for _ in axis_of_rotation]

        rotation_matrix = np.eye(4)
        for ax, th in zip(axis_of_rotation, thetas):
            c, s = np.cos(th), np.sin(th)
            R = np.eye(4)
            if ax == 0:
                R[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            if ax == 1:
                R[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            if ax == 2:
                R[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotation_matrix = np.dot(rotation_matrix, R)
        return rotation_matrix

    def _random_shear( self ):

        shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range, 6))

        transform_matrix = np.array([
            [1., shear[0], shear[1], 0.],
            [shear[2], 1., shear[3], 0.],
            [shear[4], shear[5], 1., 0.],
            [0., 0., 0., 1.]
        ])
        return transform_matrix

    def _random_shift( self ):

        if not isinstance(self.shift_range, list):
            self.shift_range = [self.shift_range] * 3

        rg = self.shift_range
        t = [np.random.uniform(-rg[i], rg[i]) for i in range(3)]
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 3] = t
        return transform_matrix

    def _random_zoom( self ):

        if not isinstance(self.zoom_range, list):
            self.zoom_range = [1 + self.zoom_range, 1 - self.zoom_range]

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = np.diag([zx, zy, zz])

        return transform_matrix

    def _random_axis_flip( self ):

        flip = lambda: np.random.choice([1, -1])
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = np.diag([flip(), flip(), flip()])
        return transform_matrix

    def _transform_matrix_offset_center( self, matrix, x, y, z ):

        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix, reset_matrix = np.eye(4), np.eye(4)
        offset_matrix[0:3, 3] = [o_x, o_y, o_z]
        reset_matrix[0:3, 3] = [-o_x, -o_y, -o_z]
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


class RandomNonLinearDisplacement(Augmentation):

    def __init__( self, magnitude, smoothness, interpolator=sitk.sitkLinear, cval=0. ):
        self.magnitude = magnitude
        self.smoothness = smoothness
        self.interpolator = interpolator
        self.cval = cval
        super(RandomNonLinearDisplacement, self).__init__()

    def __call__( self, image, *args ):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        displacement = args[0]

        if len(args) > 1:
            for df in args:
                displacement = displacement * df

        displacement = sitk.GetImageFromArray(displacement, isVector=True)
        displacement = sitk.DisplacementFieldTransform(displacement)

        # apply it
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(self.interpolator)
        resampler.SetDefaultPixelValue(self.cval)
        resampler.SetTransform(displacement)
        image = resampler.Execute(image)
        return image

    def get_random_params( self, image ):

        np.random.seed(self.random_seed)

        # generate a random displacement field
        displacement = np.random.randn(*image.GetSize() + (3,))
        displacement = gaussian_filter(displacement, self.smoothness, mode='constant', cval=0)
        displacement = displacement * self.magnitude

        return displacement


class RandomNoise(Augmentation):

    def __init__(self, gaussian=None, poisson=None, gamma=None):
        self.gaussian = gaussian
        self.poisson = poisson
        self.gamma = gamma
        super(RandomNoise, self).__init__()

    def __call__( self, image, *args ):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        transform = args[0]

        if len(args) > 1:
            for t in args[1:]:
                transform = transform + t

        transform = sitk.GetImageFromArray(transform)
        image = sitk.Add(image, transform)

        return image


    def get_random_params( self, image ):

        np.random.seed(self.random_seed)

        s = image.GetSize()
        transform = np.zeros(s)

        if self.gaussian is not None:
            transform += np.random.normal(self.gaussian[0], self.gaussian[1], size=s)

        if self.poisson is not None:
            transform += np.random.poisson(self.poisson[0], size=s)

        if self.gamma is not None:
            transform += np.random.gamma(self.gamma[0], self.gamma[1], size=s)

        return transform


class RandomDistortion(Augmentation):

    def __init__(self, range_add=None, range_mul=None, range_pow=None):
        self.range_add = range_add
        self.range_mul = range_mul
        self.range_pow = range_pow
        super(RandomDistortion, self).__init__()

    def __call__(self, image, *args):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        transform = args[0]

        if len(args) > 1:
            for t in args:
                transform = transform + t

        add, mul, pow = transform

        neg = sitk.Cast(image < 0, image.GetPixelID())
        pos = sitk.Cast(image > 0, image.GetPixelID())

        image = sitk.Multiply(sitk.Add(image, add), mul)
        image = sitk.Pow(pos * image + 1e-3, pow) - sitk.Pow(- neg * image + 1e-3, pow)

        return image

    def get_random_params( self, image ):

        np.random.seed(self.random_seed)

        add = 0
        mul = pow = 1

        if self.range_add is not None:
            add = np.random.uniform(self.range_add[0], self.range_add[1])
        if self.range_mul is not None:
            mul = np.random.uniform(self.range_mul[0], self.range_mul[1])
        if self.range_pow is not None:
            pow = np.random.uniform(self.range_pow[0], self.range_pow[1])

        return np.array([add, mul, pow])

