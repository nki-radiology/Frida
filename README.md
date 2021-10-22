## Usage ##

This are common functions and procedures I have used during my PhD. 
I thought to put them together in a single package. 

Very early development. 

```
from frida.base import Pipeline
from frida.reader ReadVolume, ReadDICOM
from frida.pipeline import FirstToSucceed
from frida.transforms import \ 
    PadAndCropTo, Resample, ITKFilter, ProbabilisticTransform, ToNumpyArray
from frida.augmentation import RandomLinearDisplacement, RandomNoise

from SimpleITK import AdaptiveHistogramEqualizationFilter
from SimpleITK import GaussianNoiseImageFilter

loader = Pipeline(
        FirstToSucceed(ReadVolume(), ReadDICOM()),
	Resample(2.), 
	PadAndCropTo((160, 160, 160)),
	LinearAugmentation(rotation_range=10.),
        ProbabilisticTransform(probability=0.5, RandomNoise(gaussian=(0., 1.), poisson=(1.))),
	ITKFilter(AdaptiveHistogramEqualizationFilter()),
        ToNumpyArray(add_singleton_dim=True)
)

image = loader(r'path/to/image/nrrd_or_dicom')
```
