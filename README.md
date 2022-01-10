![GitHub](https://img.shields.io/github/license/stefano-trebeschi/Frida)
# Frida

Frida results as a collection of different functions and procedures for medical image analysis that used and re-used during my PhD. 

### Getting Started

#### Dependencies

Frida requires 
* Python (>=3.6)
* NumPy (>=1.19) 
* SciPy (>=1.5.2)
* SimpleITK (>=2.0)
* SciKit-Learn (>=0.24.2)
* SciKit-Image (>=0.17.2)

Older packages might still work, but not guaranteed. 

### Usage

Frida is written to be a modular package to design pipelines for medical image processing. 
With that in mind, there are three main operations/modules that you can find in frida: 
* a ```Read``` operator, which takes in input a physical location and returns a ```SimpleITK.Image```;
* a ```Transform``` operator, which takes in input a ```SimpleITK.Image``` and returns a ```SimpleITK.Image```;
* a ```Pipeline``` operator, which is defined as a set of operations to be executed sequentially.

Here is an example of these

```
from frida import Pipeline
from frida.readers ReadDICOM
from frida.transforms import PadAndCropTo, Resample

my_pipeline = Pipeline(
    ReadDICOM(),
    Resample((2., 2., 1.), cval=0),
    PadAndCrop((160, 160, 160), cval=0)
)

image = my_pipeline(r'path/to/image/nrrd_or_dicom')
```

This pipeline reads a DICOM image, resamples it to anisotropic voxel size 2x2x1 units (mm, most likely), and applies 
pad and cropping operations to bring the size of the image to a cube of size 160 voxels.

There are two additional types of ```Transform``` operators with special functions:

* a ```Cast``` operator, which takes in input a ```SimpleITK.Image``` and returns a format defined by the user; 
* a ```Augemtation``` operator, which applies a transform operator to the image with a random component in it.

Here is an example of a pipeline using them

```
from frida import Pipeline
from frida.readers ReadDICOM
from frida.transforms import PadAndCropTo

from frida.transforms.aumentations import RandomLinearDisplacement
from frida.transforms.cast import ToNumpyArray

my_pipeline = Pipeline(
    ReadDICOM(),
    PadAndCrop((160, 160, 160)),
    RandomLinearDisplacement(rotation_range=10, zoom_range=0.5),
    ToNumpyArray(add_batch_dim=True, add_singleton_dim=False)
)

image_arr = my_pipeline(r'path/to/image/nrrd_or_dicom')
print(image_arr.shape)

>> (1, 160, 160, 160)
```

More can be found on the documentation!

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!