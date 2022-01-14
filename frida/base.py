from SimpleITK import Image


class Pipeline(object):
    """Defines an executable, sequential set of operations.
    Operations can be Transform or Read object.

    :Example:

    >>> pipl = Pipeline(ReadVolume(), Resample(2), ToNumpyArray())
    >>> im_arr = pipl('path/to/image.nrrd')

    :param steps: operations that define the pipeline
    :type steps: Transform and/or Read
    ...
    :raises Warning: if the input/output of each operation is not as expected.
    ...
    :return: transformed input
    :rtype: rtype of the last operation in the pipeline
    """
    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, inputs):

        for s in self.steps:

            if inputs is None:
                raise Exception("None inputs detected at " + str(s))

            if isinstance(s, Transform) and not isinstance(inputs, Image):
                raise Warning("Transform objects should require SimpleITK.Image objects on input and outputs. "
                              "The Transform " + str(s) + " however, did not receive a SimpleITK.Image as input. "
                              "This could be result of a custom made Transform class. ")

            if isinstance(s, Read) and not isinstance(inputs, str):
                raise Warning("Read objects should require a path in input. "
                              "The Read " + str(s) + " however, did not receive a path as input. "
                              "This could be result of a custom made Read class. ")

            inputs = s(inputs)

        return inputs

# !- Interface for base classes, defining atomic operations


class Read(object):
    """"Abstract interface, defines a read operation.
    Read operation usually have filepath on input and return a SimpleITK.Image object.
    """
    def __call__(self, filename):
        return Image()


class Transform(object):
    """"Abstract interface, defines a transform operation.
    Read operation usually have SimpleITK.Image objects on both input and output.
    """
    def __call__(self, image):
        return image


