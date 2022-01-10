from SimpleITK import Image


class Pipeline(object):
    """
    Defines a set of operations to be applied on an image.
    Operations can be either transforms or reads.
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
    """
    Defines the operation to read an image from disk.
    """
    def __init__(self):
        pass

    def __call__(self, filename):
        pass


class Transform(object):
    """
    Defines the operation to be applied on an image.
    """
    def __init__(self):
        pass

    def __call__(self, image):
        return image


