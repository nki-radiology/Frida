# !- Interface for base classes, defining atomic operations

class Read(object):
    '''
    Defines the operation to read an image from disk. 
    '''
    def __init__(self):
        pass

    def __call__(self, filename):
        pass


class Transform(object):
    '''
    Defines the operation to be applied on an image. 
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        return image


class Pipeline(object):
    '''
    Defines a set of operations to be applied on an image. 
    Operations can be either transforms or reads. 
    '''
    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, inputs):

        for s in self.steps:
            inputs = s(inputs)

        return inputs

# !- Interface for augmentation classes

class Augmentation(Transform):

    def __init__(self):
        self.random_seed = 0
        super(Augmentation, self).__init__()

    def __call__(self, image, *args):
        return image

    def get_random_params(self, image):
        return None


