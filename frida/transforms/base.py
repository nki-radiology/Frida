from ..base import Transform


class Augmentation(Transform):

    def __init__(self):
        self.random_seed = 0
        super(Augmentation, self).__init__()

    def __call__(self, image, *args):
        return image

    def get_random_params(self, image):
        return None


class Cast(Transform):

    def __init__(self):
        self.random_seed = 0
        super(Cast, self).__init__()

    def __call__(self, image, *args):
        pass