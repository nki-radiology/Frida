from .base import Pipeline


class FirstToSucceed(Pipeline):
    '''
    Selects the first transform/reader to succeed. 
    
    This is useful when you do not know a priori. 
    
    For example, in a heterogeneous dataset (where some images 
    are saved as DICOM and some are Nifti).
    
    read = FirstToSucceed(ReadDICOM(), ReadVolume())
    '''
    def __init__(self, *steps):
        super(FirstToSucceed, self).__init__(*steps)

    def __call__(self, inputs):

        outputs = None
        for s in self.steps:

            outputs = s(inputs)

            if outputs is not None:
                break

        return outputs
