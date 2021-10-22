from .base import Read

import SimpleITK as sitk

# !- Readers for common medical imaging formats

class ReadVolume(Read):

    def __call__( self, filename ):

        try:
            image = sitk.ReadImage(filename)
            return image
        except:
            return None


class ReadDICOM(Read):

    def __call__( self, filename ):

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            return image
        except:
            return None