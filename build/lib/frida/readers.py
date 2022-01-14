from .base import Read
import SimpleITK as sitk


# !- Readers for common medical imaging formats


class ReadVolume(Read):
    """Reads imaging data.
    These are the format commonly supported by ITK.

    :Example:

    >>> ppl = Pipeline(ReadVolume(), ToNumpyArray())
    >>> arr = ppl('path/to/image')

    :param filename: full path of the volume
    :type filename: str
    ...
    :return: the image if successful, None otherwise
    :rtype: SimpleITK.Image, or None
    """
    def __call__(self, filename):

        try:
            image = sitk.ReadImage(filename)
            return image
        except:
            # TODO: improve exception handling
            return None


class ReadDICOM(Read):
    """Reads a DICOM series data.

    :Example:

    >>> ppl = Pipeline(ReadDICOM(), ToNumpyArray())
    >>> arr = ppl('path/to/DICOM_series')

    :param filename: full path of the volume
    :type filename: str
    ...
    :return: the image if successful, None otherwise
    :rtype: SimpleITK.Image, or None
    """
    def __call__(self, filename):

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            return image
        except:
            # TODO: improve exception handling
            return None
