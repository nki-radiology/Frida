from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package for creating modular data pipeline to process medical images'

# Setting up
setup(
    name="frida",
    version=VERSION,
    author="Stefano Trebeschi",
    author_email="<s.trebeschi@nki.nl>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['simpleitk', 'numpy'],
    keywords=['python', 'medical imaging', 'radiology', 'pipeline'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)