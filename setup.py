from setuptools import find_packages
from setuptools import setup

setup(
    name = 'pyapep',
    version = '0.0.7',
    license = 'PNU_Seongbin_Ga',
    author = 'Seongbin Ga',
    author_email = 'sebyga@gmail.com',
    url = 'https://sebyga.github.io/HompyTest/',
    keywords = ['adsorption','PSA','process simulation'],
    description = 'This is adsorption process simulation package made by Seongbin Ga.',
    #packages = ['pyapep'],
    packages = find_packages(),
    pytho_require='>=3.5'
    #install_requires = ['numpy','scipy']
)
