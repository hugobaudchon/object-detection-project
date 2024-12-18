from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

print('Setting up object_detection package...')

d = generate_distutils_setup(
    packages=['object_detection'],
    package_dir={'': 'src'}
)

setup(**d)