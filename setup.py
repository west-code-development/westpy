from setuptools import setup
from setuptools import find_packages
import json

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('VERSION.json', 'r') as file:
    data = json.load(file)

setup(name='westpy',
      version=data['version'],
      packages=find_packages(),
      description='Python analysis tools for WEST',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/west-code-development/westpy.git',
      author='Marco Govoni',
      author_email='mgovoni@anl.gov',
      license='GPLv3',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pyyaml',
          'datetime',
          'requests',
          'mendeleev',
          'setuptools',
          'urllib3',
          'sphinx',
          'sphinx_rtd_theme',
          'py3Dmol',
          'pyscf',
          'ipython',
          'pandas',
          'six',
          'ase',
          'qiskit_nature',
          'h5py'
      ],
      python_requires='>=3.6, <4',
      zip_safe=True)
