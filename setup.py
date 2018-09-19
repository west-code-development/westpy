from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='westpy',
      version='3.1.1',
      packages=find_packages(),
      description='Python analysis tools for WEST',
      long_description=long_description,
      long_description_content_type="text/markdown",
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
          'signac',
          'setuptools',
          'urllib3', 
          'prompt-toolkit<2.0.0',
          'sphinx', 
          'sphinx_rtd_theme'
      ],
      python_requires='>=2.7, >=3.0, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      zip_safe=True)
