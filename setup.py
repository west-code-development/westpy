from setuptools import setup

setup(name='westpy',
      version='3.1.0',
      packages=['westpy'],
      description='Python analysis tools for WEST',
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
          'signac'
          'urllib3'
      ],
      python_requires='>=2.7, >=3.0, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      zip_safe=True)
