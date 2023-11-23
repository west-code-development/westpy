.. _installation:

============
Installation
============

The recommendend installation method for **WESTpy** is via python install.
The software is tested for Python version 3.x and has the following dependencies:

   - ``numpy``
   - ``scipy``
   - ``matplotlib``
   - ``pyyaml``
   - ``datetime``
   - ``requests``
   - ``mendeleev``
   - ``signac``
   - ``setuptools``
   - ``urllib3``
   - ``prompt-toolkit``
   - ``sphinx``
   - ``sphinx_rtd_theme``
   - ``py3Dmol``
   - ``pyscf``
   - ``ipython``
   - ``pandas``
   - ``six``
   - ``ase``
   - ``qiskit_nature``
   - ``h5py``

The dependencies will all be installed automatically, following instructions reported below.


Source Code Installation
========================

To install **WESTpy** you need to execute:

.. code:: bash

    $ git clone https://github.com/west-code-development/westpy.git
    $ cd westpy
    $ python setup.py install --user

or simply execute:

.. code:: bash

    $ git clone https://github.com/west-code-development/westpy.git
    $ cd westpy
    $ make

If the name of your Python interpreter is not standard, you can edit the varyable **PYT** in the file **Makefile**.

