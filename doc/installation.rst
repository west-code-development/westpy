.. _installation:

============
Installation
============

The recommendend installation method for **westpy** is via python install. 
The software is tested for python version 3.x and has the following dependencies: 

   - ``numpy``
   - ``scipy``
   - ``matplotlib``
   - ``datetime``
   - ``requests``
   - ``mendeleev`` 

Source Code Installation
========================

To install **westpy** you need to execute:  

.. code:: bash

    $ git clone http://greatfire.uchicago.edu/west-public/westpy.git
    $ cd westpy 
    $ python setup.py install --user
 
or simply execute: 

.. code:: bash

    $ http://greatfire.uchicago.edu/west-public/westpy.git
    $ cd westpy 
    $ make

If the name of your Python compiler is not standard, you can edit the varyable **PYT** in the file **Makefile**.  

