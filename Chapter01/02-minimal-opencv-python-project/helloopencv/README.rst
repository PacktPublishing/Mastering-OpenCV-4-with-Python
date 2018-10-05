===========
HelloOpenCV
===========
OpenCV and Python first example.This first project loads an image from a specified path. Afterwards, this image is converted to grayscale and, finally, the grayscale image is saved to a specified file. Additionally, the two images (both the original and the grayscale one) are shown. 

***************
Installation
***************
.. code:: python

   python setup.py install

***************
Usage example
***************
This example can be used as follows:

.. code:: python

   python helloopencv.py

This script loads the image from "images/logo.png" and it is converted to grayscale, writing it in "images/gray_logo.png"

*****************
Development setup
*****************
To install all development dependencies:

.. code:: python

   pip install -r requirements.txt

To run the tests (verbose):

.. code:: python

   py.test -s -v helloopencvtests.py

*****************
Release History
*****************

    - 0.1.0
        - The first proper release
        - CHANGE: Rename `loadimage()` to `load_image()`
    - 0.0.1
        - Work in progress

*****************
Meta
*****************
Alberto Fernandez  – fernandezvillan.alberto@gmail.com
Distributed under the MIT license. See ``LICENSE`` for more information.
