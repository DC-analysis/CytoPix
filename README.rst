|CytoPix|
=========

|PyPI Version| |Build Status| |Coverage Status| |Docs Status|


**CytoPix** is a graphical user interface for manually segmenting
deformability cytometry (DC) images. The mask data (labeled images)
can be used for training machine-learning based segmentation
algorithms and are also generally useful for benchmarking segmentation
algorithms.


Documentation
-------------

The documentation is available at
`cytopix.readthedocs.io <https://cytopix.readthedocs.io>`__.


Installation
------------
Installers for Windows and macOS are available at the `release page
<https://github.com/DC-analysis/CytoPix/releases>`__.

If you have Python installed, you can install CytoPix from PyPI

::

    # graphical user interface
    pip install cytopix


Execution
---------
If you have installed CytoPix from PyPI, you can start it with

::

    cytopix
    # or
    python -m cytopix


Citing CytoPix
-----------------
Please cite CytoPix either in-line

::

  (...) using the pixel-based segmentation software CytoPix version X.X.X
  (available at https://github.com/DC-analysis/CytoPix).

or in a bibliography

::

  Paul Müller and others (2025), CytoPix version X.X.X: Pixel-based
  manual segmentation of deformability cytometry images [Software].
  Available at https://github.com/DC-analysis/CytoPix.

and replace ``X.X.X`` with the version of CytoPix that you used.


Testing
-------

::

    pip install -e .
    pip install -r tests/requirements.txt
    pytest tests


.. |CytoPix| image:: https://raw.github.com/DC-analysis/CytoPix/master/docs/artwork/cytopix_splash.png
.. |PyPI Version| image:: https://img.shields.io/pypi/v/CytoPix.svg
   :target: https://pypi.python.org/pypi/CytoPix
.. |Build Status| image:: https://img.shields.io/github/actions/workflow/status/DC-analysis/CytoPix/check.yml?branch=master
   :target: https://github.com/DC-analysis/CytoPix/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/DC-analysis/CytoPix/master.svg
   :target: https://codecov.io/gh/DC-analysis/CytoPix
.. |Docs Status| image:: https://img.shields.io/readthedocs/cytopix
   :target: https://readthedocs.org/projects/cytopix/builds/
