Get the module up and running
=============================

Mac OS X
--------

You first need to pull the directory from git on your computer.
From a terminal, tape ::

    git pull https://github.com/cthorey/pdsimage

Then, the easiest way to get everything up and running is to work
within   the    pdsimage   conda    environment   provided    in   the
**environment.yml** file.

For those who have conda installed on their machine, just use::

    conda env create -f environment.yml
    source activate pdsimage

This will create the pdsimage environment which contains all the
dependency for the library to work properly.

For those who don't use conda, you can:

    1. follow the instruction available `here`_ to install it and give
       it a try. It really make everything easier ;)
    2. Use pip on the commande line::

           pip install -r requirements.txt
           pip install pdsimage

However, you might run into  some problem of depencies. In particular,
you might need to install the geospatial libray `gdal`_ as one of the
module use to realize the  image used `Cartopy`_ which itself depends
on **gdal** for proper running.

The easiest is to use brew::
  
    brew install gdal

Unix
----

While I did  not try it myself, repeting the  same exercice than above
on whatever unix distributed sytem should work properly.

.. _here:
    http://stiglerdiet.com/blog/2015/Nov/24/my-python-environment-workflow-with-conda/

.. _gdal:
    http://www.gdal.org/

.. _Cartopy:
    http://scitools.org.uk/cartopy/
