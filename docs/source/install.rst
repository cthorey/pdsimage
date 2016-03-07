Get the module up and running
=============================

Mac OS X
--------

You first need to pull the directory from git on your computer.
From a terminal, tape ::

    git clone https://github.com/cthorey/pdsimage

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
    2. Use pip on the command line::

           pip install pdsimage

       Then,  if  the  basemap  tool_kit  is  not  installed  in  your
       environment,  you will  need  to  install it  as  well for  the
       library to work properly.
       First, make sure to install the dependency GEOS/PROJ4::

           brew install geos
           brew install proj

       should give you a working  installation for both packages. Then,
       install basemap from source::

            pip install https://downloads.sourceforge.net/project/matplotlib/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz

       Now, you are good to go as well.
       
Unix
----

While I did  not try it myself, repeating the  same exercise than above
on whatever Unix distributed system should work properly.

.. _here:
    http://stiglerdiet.com/blog/2015/Nov/24/my-python-environment-workflow-with-conda/

.. _gdal:
    http://www.gdal.org/

.. _Cartopy:
    http://scitools.org.uk/cartopy/
