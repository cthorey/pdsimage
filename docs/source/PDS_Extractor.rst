Extracting data on a specific region
====================================

The  main aim  of  this library  is to  facilitate  the extraction  of
knowledge over a specific region at the lunar surface. For the moment,
only topography and  images are available but more should  come in the
future.

This  module is  composed of  three  different class  which I  briefly
detail in the following.

BinaryTable
-----------

This class allows to read binary file from the PDS site to retreive
the desired information.

The different possible binary files are all the **ldem** files listed
here `LRO/LOLA ldem`_ for topography and all the images listed here
`LRO/WAC wac`_ for WAC images.

For  instance,  if you  want  to  read  the information  contained  in
**ldem_512_90s_45s_090_180.img**, simply load it  in a variable called
**ldem** here for the example::

    from PDS_Extractor import BinaryTable
    ldem = BinaryTable('ldem_16')

If the file is  not already on your computer, the  program is going to
ask if you want to download it  on the fly.  Be carefull with the size
of high resolution image though, the download can take a few minutes !

When creating the object, the class store information contained in the
header,  either a  separate *.lbl*  file  for files  on the  `LRO/LOLA
ldem`_ or  contained in the *.img*  file for files on  the `LRO/WAC
wac`_. All information can be access as attribute of the object.

However,  the class  does not  load the  data contained  in the  image
during initialization.  PDS images  can be  very large  and resolution
higher than 16  pixel per degree (ppd) usually do  not fit into memory
if you want to load the full image.

In agreement, the class proposed two methods to extract the data:

    - **extract_all** which extract the  whole image. Again, make sure
      that information will fit into your computer memory.
    - **extract_grid** which  extract the  data contained in  a window
      defined by its latitude/longitude boundaries.
      
In  addition,  the  class  proposed  two methods  than  can  give  the
latitude/longitude  boundaries given  a  center long/lat  point and  a
radius:

    - **lambert_window**  will return  square Lambert  Azimuthal equal
      area projection of the region and
    - **cylindrical_window**  will   return  the   simple  cylindrical
      projection of the region.

For instance, taken  the object ``ldem`` we have just  created, we can
ask to get the data covering a  region centred at (120 E, 60S) spaning
a radius of 5 km by simply taping::

    boundary = ldem.lambert_window(5,-60,120)
    X,Y,Z = ldem.extract_grid(*boundary)
    
By default, if the given regime  boundaries end up outside of the map,
the class  will set resize them  so that they correspond  to the image
boundary. This should not happen for resolution smaller than 64 ppd as
each image contains the whole lunar surface though.

However, for  large resolution, the lunar  surface is cut off  in many
subregions, i.e.  subimages of reasonable  size and it can be annowing
to  identify  which one  is  relevant  for  the particular  region  of
interest.

For  this reason,  this module  also integrated  two other  class, one
specialized in  the topography  and one  specialized for  images, that
should be prefered to BinaryTable. 

WacMap
-----------
WacMap is class which handle the treatment of WAC images.
Indeed, getting an image over a specific region at the lunar
surface can be annowing, you have to:

    - Identify the region of interest.
    - Identify  the  relevant *img*  file  (  possibly multiple  due  to
      overlap).
    - Load the information about each *img*  files to be able to read the
      binary record into it.
    - Extract the data from the binary record.

With WacMap, you basically only need to  feed it an imput region and a
desired resolution, and it does all that work for you.

For instance, for an image  with a resolution of 512 ppd centred
centred at (120 E, 60S) spaning a radius of 20 km, I simply use::

    from PDS_Extractor import WacMap
    boundary = ldem.lambert_window(20,-60,120)
    wac_image = WacMap(512,*boundary)
    X,Y,Z = wac_image.image()

with ``X`` containing  the array of longitude where  data exist, ``Y``
the latitudes and ``Z`` the information about the wac image itself. If
you want to take a rapid look, just try::

    import matplotlib as plt
    plt.imshow(Z)

.. image:: _static/test.png
   :align: center
           
LolaMap
-----------

Same for the topography !

Say we'd like to overlap our nice images with the topography to get
a beatufiful image to show off with, simply add::

    from PDS_Extractor import LolaMap
    ldem_image = LolaMap(512,*boundary)
    Xl,Yl,Zl = ldem_image.Image()

Now,  you  can  simply  overlay  Zl   and  Z  to  get  it.  While  the
``PDS_Extractor``  is  designed to  facilate  the  extraction of  data
(X,Y,Z) arrays,  the library also  contained a second module  that you
can also use if you want to make beautiful images ``Structure``.


.. _LRO/LOLA ldem:
    http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/

.. _LROC/WAC wac:
    http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/


Index
-----
.. automodule:: pdsimage.PDS_Extractor
    :members:
    :undoc-members:
    :show-inheritance:

