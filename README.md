# Features

This library has been designed to facilitate the extraction of subpart of a PDS IMAGES from the LOLA/LROC_WAC experiment. 

- **Highlighted features**

    - Unlike previous  python PDS  library, this module  allows to
      easily extract part of a PDS image without loading the whole
      thing into memory (Sizes can be up to Gigabytes).
    - Great for looking at geological unit on the Moon (Currently built to deal with crater and domes

# Requirement
- python 2.7>
- numpy
- scipy
- pandas
- os, sys, distutils
- pvl to read binary header
- urlib,requests for downloading PDS FILES
- cartopy/matplotlib/mpl_toolkits(basemap) for plotting

# Setup

- Install the above dependencies
- Get in the the PDS_Extrator.py file and replace in the header of the BinaryTable class

		racine = '/Users/thorey/Documents/These/Projet/FFC/CraterInspector'

	by your corresponding path. Something like 

		racine = '/PathToCraterInspector/CraterInspector
- That's it !

# Example 

## Loading part of an image

Lola PDS Images can be found [here](http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/) and WAC images can be found [here](http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/).

Let's say we want to load a window defined at the bottom left by (lon0,lat0) and at the upper right by (lon1,lat1) on the 128 ppd (pixel/degree) LOLA image. 

		from PDS_Extractor import *
		
		img = BinaryTable('LDEM_128')
		X, Y, Z = img.Extract_Grid(lon0,lon1,lat0,lat1)

which returns an array of longitudes (X), an array of latitudes (Y) and the grid of altitude are contained in Z. No more. For a window 10° by 10°, it runs in less than 2s on my mac book pro.


## Looking at impact crater ?
Let's say, we want to get some detail about the crater Copernicus.
		
	from Structure import *
	Copernicus = Structure('n','Copernicus','Crater')
	Copernicus.ppdlola = 512
	Copernicus.ppdwac = 128
    Copernicus.Overlay(True)

![Alt](https://raw.githubusercontent.com/cthorey/CraterInspector/master/Image/Copernicus.png)

let you with this nice beautiful plot which overlay a WAC image and a LOLA image. Pixel/degree are pretty high by default. 

For a specific location, the program is able to automatically detect the corresponding patch images at the lunar surface and proposed to download it for you. Be careful with large resolution though, downloads can be very long. 

The default window in centred on the crater with a radius equal to the 80% of the crater diameter. However this can easily be changed and for instance, zooming in resume to 

	    Copernicus.Change_window(0.4*Copernicus.Diameter)
		Copernicus.Overlay(True)
![Alt](https://raw.githubusercontent.com/cthorey/CraterInspector/master/Image/CopernicusZoom.png)

If you prefer working with the array directly, use the method Get_Arrays...

    Xl , Yl , Zl = Copernicus.Get_Arrays('Lola')
    Xw , Yw , Zw = Copernicus.Get_Arrays('Wac')

They can then be used for further analysis, histograms of the topography...

## Topographic profiles

The **Structure** class also contained a method which let your draw topographic profiles (or WAC profile if you want) without effort. For instance, if we look at an intrusive dome called 'M13' within the lunar maria and we want to plot three topographic profile
	- one vertical passing through the centre
	- one horizontal passing through the centre
	- one oblique 
				
		from Structure import *
		
		M13 = Structure('n','M13','Dome')
		M13.Change_window(.9*M13.Diameter)
		M13.ppdlola = 512
		midlon = (M13.window[0]+M13.window[1])/2.0
		midlat = (M13.window[2]+M13.window[3])/2.0
		profile1 = (midlon,midlon,11.1,12.5)
		profile2 = (M13.window[0]+0.2,M13.window[1]-0.2,midlat,midlat)
		profile3 = (360-32.1,360-31.3,11.1,12.5)
		M13.Draw_Profile((profile1,profile2,profile3), save = True)
 
![Alt](https://raw.githubusercontent.com/cthorey/CraterInspector/master/Image/BaseProfile.png)

# Data

   - **Data_Crater.csv** : Comprehensive dataset of lunar impact craters.
	    - Source of the dataset
		    + [Salamuniccar et al 2011](http://www.sciencedirect.com/science/article/pii/S0032063310003405)
		    + [Jozwiak et al 2015](http://dx.doi.org/10.1016/j.icarus.2014.10.052)
		    + [Losiak et al 2009](http://adsabs.harvard.edu/abs/2009LPI....40.1532L)
		-  Columns
			- Name (when given), Lat (degree) , Long (degree) , Diameter (km) + Type (1 = Floor-fractured crater, 0 = Normal impact craters)
			- A comprehensive description of FFC can be found [here](http://www.lpod.org/cwm/DataStuff/ffc.htm)

	- **Data_Dome.csv**
		- Intrusive domes found in [Wöhler et al 2009](http://linkinghub.elsevier.com/retrieve/pii/S0019103509003236)

* **PDS_FILE**
	- Binary image .IMG files to download on the [PDS node](http://pds-geosciences.wustl.edu/)
	- Library optimize for LOLA and LROC_WAC Images (FOLDERs inside PDS_FILE). Nevertheless, It could be easily extended to account fo other type of data.

