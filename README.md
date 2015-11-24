
# CRATERINSPECTOR

Library  to easily  acces part  of the  PDS IMAGES  from LOLA/LROC_WAC
Experiment. Given a location and a radius, for instance the latitude, the longitude and the radius of the crater, this library allows to easily extract both Topography (LOLA) and WAC images of the desired area.

## Example

Let's say, we want to get some idea about the crater Copernicus.

    C = Copernicus('n','Copernicus',racine,'Crater')
    C.Overlay(True)

![Alt](https://raw.githubusercontent.com/cthorey/CraterInspector/master/Image/Crater_Copernicus.png)


## CraterInspec

Contain:

* CraterInspec:
    - **PDS_Extractor** : Classes to parse .IMG file from the PDS
    - **Structure**  :  Usefull  library to  visualize  the  desired
      structure

* Data:
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

* PDS_FILE
	- Binary image .IMG files to download on the [PDS node](http://pds-geosciences.wustl.edu/)
	- Library optimize for LOLA and LROC_WAC Images (FOLDERs inside PDS_FILE). Nevertheless, It could be easily extended to account fo other type of data.
## Requirement

- numpy
- pandas
- matplotlib
- palettable for beautiful color
- os, sys
- pvl to read binary header
- urlib,requests for downloading PDS FILES




