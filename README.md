# CRATERINSPECTOR

Library  to easily  acces part  of the  PDS IMAGES  from LOLA/LROC_WAC
Experiment.

## CraterInspec

Contain:

* CraterInspec:
    - **PDS_Extractor** : Classes to parse .IMG file from the PDS
    - **Structure**  :  Usefull  library to  visualize  the  desired
      structure

* Data:
    - **Data_Crater.csv** : Comprehensive dataset of lunar impact (Crater, FFC , dome)
      craters. Taken from [http://www.sciencedirect.com/science/article/pii/S0032063310003405][Salamuniccar et al, 2011]
    - **Data_Dome.csv** 

## requirement

- numpy
- pandas
- matplotlib
- palettable for beautiful color
- os, sys
- pvl to read binary header
- urlib,requests for downloading PDS FILES


