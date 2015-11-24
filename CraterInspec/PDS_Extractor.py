# Import library
import numpy as np
import pickle
import pandas as pd
import os,sys

#Library to help read header in binary WAC FILE
import pvl
from pvl import load as load_label

# Helper to catch images from URLs
import urllib,requests

        
class BinaryTable(object):
    ''' Class which is able to read PDS image file for LOLA/WAC images.

    LOLA - All information can be found at
    http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/aareadme.txt
    In particular, HEADER in a separate file .LBL that contains all the informations.
    Read manually

    WAC - All information can be found at
    http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/AAREADME.TXT
    In particular, HEADER in the binary file that contain all the information -
    Read with the module pvl

    This class has a method able to download the images, though it might be better to download
    them before as it takes a lot of time, especially for larger resolution.

    Both are NASA PDS FILE - Meaning, they are binary table whose format depends on
    the file. All the information can be found in the Header whose reference are above.
    Line usualy index latitude while sample on the line refers to longitude.

    THIS CLASS SUPPORT ONLY CYLINDRICAL PROJECTION.
    PROJECTION : [WAC : 'EQUIRECTANGULAR', LOLA : '"SIMPLE"']
    FURTHER WORK IS NEEDED FOR IT TO BECOMES MORE GENERAL.
    
    parameter:
    - fname : name of the binary file without the extension
    - self.PDS_FILE : Path where the binary images are download. It should
    consit of two folder, LOLA and LROC_WAC.
    - self.LOLApath : path for LOLA images
    - self.WACpath : path for WAC images

    method:
    - self._Category() : Identify WAC or LOLA Image according to name formating
    - self._Load_Info_LBL : Load the Header of the file (using pvl for WAC and the .LBL
    file for LOLA) and return all the specification as attribute of the class.
    - self.Lat_id : From line return lat
    - self.Long_id : From sample, return long
    - self.Sample_id : From long, return sample
    - self.Line_id : From lat, return line
    - self.Array : Read the binary file where it should and return the values in an
    array
    - self.Extract_All() : Extract the whole image in the file
    - self.Extract_Grid() : Extract only part of the image centered around the structure
    -self.Lamber_Window, self.cylindrical_Window : Return longmin,longmax,latmin,latmax in
    both configuration.
    
    '''

    def __init__(self,fname):
        ''' Parameter
        self.name : name of the file
        self._Category : Identify weither its WAC/LOLA image
        self._Load_Info_LBL : Load corresponding information '''

        self.fname = fname
        self.PDS_FILE = '/Users/thorey/Documents/These/Projet/FFC/Classification/PDS_FILE/'
        self.LOLApath = os.path.join(self.PDS_FILE,'LOLA')
        self.WACpath = os.path.join(self.PDS_FILE,'LROC_WAC')
        self._Category()
        self._maybe_download()
        self._Load_Info_LBL()

        assert self.MAP_PROJECTION_TYPE in ['"SIMPLE','EQUIRECTANGULAR']

    def _Category(self):
            
        if self.fname.split('_')[0] == 'WAC':
            self.Grid = 'WAC'
            self.img = os.path.join(self.WACpath, self.fname+'.IMG')
            self.lbl = ''
        elif self.fname.split('_')[0] == 'LDEM':
            self.Grid = 'LOLA'
            self.img = os.path.join(self.LOLApath, self.fname+'.IMG')
            self.lbl = os.path.join(self.LOLApath, self.fname+'.LBL')
        else:
            print 'Not implemented yet'
            raise Exception

    def _report(self,blocknr, blocksize, size):
        ''' helper for downloading the file '''
        current = blocknr*blocksize
        sys.stdout.write("\r{0:.2f}%".format(100.0*current/size))

    def _downloadFile(self,url,fname):
        ''' Download the file '''
        print "The file %s need to be download - Wait\n "%(fname.split('/')[-1])
        urllib.urlretrieve(url, fname, self._report)
        print "\n The download of the file %s has succeded \n "%(fname.split('/')[-1])
        
    def _maybe_download(self):
        if self.Grid == 'WAC':
            urlpath = 'http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/'
            r = requests.get(urlpath) # List file in the cloud        
            images = [elt.split('"')[7].split('.')[0] for elt in r.iter_lines() if len(elt.split('"'))>7]
            if not os.path.isfile(self.img):
                urlname = os.path.join(urlpath , self.img.split('/')[-1])
                self._downloadFile(urlname,self.img)
            elif self.fname not in images:
                print 'The image does not exist on the cloud !'
                raise NameError

        elif self.Grid == 'LOLA':
            urlpath = 'http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/'        
            r = requests.get(urlpath) # List file in this server        
            images = [elt.split('"')[7].split('.')[0] for elt in r.iter_lines() if len(elt.split('"'))>7]                      
            if (not os.path.isfile(self.img)) and (self.fname in images):
                urlname = os.path.join(urlpath , self.img.split('/')[-1])
                self._downloadFile(urlname,self.img)
                urlname = os.path.join(urlpath , self.lbl.split('/')[-1])
                self._downloadFile(urlname,self.lbl)
            elif self.fname not in images:
                print 'The image does not exist on the cloud !'
                raise NameError
                
    def _Load_Info_LBL(self):
        if self.Grid == 'WAC':
            label = load_label(self.img)
            for key,val in label.iteritems():
                if type(val)== pvl._collections.PVLObject:
                    for key,value in val.iteritems():
                        try:
                            setattr(self,key,value.value)
                        except:
                            setattr(self,key,value)
                else:
                    setattr(self,key,val)
            self.start_byte = self.RECORD_BYTES
            self.bytesize = 4
            self.projection = str(label['IMAGE_MAP_PROJECTION']['MAP_PROJECTION_TYPE'])
            self.dtype = np.float32
        else:
            with open(self.lbl, 'r') as f:
                for line in f:
                    attr = [f.strip() for f in line.split('=')]
                    if len(attr) == 2:
                        setattr(self,attr[0],attr[1].split(' ')[0])
            self.start_byte = 0
            self.bytesize = 2
            self.projection = ''
            self.dtype = np.int16
                
    def Lat_id(self,line):
        ''' Return the corresponding line latitude'''
        if self.Grid == 'WAC':
            lat = ((1 + self.LINE_PROJECTION_OFFSET- line)*self.MAP_SCALE*1e-3/self.A_AXIS_RADIUS)
            return lat*180/np.pi
        else:
            lat =  float(self.CENTER_LATITUDE) - (line -float(self.LINE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
            return lat
    
    def Long_id(self,sample):
        ''' Return the corresponding sample longitude'''            
        if self.Grid == 'WAC':
            lon = self.CENTER_LONGITUDE + (sample - self.SAMPLE_PROJECTION_OFFSET -1)*self.MAP_SCALE*1e-3/(self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0))
            return lon*180/np.pi
        else:
            lon = float(self.CENTER_LONGITUDE) + (sample -float(self.SAMPLE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
            return lon
    
    def Sample_id(self,lon):
        ''' Return the corresponding longitude sample'''            
        if self.Grid == 'WAC':
            return np.rint(float(self.SAMPLE_PROJECTION_OFFSET)+1.0+\
                (lon*np.pi/180.0-float(self.CENTER_LONGITUDE))*self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0)/(self.MAP_SCALE*1e-3))
        else:
            return np.rint(float(self.SAMPLE_PROJECTION_OFFSET) + float(self.MAP_RESOLUTION)\
                           * (lon - float(self.CENTER_LONGITUDE))) + 1
    def Line_id(self,lat):
        ''' Return the corresponding latitude line'''            
        if self.Grid == 'WAC':
            return np.rint(1.0 + self.LINE_PROJECTION_OFFSET-self.A_AXIS_RADIUS*np.pi*lat/(self.MAP_SCALE*1e-3*180))
        else:
            return np.rint(float(self.LINE_PROJECTION_OFFSET) - float(self.MAP_RESOLUTION)\
                           * (lat - float(self.CENTER_LATITUDE))) + 1

    def Array(self,size_chunk,start,bytesize):
        ''' Read part of the binary file
        size_chunk : taille du truc a lire
        start : debut
        bytesize : taille en byte '''

        with open(self.img,'rb') as f1:
            f1.seek(self.start_byte+start*self.bytesize)
            data = f1.read(size_chunk*self.bytesize)
            Z = np.fromstring(data, dtype=self.dtype, count = size_chunk)
            return Z

    def Extract_All(self):
        ''' Extract all the image and return three tables X (longitude),
        Y(latitude) and Z(values) '''
        
        longmin,longmax,latmin,latmax = self.Boundary()
        sample_min,sample_max = map(int,(self.SAMPLE_FIRST_PIXEL,self.SAMPLE_LAST_PIXEL))
        line_min,line_max = map(int,(self.LINE_FIRST_PIXEL,self.LINE_LAST_PIXEL))
        
        X = np.array(map(self.Long_id,(range(sample_min,sample_max+1,1))))
        Y = np.array(map(self.Lat_id,(range(line_min,line_max+1,1))))
        for i,line in enumerate(range(int(line_min),int(line_max)+1)):
            start = (line-1)*int(self.SAMPLE_LAST_PIXEL)+sample_min
            chunk_size = int(sample_max-sample_min)
            Za = self.Array(chunk_size,start,self.bytesize)
            if i == 0:
                Z = Za
            else:
                Z = np.vstack((Z,Za))

        X,Y = np.meshgrid(X,Y)

        return X , Y , Z
        
    def Extract_Grid(self,radius,lat0,long0):
        ''' Extract part of the image centered around (lat0,long0), both in degree,
        with a given radius window ( in km ).
        Return three tables X (longitude),Y(latitude) and Z(values) '''
        
        longmin,longmax,latmin,latmax = self.Cylindrical_Window(radius,lat0,long0)
        sample_min,sample_max = map(int,map(self.Sample_id,[longmin,longmax]))
        line_min,line_max = map(int,map(self.Line_id,[latmax,latmin]))
        
        X = np.array(map(self.Long_id,(range(sample_min,sample_max+1,1))))
        Y = np.array(map(self.Lat_id,(range(line_min,line_max+1,1))))
        for i,line in enumerate(range(int(line_min),int(line_max)+1)):
            start = (line-1)*int(self.SAMPLE_LAST_PIXEL)+sample_min
            chunk_size = int(sample_max-sample_min+1)
            Za = self.Array(chunk_size,start,self.bytesize)
            if i == 0:
                Z = Za
            else:
                Z = np.vstack((Z,Za))

        X,Y = np.meshgrid(X,Y)

        return X , Y , Z
                
    def Boundary(self):
        ''' Return the boundary of the image considered '''
        return (int(self.WESTERNMOST_LONGITUDE),
                int(self.EASTERNMOST_LONGITUDE),
                int(self.MINIMUM_LATITUDE),
                int(self.MAXIMUM_LATITUDE))

    def _kp_func(self,lat,lon,lat0,long0):
            
        kp = float(1.0) + np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-long0)
        kp = np.sqrt(float(2)/kp)
        return kp
        
    def Lambert_Window(self,radius,lat0,long0):
        ''' Return the equal area squared azimutahl projection of a window centered
        at (lat0,long0)  with a given radius of radius.'''
        
        radius = radius*360.0/(np.pi*2*1734.4)
        radius = radius*np.pi / 180.0
        lat0 = lat0*np.pi/180.0
        long0 = long0*np.pi/180.0

        bot = self._kp_func(lat0-radius,long0,lat0,long0)
        bot = bot * ( np.cos(lat0)*np.sin(lat0-radius) - np.sin(lat0)*np.cos(lat0-radius) )
        x = bot
        y = bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho/float(2.0))
        latll = np.arcsin(np.cos(c)*np.sin(lat0)  + y*np.sin(c)*np.cos(lat0)/rho ) * float(180.0) / np.pi
        lon = long0  + np.arctan2(x*np.sin(c), rho*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c))
        longll = lon * 180.0 / np.pi
	
        x = -bot
        y = -bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho/2.0)
        lattr =np.arcsin(np.cos(c)*np.sin(lat0)  + y*np.sin(c)*np.cos(lat0)/rho ) * float(180.0) / np.pi
        lon = long0  + np.arctan2(x*np.sin(c), rho*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c))
        longtr = lon * 180.0 / np.pi

        return longll,longtr,latll,lattr

    def Cylindrical_Window(self,radius,lat0,long0):
        ''' Return the cylindrical projection of a window centered
        at (lat0,long0)  with a given radius of radius.'''
        
        # Passage en radian
        radi = radius*2*np.pi/(2*1734.4*np.pi)
        lamb0 = long0*np.pi/180.0
        phi0 = lat0*np.pi/180.0

        #Long/lat min (voir wikipedia)
        longll = -radi/np.cos(phi0)+lamb0
        latll = np.arcsin((-radi+np.sin(phi0)/np.cos(phi0))*np.cos(phi0))
        if np.isnan(latll):
          latll = -90*np.pi/180.0
        #Long/lat max (voir wikipedia)
        longtr = radi/np.cos(phi0)+lamb0
        lattr = np.arcsin((radi+np.tan(phi0))*np.cos(phi0))

        # print latll*180/np.pi,lat0,lattr*180/np.pi
        # print longll*180/np.pi,long0,longtr*180/np.pi
        return longll*180/np.pi,longtr*180/np.pi,latll*180/np.pi,lattr*180/np.pi


class WacMap(object):
    ''' Class used to identified the image (or the groupe of images) necessary
    to extract an array around a particular structure.
    4 Cases are possible:
    1 - The desired structure is entirely contained into one image.
    2 - The span in latitude of the image is ok but not longitudes (2 images).
    3 - The span in longitude of the image is ok but not latitudes (2 images).
    4 - Both latitude and longitude are not contained in one image (4 images).

    ONLY THE FIRST CASE IS IMPLEMENTED FOR THE MOMENT

    parameters:
    ppd : Resolution required
    lonm,lonM,latm,latM : Parameterize the window around the structure

    methods:
    Image : Return a BinaryTable Class containing the image.
    
    '''
    def __init__(self,lon_m,lon_M,lat_m,lat_M,ppd):
        self.ppd = ppd
        self.lonm = lon_m
        self.lonM = lon_M
        self.latm = lat_m
        self.latM = lat_M
        # All resolution are not implemented
        assert self.ppd in [4,8,16,32,128]

    def _map_center(self,coord,val):

        ''' Identitify the center of the Image correspond to one coordinate.

        parameters:
        coord : "lat" or "long"
        val : value of the coordinate

        variable:
        res : {Correspond lat center for the image +
        longitude span of the image}'''
        
        if self.ppd in [4,8,16,32,64]:
            res = {'lat' : 0, 'long' : 360}
        elif self.ppd in [128]:
            res = {'lat' : 45, 'long' : 90}
        else:
            print 'Not implemented'
            raise Exception
            
        return (val//res[self.ppd][coord]+1)*res[self.ppd][coord]-res[self.ppd][coord]/2.0

    def _Define_Case(self):
        ''' Identify case:
        1 - The desired structure is entirely contained into one image.
        2 - The span in latitude of the image is ok but not longitudes (2 images).
        3 - The span in longitude of the image is ok but not latitudes (2 images).
        4 - Both latitude and longitude are not contained in one image (4 images).
        
        '''
        
        lonBool = self._map_center('long',self.lonM) != self._map_center('long',self.lonm)
        latBool = self._map_center('lat',self.latM) != self._map_center('lat',self.latm)

        if not lonBool and not latBool:
            print 'Cas1'
            return self._Cas_1()
        elif lonBool and not latBool:
            print 'Cas2'
            print 'Pas implementer'
            raise NameError
        elif not lonBool and latBool:
            print 'Cas3'
            print 'Pas implementer'
            raise NameError
        else:
            print 'Cas4'
            print 'Pas implementer'
            raise NameError

    def _format_lon(self,lon):
        lonf = self._map_center('long',lon)
        st = str(lonf).split('.')
        loncenter =''.join(("{0:0>3}".format(st[0]),st[1]))
        return loncenter

    def _format_lat(self,lat):
        if lat<0:
            latcenter = '450S'
        else:
            latcenter = '450N'
        return latcenter

    def _Cas_1(self):
        '''1 - The desired structure is entirely contained into one image.'''

        lonc = self._format_lon(self.lonm)
        latc = self._format_lat(self.latm)
        wac = '_'.join(['WAC','GLOBAL','E'+latc+lonc,str(self.ppd)+'P'])
        return BinaryTable(wac)

    def Image(self):
        return self._Define_Case()
        
class LolaMap(object):
    
    def __init__(self,lon_m,lon_M,lat_m,lat_M,ppd,racine):
        self.racine = racine
        self.ppd = ppd
        self.lonm = lon_m
        self.lonM = lon_M
        self.latm = lat_m
        self.latM = lat_M

    def map_center(self,coord,val):
        res = {512 : {'lat' : 45,'long' : 90}}
        return (val//res[self.ppd][coord]+1)*res[self.ppd][coord]-res[self.ppd][coord]/2.0

    def map_border(self,coord,val):
        res = {512 : {'lat' : 45,'long' : 90}}
        c = (val//res[self.ppd][coord]+1)*res[self.ppd][coord]
        return c-res[self.ppd][coord],c

    def Define_Case(self):
        ''' Case 1 : 0 Pas d'overlap, crater contenu ds une carte
        Case 2: 1Overlap au niveau des long
        Case 3:2 OVerlap au niveau des lat
        Case 4 :3 Overlap partout
        Boolean : True si overlap, false sinon'''

        lonBool = self.map_center('long',self.lonM) != self.map_center('long',self.lonm)
        latBool = self.map_center('lat',self.latM) != self.map_center('lat',self.latm)

        if not lonBool and not latBool:
            print 'Cas1'
            return self.Cas_1()
        elif lonBool and not latBool:
            print 'Cas2'
            print 'Pas implementer'
            #sys.exit()
        elif not lonBool and latBool:
            print 'Cas3'
            print 'Pas implementer'
            #sys.exit()
        else:
            print 'Cas4'
            print 'Pas implementer'
            #sys.exit()

    def _format_lon(self,lon):
        lonm,lonM = map(lambda x:"{0:0>3}".format(int(x)),self.map_border('long',lon))
        return lonm,lonM

    def _format_lat(self,lat):
        if lat<0:
            latm,latM = map(lambda x:"{0:0>2}".format(int(np.abs(x)))+'s',self.map_border('lat',lat))
        else:
            latm,latM = map(lambda x:"{0:0>2}".format(int(x))+'n',self.map_border('lat',lat))

        return latm,latM

    def Cas_1(self):
        ''' Ni long ni lat ne croise en bord de la carte
        colle le bon wac sur self.wac '''
        
        lonm,lonM = self._format_lon(self.lonm)
        latm,latM = self._format_lat(self.latm)
        lola = '_'.join(['LDEM',str(self.ppd),latm,latM,lonm,lonM])
        return BinaryTable(lola)

    def Name(self):
        return self.Define_CaseLola()
