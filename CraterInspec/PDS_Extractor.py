# Import library
from __future__ import print_function
import numpy as np
import pandas as pd
import os,sys
from distutils.util import strtobool
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

    racine = '/Users/thorey/Documents/These/Projet/FFC/CraterInspector'

    def __init__(self,fname):
        ''' Parameter
        self.name : name of the file
        self._Category : Identify weither its WAC/LOLA image
        self._Load_Info_LBL : Load corresponding information
        '''

        self.fname = fname
        self.PDS_FILE = os.path.join(BinaryTable.racine,'PDS_FILE')
        self.LOLApath = os.path.join(self.PDS_FILE,'LOLA')
        self.WACpath = os.path.join(self.PDS_FILE,'LROC_WAC')
        self._Category()
        self._maybe_download()
        self._Load_Info_LBL()

        assert self.MAP_PROJECTION_TYPE in ['"SIMPLE','EQUIRECTANGULAR'],\
            "Only cylindrical projection is possible - %s NOT IMPLEMENTED"%(self.MAP_PROJECTION_TYPE)

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
            raise ValueError("%s : This type of image is not recognized. Possible\
                             images are from %s only"% (self.fname, ', '.join(('WAC','LOLA'))))

    def _report(self,blocknr, blocksize, size):

        ''' helper for downloading the file '''

        current = blocknr*blocksize
        sys.stdout.write("\r{0:.2f}%".format(100.0*current/size))

    def _downloadFile(self,url,fname):
            
        ''' Download the file '''
        
        print("The file %s need to be download - Wait\n "%(fname.split('/')[-1]))
        urllib.urlretrieve(url, fname, self._report)
        print("\n The download of the file %s has succeded \n "%(fname.split('/')[-1]))



    def _user_yes_no_query(self,question):
        sys.stdout.write('%s [y/n]\n' % question)
        while True:
            try:
                return strtobool(raw_input().lower())
            except ValueError:
                sys.stdout.write('Please respond with \'y\' or \'n\'.\n')
            
    def _detect_size(self,url):

        '''
        Detect the size of the target and return it 
        '''
        site = urllib.urlopen(url)
        meta = site.info()
        return float(meta.getheaders("Content-Length")[0])/1e6
            
    def _maybe_download(self):
            
        if self.Grid == 'WAC':
            urlpath = 'http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/'
            r = requests.get(urlpath) # List file in the cloud
            images = [elt.split('"')[7].split('.')[0] for elt in r.iter_lines() if len(elt.split('"'))>7]
            if self.fname not in images:
                raise ValueError("%s : Image does not exist\n.\
                                 Possible images are:\n %s"% (self.fname, '\n, '.join(images[2:])))
            elif not os.path.isfile(self.img):
                urlname = os.path.join(urlpath , self.img.split('/')[-1])
                print("The size is ?: %.1f Mo \n\n"%(self._detect_size(urlname)))
                download = self._user_yes_no_query('Do you really want to download %s ?\n\n'%(self.fname))
                if download:
                    self._downloadFile(urlname,self.img)
                else:
                    raise ValueError("You need to download the file somehow")
                

        elif self.Grid == 'LOLA':
            urlpath = 'http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/'
            r = requests.get(urlpath) # List file in this server
            images = [elt.split('"')[7].split('.')[0] for elt in r.iter_lines() if len(elt.split('"'))>7]
            if self.fname not in images:
                raise ValueError("%s : Image does not exist\n.\
                                 Possible images are:\n %s"% (self.fname, '\n, '.join(images[2:])))

            elif (not os.path.isfile(self.img)) and (self.fname in images):
                urlname = os.path.join(urlpath , self.img.split('/')[-1])
                print("The size is ?: %.1f Mo \n\n"%(self._detect_size(urlname)))
                download = self._user_yes_no_query('Do you really want to download %s ?\n\n'%(self.fname))
                if download:
                    self._downloadFile(urlname,self.img)
                else:
                    raise ValueError("You need to download the file somehow")
                
                urlname = os.path.join(urlpath , self.lbl.split('/')[-1])
                self._downloadFile(urlname,self.lbl)

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
            lat =  float(self.CENTER_LATITUDE) - \
                   (line -float(self.LINE_PROJECTION_OFFSET) -1)\
                   / float(self.MAP_RESOLUTION)
            return lat

    def Long_id(self,sample):
        ''' Return the corresponding sample longitude'''
        if self.Grid == 'WAC':
            lon = self.CENTER_LONGITUDE + (sample - self.SAMPLE_PROJECTION_OFFSET -1)\
                  *self.MAP_SCALE*1e-3/(self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0))
            return lon*180/np.pi
        else:
            lon = float(self.CENTER_LONGITUDE) + \
                  (sample -float(self.SAMPLE_PROJECTION_OFFSET) -1)\
                  / float(self.MAP_RESOLUTION)
            return lon

    def _control_sample(self, sample):
        if sample > float(self.SAMPLE_LAST_PIXEL):
            return int(self.SAMPLE_LAST_PIXEL)
        elif sample < float(self.SAMPLE_FIRST_PIXEL):
            return int(self.SAMPLE_FIRST_PIXEL)
        else:
            return sample
            
    def Sample_id(self,lon):
        ''' Return the corresponding longitude sample'''
        if self.Grid == 'WAC':
            sample =  np.rint(float(self.SAMPLE_PROJECTION_OFFSET)+1.0+\
                             (lon*np.pi/180.0-float(self.CENTER_LONGITUDE))*\
                             self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0)\
                              /(self.MAP_SCALE*1e-3))
        else:
            sample =  np.rint(float(self.SAMPLE_PROJECTION_OFFSET) + float(self.MAP_RESOLUTION)\
                              * (lon - float(self.CENTER_LONGITUDE))) + 1
        return self._control_sample(sample)

    def _control_line(self, line):
        if line>float(self.LINE_LAST_PIXEL):
            return int(self.LINE_LAST_PIXEL)
        elif line< float(self.LINE_FIRST_PIXEL):
            return int(self.LINE_FIRST_PIXEL)
        else:
            return line
                
    def Line_id(self,lat):
        ''' Return the corresponding latitude line'''
        if self.Grid == 'WAC':
            line = np.rint(1.0 + self.LINE_PROJECTION_OFFSET-self.A_AXIS_RADIUS*np.pi*lat/(self.MAP_SCALE*1e-3*180))
        else:
            line =  np.rint(float(self.LINE_PROJECTION_OFFSET) - float(self.MAP_RESOLUTION)\
                          * (lat - float(self.CENTER_LATITUDE))) + 1
        return self._control_line(line)

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

    def Extract_Grid(self,longmin,longmax,latmin,latmax):
        ''' Extract part of the image centered defined by a rectangle (longmin,
        longmax,latmin,latmax) - all in degree.

        Return three tables X (longitude),Y(latitude) and Z(values) '''

        sample_min,sample_max = map(int,map(self.Sample_id,[longmin,longmax]))
        line_min,line_max = map(int,map(self.Line_id,[latmax,latmin]))
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

    implemented_res = [4,8,16,32,64,128]
    
    def __init__(self,ppd,lonm,lonM,latm,latM):
        self.ppd = ppd
        self.lonm = lonm
        self.lonM = lonM
        self.latm = latm
        self.latM = latM
        self._Confirm_Resolution(WacMap.implemented_res)
        
    def _Confirm_Resolution(self,implemented_res):
        # All resolution are not implemented
        assert self.ppd in implemented_res, ' Resolution %d ppd not implemented yet\n.\
        Consider using one of the implemented resolutions %s'\
        %(self.ppd, ', '.join([f+' ppd' for f in map(str,implemented_res)]))

    def _map_center(self,coord,val):

        ''' Identitify the center of the Image correspond to one coordinate.

        parameters:
        coord : "lat" or "long"
        val : value of the coordinate

        variable:
        res : {Correspond lat span for the image +
        longitude span of the image}'''

        if self.ppd in [4,8,16,32,64]:
            res = {'lat' : 0, 'long' : 360}
            return res[coord]/2.0
        elif self.ppd in [128]:
            res = {'lat' : 90, 'long' : 90}
            return (val//res[coord]+1)*res[coord]-res[coord]/2.0

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
            print('No overlap - Processing should be quick')
            return self._Cas_1()
        elif lonBool and not latBool:
            print('Longitude overlap - 2 images have to be proceded \n \
                  Processing could take a few seconds')
            return self._Cas_2()
        elif not lonBool and latBool:
            print('Latitude overlap - 2 images have to be proceded \n\
                  Processing could take a few seconds')
            return self._Cas_3()
        else:
            print('Latitude/Longidude overlaps - 4 images have to be proceded \n\
                  Processing could take a few seconds')
            return self._Cas_4()

    def _format_lon(self,lon):
        lonf = self._map_center('long',lon)
        st = str(lonf).split('.')
        loncenter =''.join(("{0:0>3}".format(st[0]),st[1]))
        return loncenter

    def _format_lat(self,lat):
        if self.ppd in [4,8,16,32,64]:
            latcenter = '000N'
        elif self.ppd in [128]:
            if lat<0:
                latcenter = '450S'
            else:
                latcenter = '450N'

        return latcenter

    def _format_name_map(self,lonc,latc):
        '''
        Return the name of the map in the good format
        '''
        return '_'.join(['WAC','GLOBAL','E'+latc+lonc,"{0:0>3}".format(self.ppd)+'P'])
        
    def _Cas_1(self):
        '''1 - The desired structure is entirely contained into one image.'''

        lonc = self._format_lon(self.lonm)
        latc = self._format_lat(self.latm)
        img = self._format_name_map(lonc,latc)
        img_map = BinaryTable(img)

        return img_map.Extract_Grid(self.lonm,self.lonM,self.latm,self.latM)

    def _Cas_2(self):
        '''1 - The span in latitude of the image is ok but
        not longitudes (2 images). The desired structure longitude
        are overlap on two different map .'''

        lonc_left = self._format_lon(self.lonm)
        lonc_right = self._format_lon(self.lonM)
        latc = self._format_lat(self.latm)

        img_name_left = self._format_name_map(lonc_left,latc)
        print(img_name_left)
        img_left = BinaryTable(img_name_left)
        X_left,Y_left,Z_left  = img_left.Extract_Grid(self.lonm,
                                                      float(img_left.EASTERNMOST_LONGITUDE),
                                                      self.latm,
                                                      self.latM)

        img_name_right = self._format_name_map(lonc_right,latc)
        img_right = BinaryTable(img_name_right)
        X_right,Y_right,Z_right  = img_right.Extract_Grid(float(img_right.WESTERNMOST_LONGITUDE),
                                                          self.lonM,
                                                          self.latm,
                                                          self.latM)

        X_new = np.hstack((X_left,X_right))
        Y_new = np.hstack((Y_left,Y_right))
        Z_new = np.hstack((Z_left,Z_right))
        
        return X_new, Y_new, Z_new

    def _Cas_3(self):
        '''1 - The span in longitude of the image is ok but
        not latitudes (2 images). The desired structure latitude
        are overlaped on two different maps .'''

        lonc = self._format_lon(self.lonm)
        latc_top = self._format_lat(self.latM)
        latc_bot = self._format_lat(self.latm)

        img_name_top = self._format_name_map(lonc,latc_top)
        print(img_name_top)
        img_top = BinaryTable(img_name_top)
        print(self.lonm,self.lonM,float(img_top.MINIMUM_LATITUDE),self.latM)
        X_top,Y_top,Z_top  = img_top.Extract_Grid(self.lonm,
                                                  self.lonM,
                                                  float(img_top.MINIMUM_LATITUDE),
                                                  self.latM)

        img_name_bottom = self._format_name_map(lonc,latc_bot)
        print(img_name_bottom)
        img_bottom = BinaryTable(img_name_bottom)
        X_bottom,Y_bottom,Z_bottom  = img_bottom.Extract_Grid(self.lonm,
                                                              self.lonM,
                                                              self.latm,
                                                              float(img_bottom.MAXIMUM_LATITUDE))

        X_new = np.vstack((X_top,X_bottom))
        Y_new = np.vstack((Y_top,Y_bottom))
        Z_new = np.vstack((Z_top,Z_bottom))
        
        return X_new, Y_new, Z_new

    def _Cas_4(self):
        '''1 - Neither the span in longitude, nor the span in
        latitudes is ok. Required the ensemble of 4 images'''

        lonc_left = self._format_lon(self.lonm)
        lonc_right = self._format_lon(self.lonM)
        latc_top = self._format_lat(self.latM)
        latc_bot = self._format_lat(self.latm)

        img_name_00 = self._format_name_map(lonc_left,latc_top)
        img_00 = BinaryTable(img_name_00)
        X_00,Y_00,Z_00  = img_00.Extract_Grid(self.lonm,
                                              float(img_00.EASTERNMOST_LONGITUDE),
                                              float(img_00.MINIMUM_LATITUDE),
                                              self.latM)

        img_name_01 = self._format_name_map(lonc_right,latc_top)
        img_01 = BinaryTable(img_name_01)
        X_01,Y_01,Z_01  = img_01.Extract_Grid(float(img_01.WESTERNMOST_LONGITUDE),
                                              self.lonM,
                                              float(img_01.MINIMUM_LATITUDE),
                                              self.latM)

        img_name_10 = self._format_name_map(lonc_left,latc_bot)
        img_10 = BinaryTable(img_name_10)
        X_10,Y_10,Z_10  = img_10.Extract_Grid(self.lonm,
                                              float(img_10.EASTERNMOST_LONGITUDE),
                                              self.latm,
                                              float(img_10.MAXIMUM_LATITUDE))

        img_name_11 = self._format_name_map(lonc_right,latc_bot)
        img_11 = BinaryTable(img_name_11)
        X_11,Y_11,Z_11  = img_11.Extract_Grid(float(img_11.WESTERNMOST_LONGITUDE),
                                              self.lonM,
                                              self.latm,
                                              float(img_11.MAXIMUM_LATITUDE))

        X_new_top = np.hstack((X_00,X_01))
        X_new_bot = np.hstack((X_10,X_11))
        X_new = np.vstack((X_new_top,X_new_bot))

        Y_new_top = np.hstack((Y_00,Y_01))
        Y_new_bot = np.hstack((Y_10,Y_11))
        Y_new = np.vstack((Y_new_top,Y_new_bot))

        Z_new_top = np.hstack((Z_00,Z_01))
        Z_new_bot = np.hstack((Z_10,Z_11))
        Z_new = np.vstack((Z_new_top,Z_new_bot))
        
        return X_new, Y_new, Z_new
        
        
    def Image(self):
        ''' Return three array X,Y,Z corresponding tp
        X : longitudes
        Y : latitudes
        Z : values

        '''
        return self._Define_Case()

        
class LolaMap(WacMap):

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
    Image : Return X,Y,Z values for the window.

    '''

    implemented_res = [4,16,64,128,512]

    def __init__(self,ppd,lonm,lonM,latm,latM):
        self.ppd = ppd
        self.lonm = lonm
        self.lonM = lonM
        self.latm = latm
        self.latM = latM
        self._Confirm_Resolution(LolaMap.implemented_res)
        
    def _map_center(self,coord,val):

        ''' Identitify the center of the Image correspond to one coordinate.

        parameters:
        coord : "lat" or "long"
        val : value of the coordinate

        variable:
        res : {Correspond lat center for the image +
        longitude span of the image}'''

        if self.ppd in [4,16,64,128]:
            res = {'lat' : 0, 'long' : 360}
            return res[coord]/2.0
        elif self.ppd in [512]:
            res = {'lat' : 45, 'long' : 90}
            c = (val//res[coord]+1)*res[coord]            
            return c-res[coord], c
        
    def _format_lon(self,lon):
        if self.ppd in [4,16,64,128]:
            return None
        else:
            return  map(lambda x:"{0:0>3}".format(int(x)),self._map_center('long',lon))

    def _format_lat(self,lat):
        if self.ppd in [4,16,64,128]:
            return None
        else:
            if lat<0:
                return map(lambda x:"{0:0>2}"\
                           .format(int(np.abs(x)))+'S',self._map_center('lat',lat))
            else:
                return map(lambda x:"{0:0>2}"\
                           .format(int(x))+'N',self._map_center('lat',lat))

    def _format_name_map(self,lon,lat):
        '''
        Return the name of the map in the good format
        '''

        if self.ppd in [4,16,64,128]:
            lolaname = '_'.join(['LDEM',str(self.ppd)])
        elif self.ppd in [512]:
            lolaname = '_'.join(['LDEM',str(self.ppd),lat[0],lat[1],lon[0],lon[1]])
        return lolaname
