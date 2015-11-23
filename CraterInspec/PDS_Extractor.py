
import numpy as np
import pickle
import pandas as pd
import os,sys
from pvl import load as load_label

        
class BinaryTable(object):

    def __init__(self,file_data):
        ''' Parameter
        self.img : nom du fichier
        self.lbl : '''

        self.name = file_data
        self._Category()
        self._Load_Info_LBL()

    def _Category(self):
        if self.name.split('/')[-1].split('_')[0] == 'WAC':
            self.Grid = 'WAC'
            self.img = self.name+'.IMG'
            self.lbl = ''
        elif self.name.split('/')[-1].split('_')[0] == 'ldem':
            self.Grid = 'LOLA'
            self.img = self.name+'.img'
            self.lbl = self.name+'.lbl'
        else:
            self.Grid = 'GRAIL'
            self.img =  self.name + '.dat'

        
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
        if self.Grid == 'WAC':
            lat = ((1 + self.LINE_PROJECTION_OFFSET- line)*self.MAP_SCALE*1e-3/self.A_AXIS_RADIUS)
            return lat*180/np.pi
        else:
            lat =  float(self.CENTER_LATITUDE) - (line -float(self.LINE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
            return lat
    
    def Long_id(self,sample):
        if self.Grid == 'WAC':
            lon = self.CENTER_LONGITUDE + (sample - self.SAMPLE_PROJECTION_OFFSET -1)*self.MAP_SCALE*1e-3/(self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0))
            return lon*180/np.pi
        else:
            lon = float(self.CENTER_LONGITUDE) + (sample -float(self.SAMPLE_PROJECTION_OFFSET) -1)/ float(self.MAP_RESOLUTION)
            return lon
    
    def Sample_id(self,lon):
        if self.Grid == 'WAC':
            return np.rint(float(self.SAMPLE_PROJECTION_OFFSET)+1.0+\
                (lon*np.pi/180.0-float(self.CENTER_LONGITUDE))*self.A_AXIS_RADIUS*np.cos(self.CENTER_LATITUDE*np.pi/180.0)/(self.MAP_SCALE*1e-3))
        else:
            return np.rint(float(self.SAMPLE_PROJECTION_OFFSET) + float(self.MAP_RESOLUTION)\
                           * (lon - float(self.CENTER_LONGITUDE))) + 1
    def Line_id(self,lat):
        if self.Grid == 'WAC':
            return np.rint(1.0 + self.LINE_PROJECTION_OFFSET-self.A_AXIS_RADIUS*np.pi*lat/(self.MAP_SCALE*1e-3*180))
        else:
            return np.rint(float(self.LINE_PROJECTION_OFFSET) - float(self.MAP_RESOLUTION)\
                           * (lat - float(self.CENTER_LATITUDE))) + 1

    def Array(self,size_chunk,start,bytesize):
        ''' Read part of the binary file
        size_chunk : taille du truc a lire
        start : debut
        bytesize : taille en bye int16 = 2'''

        with open(self.img,'rb') as f1:
            f1.seek(self.start_byte+start*self.bytesize)
            data = f1.read(size_chunk*self.bytesize)
            Z = np.fromstring(data, dtype=self.dtype, count = size_chunk)
            return Z

    def Extract_All(self):
        
        longmin,longmax,latmin,latmax = 0.01,359.9,-89.0,89.0
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
        
    def Extract_Grid(self,radius,lat0,long0):
        
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
        self._Load_Info_LBL()
        print 
        return (int(self.WESTERNMOST_LONGITUDE),
                int(self.EASTERNMOST_LONGITUDE),
                int(self.MINIMUM_LATITUDE),
                int(self.MAXIMUM_LATITUDE))

    def kp_func(self,lat,lon,lat0,long0):
        kp = float(1.0) + np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-long0)
        kp = np.sqrt(float(2)/kp)
        return kp
        
    def Lambert_Window(self,radius,lat0,long0):

        radius = radius*360.0/(np.pi*2*1734.4)
        radius = radius*np.pi / 180.0
        lat0 = lat0*np.pi/180.0
        long0 = long0*np.pi/180.0

        bot = self.kp_func(lat0-radius,long0,lat0,long0)
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
    
    def __init__(self,lon_m,lon_M,lat_m,lat_M,ppd,racine):
        self.racine = racine
        self.ppd = ppd
        self.lonm = lon_m
        self.lonM = lon_M
        self.latm = lat_m
        self.latM = lat_M

    def map_center(self,coord,val):
        res = {128 : {'lat' : 45,'long' : 90},
               8   : {'lat' : 0, 'long' : 180}}
        return (val//res[self.ppd][coord]+1)*res[self.ppd][coord]-res[self.ppd][coord]/2.0

    def Define_CaseLola(self):
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
        lonf = self.map_center('long',lon)
        st = str(lonf).split('.')
        loncenter =''.join(("{0:0>3}".format(st[0]),st[1]))
        return loncenter

    def _format_lat(self,lat):
        if lat<0:
            latcenter = '450S'
        else:
            latcenter = '450N'
        return latcenter

    def Cas_1(self):
        ''' Ni long ni lat ne croise en bord de la carte
        colle le bon wac sur self.wac '''

        print 'hello'
        lonc = self._format_lon(self.lonm)
        latc = self._format_lat(self.latm)
        wac = '_'.join(['WAC','GLOBAL','E'+latc+lonc,str(self.ppd)+'P'])
        f = os.path.join(self.racine,'PDS_FILE','LROC_WAC',wac)
        return BinaryTable(f)

    def Name(self):
        return self.Define_CaseLola()
        
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

    def Define_CaseLola(self):
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
        lola = '_'.join(['ldem',str(self.ppd),latm,latM,lonm,lonM])
        f = os.path.join(self.racine,'PDS_FILE','Lola',lola)
        return BinaryTable(f)

    def Name(self):
        return self.Define_CaseLola()
