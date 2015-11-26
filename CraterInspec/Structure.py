# To be able to pass *(tuple) to function
from __future__ import print_function
# Library import
import os,sys


# Load specific library
from PDS_Extractor import *
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


class Structure(object):
    ''' A class able to gather information on Topography and Image about
    a specified area around a geological unit (Crater or dome).

    parameters:

    - structure : Name of the structure. Either "Crater" or "Dome".
    - ide : This class allow to defined the unit by its name "n" or its
    index "i".
    - idx : Corresponding name or index

    attributes:

    - structure : Type of the geological unit (Crater/Dome)
    - racine : path for project
    - ppdlola : resolution of the Lola image
    - ppdwac : resolution of the Wac image
    - Name/Diameter/Radius/Lat/Long : property of the geological unit
    - Taille_Window : Size of the window to consider around the unit

    methods:
    
    - Lambert_Window :  This method is used to determine the bottom-left
    and upper-right coordinates for a square Lambert Azimuthal
    equal area projection centered at (lat0,long0 with a size radius).

    - Cylindrical_Window : This method is used to determine the bottom-left
    and upper-right coordinates for a cylindrical projection
    centered at (lat0,long0 with a size radius). See
    Wikipedia

    - Get_Arrays : Return X,Y,Z of the region for the img asked.
    img is either 'Lola' for the topography or
    'wac' for WAC image.

    - Lola_Image, Wac_Image, Overlay : Return the corresponding
    plot. A blue triangle is added as well as a scale for completnesss
    of the plot.
    
    Example:
    
    For instance, say we want to image the crater Copernicus. We can the
    define an instance of the class
    
    C = Structure('n','Copernicus','Crater')
    C.Overlay(True)
    
    '''
    
    def __init__(self,ide,idx,structure):
        '''
        - structure : Name of the structure. Either "Crater" or "Dome".
        - ide : This class allow to defined the unit by its name "n" or its
        index "i".
        - idx : Corresponding name or index
        '''

        self.structure = structure
        self.racine = BinaryTable.racine
        self.ppdlola = 512
        self.ppdwac = 128

        if structure == 'Dome':
            self.structures = pd.read_csv(os.path.join(self.racine,'Data',
            'Data_Dome.csv'))
        elif structure == 'Crater':
            self.structures = pd.read_csv(os.path.join(self.racine,'Data',
            'Data_Crater.csv'))
        else:
            raise ValueError("Structure %s is not recognized. Possible\n\
                             values are: %s"% (structure, ', '.join(['Dome','Crater'])))

        inde = {'n':'Name','i':'Index'}
        df = self.structures[self.structures[inde[ide]] == idx]
        if len(df) == 0:
            raise ValueError("The tuple (%s,%s) does not correspond\n \
                             to any structure in the dataset. "%(ide,idx))
            
        [setattr(self,f,float(df[f])) for f in df.columns if f not in ['Name']]

        assert self.Long>0.0, 'Longitude has to span 0-360 !!!'
        self.Name = df.Name.iloc[0]
        self.Change_window(0.8*self.Diameter)
        

    def Change_window(self,Taille_Window):
        self.Taille_Window = Taille_Window
        self.window = self.Cylindrical_Window(self.Taille_Window,self.Lat,self.Long)
        

    def _kp_func(self,lat,lon,lat0,long0):
        kp = float(1.0) + np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-long0)
        kp = np.sqrt(float(2)/kp)
        return kp

    def Lambert_Window(self,radius,lat0,long0):
        '''
        This method is used to determine the bottom-left
        and upper-right coordinates for a square Lambert Azimuthal
        equal area projection centered at (lat0,long0 with a size radius).
        '''
        
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
        '''
        This method is used to determine the bottom-left
        and upper-right coordinates for a cylindrical projection
        centered at (lat0,long0 with a size radius). See
        Wikipedia
        '''
        # Radian conversion
        radi = radius*2*np.pi/(2*1734.4*np.pi)
        lamb0 = long0*np.pi/180.0
        phi0 = lat0*np.pi/180.0

        #Long/lat min (see wikipedia)
        longll = -radi/np.cos(phi0)+lamb0
        latll = np.arcsin((-radi+np.sin(phi0)/np.cos(phi0))*np.cos(phi0))
        if np.isnan(latll):
            latll = -90*np.pi/180.0

        #Long/lat max (see wikipedia)
        longtr = radi/np.cos(phi0)+lamb0
        lattr = np.arcsin((radi+np.tan(phi0))*np.cos(phi0))

        return longll*180/np.pi,longtr*180/np.pi,latll*180/np.pi,lattr*180/np.pi

    def _Add_Scale(self,m,ax1):
        ''' Add scale to the map instance '''
        
        lol,loM,lam,laM = self.Lambert_Window(0.6*self.Taille_Window,self.Lat,self.Long)
        m.drawmapscale(loM,lam, self.Long,self.Lat,10,
                       barstyle='fancy', units='km',
                       fontsize=24, yoffset=None,
                       labelstyle='simple',
                       fontcolor='k',
                       fillcolor1='w',
                       fillcolor2='k', ax=ax1,
                       format='%d',
                       zorder=2)

    def Get_Arrays(self,type_img):
        '''
        Return X,Y,Z of the region for the img asked.
        img is either 'Lola' for the topography or
        'wac' for WAC image.
        '''
            
        if type_img == 'Lola':
            return LolaMap(self.ppdlola,*self.window).Image()
        elif type_img == 'Wac':
            return WacMap(self.ppdwac,*self.window).Image()
        else:
            raise ValueError('The img type has to be either "Lola" or "Wac"')

    def _format_coordinate_cartopy(self,ax):
        '''
        Format the cartopy plot to show lat/long properly
        '''
        lonm,lonM,latm,latM = self.window
        xlocs = np.linspace(lonm,lonM,5)
        ylocs = np.linspace(latm,latM,5)
        ax.gridlines(xlocs = xlocs , ylocs = ylocs, crs = ccrs.PlateCarree())
        ax.set_xticks(xlocs, crs=ccrs.PlateCarree())
        ax.set_yticks(ylocs, crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(number_format='.1f',
                                           degree_symbol='')
        lat_formatter = LatitudeFormatter(number_format='.1f',
                                          degree_symbol='')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize = 18)

    def Get_Profile(self, img_type, coordinate, num_points):
        '''
        Extract the profiles from (lat1,lon1) to (lat2,lon2)

        parameter:

        img_type : Type of the image
        coordinate : (lon0,lon1,lat0,lat1) Be carefull, longitude has to be in between
        0-360 ! 
        num_points : Number of points in the interpolation
        '''

        lon0, lon1, lat0, lat1 = coordinate
        X, Y, Z = self.Get_Arrays(img_type)
        x0, y0 = np.argmin(np.abs(X[0,:]-lon0)),np.argmin(np.abs(Y[:,0]-lat0))
        x1, y1 = np.argmin(np.abs(X[0,:]-lon1)),np.argmin(np.abs(Y[:,0]-lat1))
        x, y = np.linspace(x0,x1, num_points), np.linspace(y0, y1, num_points)
        zi = scipy.ndimage.map_coordinates(Z, np.vstack((x,y)))
        
        return  zi


    def Draw_Profile(self, coordinates, img_type = 'Lola', num_points = 100,
                     save = False, name = 'BaseProfile.png'):

        '''
        Draw profile between a point (lon0,lat0) and (lon1,lat1).

        parameters:
        coordinates : a tuple which itself contain tupple of coordinates.
        The format is (lon0,lon1,lat0,lat1). For instance, self.window is
        an example.
        img_type : Type of the image to do the profile
        num_points : Number of points for the interpolation
        save : Save or not the figure
        name : Name of the figure.

        Example
        C2 = Structure('n','M13','Dome')
        midlon = (C2.window[0]+C2.window[1])/2.0
        midlat = (C2.window[2]+C2.window[3])/2.0
        profile1 = (midlon,midlon,C2.window[2],C2.window[3]) #Vertical profile
        profile2 = (C2.window[0],C2.window[1],midlat,midlat) #Horizontal profile
        C2.Draw_Profile((profile1,profile2,C2.window))
        
        '''
        
        fig = plt.figure(figsize=(24,len(coordinates)*5))
        gs = gridspec.GridSpec(len(coordinates), 3)
        
        if len(coordinates) == 4:
            assert type(coordinates[0])==tuple,\
                "If only one tupple is given,\n\
                the correct syntax is (tuple,) !! Not (tuple) ;)"
            
        for i,coordinate in enumerate(coordinates):

            ax1 = plt.subplot(gs[i, 0], projection=ccrs.LambertCylindrical())
            ax2 = plt.subplot(gs[i,1:])

            # Image unit
            X, Y , Z = self.Get_Arrays(img_type)
            ax1.pcolormesh(X, Y, Z,
                           transform=ccrs.PlateCarree(),
                           cmap='spectral')            
            lon1,lon0,lat1,lat0 = coordinate
            ax1.plot([lon1,lon0],[lat1,lat0], 'ro-',transform = ccrs.PlateCarree())
            self._format_coordinate_cartopy(ax1)

            # Profile
            print(coordinate)
            z_interpolated = self.Get_Profile(img_type,coordinate,num_points)
            ax2.plot(z_interpolated[::-1])
            ax2.set_ylabel('Topographic profile (m)',fontsize = 24)
            ax2.tick_params(labelsize = 18)
            
            path = os.path.join(self.racine,'Image',name)
            if save == True:
                fig.savefig(path,rasterized=True, dpi=200,bbox_inches='tight',pad_inches=0.1)

            
    def Lola_Image(self, save = False ,name = 'BaseLola.png'):
        '''
        Return the lola image corresponding to the window.
        A blue triangle is added as well as a scale for completnesss
        of the plot.

        This method is encouraged to be modified according to specific need.
        '''
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)

        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)

        Xl,Yl,Zl = self.Get_Arrays('Lola')
        Xl,Yl = m(Xl,Yl)

        m.pcolormesh(Xl,Yl,Zl,cmap = 'gist_earth' , alpha = .5, ax  = ax1,zorder = 1)
        m.contour(Xl,Yl,Zl,20, colors = 'black', alpha = 1.0 , zorder=2)

        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=200,marker ='v',zorder =2)

        self._Add_Scale(m,ax1)
        ax1.set_title('Crater %s, %d km in diameter'%(self.Name,self.Diameter),size = 42)

        path = os.path.join(self.racine,'Image',name)
        if save == True:
            fig.savefig(path,rasterized=True, dpi=200,bbox_inches='tight',pad_inches=0.1)

    def Wac_Image(self, save = False, name = 'BaseWac.png'):
        '''
        Return the Wac image corresponding to the window.
        A blue triangle is added as well as a scale for completnesss
        of the plot.

        This method is encouraged to be modified according to specific need.
        '''
        
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)

        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)

        Xw,Yw,Zw = self.Get_Arrays('Wac')
        Xw,Yw = m(Xw,Yw)
        m.pcolormesh(Xw,Yw,Zw,cmap = cm.gray ,ax  = ax1,zorder = 1)

        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=200,marker ='v',zorder =2)

        self._Add_Scale(m,ax1)
        ax1.set_title('Crater %s, %d km in diameter'%(self.Name,self.Diameter),size = 42)

        path = os.path.join(self.racine,'Image',name)
        if save == True:
            fig.savefig(path, dpi=200 , bbox_inches='tight', pad_inches=0.1)

    def Overlay(self, save = False,name = 'BaseOverlay.png'):

        '''
        Return an ovelay of Lola image over the Wac image
        corresponding to the window.
        A blue triangle is added as well as a scale for completnesss
        of the plot.

        This method is encouraged to be modified according to specific need.
        '''
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(111)

        lon_m,lon_M,lat_m,lat_M = self.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)

        Xw,Yw,Zw = self.Get_Arrays('Wac')
        Xw,Yw = m(Xw,Yw)
        m.pcolormesh(Xw,Yw,Zw,cmap = cm.gray ,ax  = ax1,zorder = 1)

        Xl,Yl,Zl = self.Get_Arrays('Lola')
        Xl,Yl = m(Xl,Yl)
        m.contourf(Xl,Yl,Zl,100,cmap='gist_earth', alpha = 0.4 , zorder=2 , antialiased=True)

        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=200,marker ='v',zorder =2)

        self._Add_Scale(m,ax1)
        ax1.set_title('Crater %s, %d km in diameter'%(self.Name,self.Diameter),size = 42)

        path = os.path.join(self.racine,'Image',name)
        if save == True:
            fig.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0.1)

    def _Deg(self,radius):
        return radius*360/(2*np.pi*1734.4)
