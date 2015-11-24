# Library import
from PDS_Extrator import *
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import os,sys
from palettable.colorbrewer.diverging import RdBu_9_r,BrBG_10_r


class Structure(object):
            
    def __init__(self,ide,idx,racine,structure):
        '''n pour le designer par son nom et i pour le designer par son
        index,structure : dome ou FFC '''

        self.structure = structure
        self.racine = racine
        self.ppdlola = 512
        self.ppdwac = 128
        inde = {'n':'Name','i':'Index'}
        
        if structure == 'Dome':
            self.structures = pd.read_csv(os.path.join(racine,'Data','Data_Dome.csv'))
        elif structure == 'Crater':
            self.structures = pd.read_csv(os.path.join(racine,'Data','Data_Crater.csv'))
        else:
            raise Exception
            
        df = self.structures[self.structures[inde[ide]] == idx]
        if len(df) == 0:
            print 'Correpond a aucun %s'%(structure)
            raise Exception
        [setattr(self,f,float(df[f])) for f in df.columns if f not in ['Name']]
        if self.Long <0.0:
            self.Long = 360+self.Long
        if structure == 'Dome':
            self.Radius = self.D/(2*1000)
            self.Diameter = 2*self.Radius
            self.Name = df.Name.iloc[0]
        else:
            self.Radius = self.Diameter/2.0
            self.Name = df.Name.iloc[0]
        
        self.Taille_Window = 0.8*self.Diameter

    def Crater_Data(self):
        Racine = os.path.join(self.racine,'Data')
        data = Data(64,Racine,'_2')
        df = pd.DataFrame(np.hstack((data.Name,data.Index,data.Lat,data.Long,data.Diameter,data.Type))
                          ,columns = ['Name','Index','Lat','Long','Diameter','Type'])
        for key in ['Lat','Long','Diameter','Type']:
            df[key] =map(float,df[key])
        return df
            
    def load_lola(self):
        lon_m,lon_M,lat_m,lat_M = self.Cylindrical_Window(self.Taille_Window,self.Lat,self.Long)
        maplola = LolaMap(lon_m,lon_M,lat_m,lat_M,self.ppdlola,self.racine)
        return maplola.Name()

    def load_wac(self):
        lon_m,lon_M,lat_m,lat_M = self.Cylindrical_Window(self.Taille_Window,self.Lat,self.Long)
        waclola = WacMap(lon_m,lon_M,lat_m,lat_M,self.ppdwac,self.racine)
        return waclola.Name()
        
            
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

        return longll*180/np.pi,longtr*180/np.pi,latll*180/np.pi,lattr*180/np.pi

    def plot_lola(self,save):

        fig = plt.figure(figsize=(24,14))
        ax1 = fig.add_subplot(111)
        ax1.set_rasterization_zorder(1)
        lon_m,lon_M,lat_m,lat_M = self.wac.Lambert_Window(self.Taille_Window,self.Lat,self.Long)
        X,Y,Z = self.wac.Extract_Grid(self.Taille_Window,self.Lat,self.Long)

        m = Basemap(llcrnrlon =lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i',projection='laea',rsphere = 1734400, lat_0 = self.Lat,lon_0 = self.Long)
        X,Y = m(X,Y)
        m.pcolormesh(X,Y,Z,cmap = cm.gray ,ax  = ax1,zorder =-1)
        Xl,Yl,Zl = self.lola.Extract_Grid(self.Taille_Window,self.Lat,self.Long)
        Xl,Yl = m(Xl,Yl)

        m.contourf(Xl,Yl,Zl,100,cmap='gist_earth', alpha = 0.6 , zorder=-1)
        #cb = m.colorbar(CS3,"right", size="5%", pad="2%")
        #cb.set_label('Topography', size = 24)
        xc,yc = m(self.Long,self.Lat)
        ax1.scatter(xc,yc,s=200,marker ='v',zorder =2)

        lol,loM,lam,laM = self.wac.Lambert_Window(0.6*self.Diameter,self.Lat,self.Long)
        m.drawmapscale(loM,lam, self.Long,self.Lat,10,
                       barstyle='fancy', units='km',
                       fontsize=24, yoffset=None,
                       labelstyle='simple',
                       fontcolor='k',
                       fillcolor1='w',
                       fillcolor2='k', ax=ax1,
                       format='%d',
                       zorder=-1)
        ax1.set_title('Crater %s, %d km in diameter'%(self.Name,self.Diameter),size = 42)

        path = os.path.join(self.racine,'Data','Image','Crater_'+str(self.Name)+'.png')
        if save == True:
            fig.savefig(path,rasterized=True, dpi=200,bbox_inches='tight',pad_inches=0.1)

    def Deg(self,radius):
        return radius*360/(2*np.pi*1734.4)
        
