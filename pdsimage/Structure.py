# To be able to pass *(tuple) to function
from __future__ import print_function
# Library import
import os
import sys

# Load specific library
from PDS_Extractor import *
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.gridspec as gridspec


class Area(object):

    ''' A class which gather information on a specific location

    It is particularly useful to study a particular location at the
    surface of the Moon. For the moment, it can gather information
    about the topography (from LRO LOLA experiment) and texture (from
    the LRO WAC experiment). More information about the Lunar Reconnaissance
    Orbiter mission (LRO) can be found `here`_

    Args:
        lon0 (float): Center longitude of the region of interest.
        lat0 (float): Center latitude of the region of interest.
        size (float): Radius of the region of interest.
        path_pdsfiles (Optional[str]): Path where the pds files are stored.
            Defaults, the path is set to the folder ``PDS_FILES`` next to
            the module files where the library is install.

            See ``defaut_pdsfile`` variable of the class.

    Attributes:
        path_pdsfiles: Path where the pds_files are stored.
        lon0 (float): Longitude of the region of interest.
        lat0 (float): Latitude of the region of interest.
        ppdlola (int): Resolution for the topography
        ppdwac (int): Resolution for the WAC image
        size_window (float): Radius of the region of interest (km)
        window (float,float,float,float): ``(longll, longtr, latll, lattr)``
            with:

            - ``longll`` the longitude of the lower left corner
            - ``longtr`` the longitude of the top right corner
            - ``latll`` the latitude of the lower left corner
            - ``lattr`` the latitude of the top right corner

    Note:
        It is important to respect the structure of the PDS_FILES folder. It
        should contain 2 subfolder called ``LOLA`` and ``LROC_WAC`` where the
        corresponding images should be download.

        The abreviations correspond to:

        - **LRO** Lunar Reconnaissance Orbiter
        - **LOLA** Lunar Orbiter Laser Altimeter
        - **LROC** Lunar Reconnaissance Orbiter Camera
        - **WAC** Wide Angle Camera

    Example:
        For instance, say we want to get an overlay, the topography
        drawn over an wide angle camera image, of a region centred
        around 10 East and 10 North of about 20 km

        >>> C = Area(10,10,20)
        >>> C.overlay()

    .. _here:
        http://www.nasa.gov/mission_pages/LRO/spacecraft/#.VpOMDpMrKL4

    '''

    defaut_pdsfile = os.path.join(
        '/'.join(os.path.abspath(__file__).split('/')[:-1]), 'PDS_FILES')

    def __init__(self, lon0, lat0, Size, path_pdsfile=defaut_pdsfile):

        self.path_pdsfiles = path_pdsfile
        self.lat0 = lat0
        self.lon0 = lon0
        self.ppdlola = 512
        self.ppdwac = 128
        assert (self.lon0 > 0.0) and (
            self.lon0 < 360.0), 'Longitude has to span 0-360 !!!'
        self.change_window(Size)

    def change_window(self, size_window):
        ''' Change the region of interest

        Args:
            size_window (float): Radius of the region of interest (km)

        Notes:
            Change the attributes ``size_window`` and ``window`` to
            correspond to the new region of interest.

        '''
        self.size_window = size_window
        self.window = self.lambert_window(
            self.size_window, self.lat0, self.lon0)

    def _kp_func(self, lat, lon, lat0, long0):
        kp = float(1.0) + np.sin(lat0) * np.sin(lat) + \
            np.cos(lat0) * np.cos(lat) * np.cos(lon - long0)
        kp = np.sqrt(float(2) / kp)
        return kp

    def lambert_window(self, radius, lat0, long0):
        ''' Square Lambert Azimuthal equal area projection of
        a window centered at (lat0, long0) with a given radius (km).

        Args:
            radius(float): Radius of the window (km).
            lat0(float): Latitude at the center (degree).
            long0(float): Longitude at the center (degree).

        Returns:
            A tuple ``(longll, longtr, latll, lattr)`` with ``longll``
            the longitude of the lower left corner, ``longtr`` the
            longitude of the top right corner, ``latll`` the latitude
            of the lower left corner and ``lattr`` the latitude of the
            top right corner.

        Note:
            All return coordinates are in degree

        '''

        radius = radius * 360.0 / (np.pi * 2 * 1734.4)
        radius = radius * np.pi / 180.0
        lat0 = lat0 * np.pi / 180.0
        long0 = long0 * np.pi / 180.0

        bot = self._kp_func(lat0 - radius, long0, lat0, long0)
        bot = bot * (np.cos(lat0) * np.sin(lat0 - radius) -
                     np.sin(lat0) * np.cos(lat0 - radius))
        x = bot
        y = bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho / float(2.0))
        latll = np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c)
                          * np.cos(lat0) / rho) * float(180.0) / np.pi
        lon = long0 + np.arctan2(x * np.sin(c), rho * np.cos(lat0)
                                 * np.cos(c) - y * np.sin(lat0) * np.sin(c))
        longll = lon * 180.0 / np.pi

        x = -bot
        y = -bot
        rho = np.sqrt(x**2 + y**2)
        c = 2.0 * np.arcsin(rho / 2.0)
        lattr = np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c)
                          * np.cos(lat0) / rho) * float(180.0) / np.pi
        lon = long0 + np.arctan2(x * np.sin(c), rho * np.cos(lat0)
                                 * np.cos(c) - y * np.sin(lat0) * np.sin(c))
        longtr = lon * 180.0 / np.pi

        return longll, longtr, latll, lattr

    def cylindrical_window(self, radius, lat0, long0):
        ''' Cylindrical projection of a window centered
        at (lat0, long0) with a given radius (km).

        Args:
            radius(float): Radius of the window (km).
            lat0(float): Latitude at the center (degree).
            long0(float): Longitude at the center (degree).

        Returns:
            A tuple ``(longll, longtr, latll, lattr)` with ``longll``
            the longitude of the lower left corner, ``longtr`` the
            longitude of the top right corner, ``latll`` the latitude
            of the lower left corner and ``lattr`` the latitude of the
            top right corner.

        Note:
            All return coordinates are in degree
        '''

        # Passage en radian
        radi = radius * 2 * np.pi / (2 * 1734.4 * np.pi)
        lamb0 = long0 * np.pi / 180.0
        phi0 = lat0 * np.pi / 180.0

        # Long/lat min (voir wikipedia)
        longll = -radi / np.cos(phi0) + lamb0
        latll = np.arcsin((-radi + np.sin(phi0) / np.cos(phi0)) * np.cos(phi0))
        if np.isnan(latll):
            latll = -90 * np.pi / 180.0
        # Long/lat max (voir wikipedia)
        longtr = radi / np.cos(phi0) + lamb0
        lattr = np.arcsin((radi + np.tan(phi0)) * np.cos(phi0))

        return longll * 180 / np.pi, longtr * 180 / np.pi, latll * 180 / np.pi, lattr * 180 / np.pi

    def _add_scale(self, m, ax1):
        ''' Add scale to the map instance '''

        lol, loM, lam, laM = self.lambert_window(
            0.6 * self.size_window, self.lat0, self.lon0)
        m.drawmapscale(loM, lam, self.lon0, self.lat0, 10,
                       barstyle='fancy', units='km',
                       fontsize=24, yoffset=None,
                       labelstyle='simple',
                       fontcolor='k',
                       fillcolor1='w',
                       fillcolor2='k', ax=ax1,
                       format='%d',
                       zorder=2)

    def _add_colorbar(self, m, CS, ax, name):
        ''' Add colorbar to the map instance '''

        cb = m.colorbar(CS, "right", size="5%", pad="2%")
        cb.set_label(name, size=34)
        cb.ax.tick_params(labelsize=18)

    def get_arrays(self, type_img):
        ''' Return arrays the region of interest

        Args:
            type_img (str): Either lola or wac.

        Returns:
            A tupple of three arrays ``(X,Y,Z)`` with ``X`` contains the
            longitudes, ``Y`` contains the latitude and ``Z`` the values
            extracted for the region of interest.

        Note:
            The argument has to be either lola or wac. Note case sensitive.
            All return arrays have the same size.

            All coordinates are in degree.
        '''

        if type_img.lower() == 'lola':
            return LolaMap(self.ppdlola, *self.window, path_pdsfile=self.path_pdsfiles).image()
        elif type_img.lower() == 'wac':
            return WacMap(self.ppdwac, *self.window, path_pdsfile=self.path_pdsfiles).image()
        else:
            raise ValueError('The img type has to be either "Lola" or "Wac"')

    def _format_coordinate(self, ax, m):
        ''' Format the basemap plot to show lat/long properly '''
        lon_m, lon_M, lat_m, lat_M = self.window
        xlocs = np.linspace(lon_m, lon_M, 5)
        ylocs = np.linspace(lat_m, lat_M, 5)
        xlocs = map(lambda x: float('%1.2f' % (x)), xlocs)
        ylocs = map(lambda x: float('%1.2f' % (x)), ylocs)
        m.drawparallels(ylocs, labels=[1, 0, 0, 1], ax=ax, fontsize=18)
        m.drawmeridians(xlocs, labels=[1, 0, 0, 1], ax=ax, fontsize=18)

    def get_profile(self, img_type, coordinate, num_points):
        ''' Extract a profile from (lat1,lon1) to (lat2,lon2)

        Args:
            img_type (str): Either lola or wac.
            coordinate (float,float,float,flaot): A tupple
                ``(lon0,lon1,lat0,lat1)`` with:

                - lon0: First point longitude
                - lat0: First point latitude
                - lon1: Second point longitude
                - lat1: Second point latitude

            num_points (int): Number of points to use in the
                interpolation process.

        Note:
            Be carefull, longitude has to be in between 0-360 !
        '''

        lon0, lon1, lat0, lat1 = coordinate
        X, Y, Z = self.get_arrays(img_type)
        y0, x0 = np.argmin(np.abs(X[0, :] - lon0)
                           ), np.argmin(np.abs(Y[:, 0] - lat0))
        y1, x1 = np.argmin(np.abs(X[0, :] - lon1)
                           ), np.argmin(np.abs(Y[:, 0] - lat1))
        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
        zi = scipy.ndimage.map_coordinates(Z, np.vstack((x, y)))

        return zi

    def draw_profile(self, coordinates, num_points=500, save=False,
                     name='BaseProfile.png'):
        ''' Draw a profile between a point (lon0,lat0) and (lon1,lat1).

        Args:
            coordinates: Tupples which list the different desired
                profiles.

                Each profil has to be defined as a tupple which follows
                (lon0,lon1,lat0,lat1) with (lon0,lat0) the first point
                coordintes and (lon1,lat1) the second point
                coordinates. Both in degree.
            num_points (Optional[int]): Number of points to use
                in the interpolation process. Defaults to 100.
            save (Optional[bool]): Weither or not to save the image.
                Defaults to False.
            name (Optional[str]): Absolut path to save the resulting
                image. Default to 'BaseProfile.png' in the working
                directory.

        Example:
            Here is an example for a region located (10E,10N) 20 km
            in diameter with three different profiles:

                - One North-South
                - One East-West
                - One inclined

            >>> Region = Area(10,10,20)
            >>> midlon = (Region.window[0]+Region.window[1])/2.0
            >>> midlat = (Region.window[2]+Region.window[3])/2.0
            >>> profile1 = (midlon,midlon,Region.window[2],Region.window[3])
            >>> profile2 = (Region.window[0],Region.window[1],midlat,midlat)
            >>> Region.draw_profile((profile1,profile2,Region.window,))

        Warning:
            If only one profile is given, ``coordinates = (profile1,)``.
            If more than one is given, use ``coordinates = (profile1,profile2,profile3,)``

            IF YOU DECIDE TO CHANGE THE PATH, YOU HAVE TO WRITE
            region.draw_profile(
                (profile1,profile2,region.window,), save = True, name = newpath)

            FOR SOME REASON, USING ONLY
            region.draw_profile(
                (profile1,profile2,region.window,), True, newpath)
            IS NOT WORKING

        '''

        fig = plt.figure(figsize=(27, len(coordinates) * 8))
        gs = gridspec.GridSpec(len(coordinates), 4)

        if len(coordinates) == 4:
            assert type(coordinates[0]) == tuple,\
                "If only one tupple is given,\n\
                the correct syntax is (tuple,) !! Not (tuple) ;)"

        for i, coordinate in enumerate(coordinates):

            ax1 = plt.subplot(gs[i, :2])
            ax2 = plt.subplot(gs[i, 2:])

            # Image unit
            lon_m, lon_M, lat_m, lat_M = self.window
            m = Basemap(llcrnrlon=lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                        resolution='i', projection='laea', rsphere=1734400, lat_0=self.lat0, lon_0=self.lon0)
            X, Y, Z = self.get_arrays('lola')
            X, Y = m(X, Y)
            CS = m.pcolormesh(X, Y, Z, cmap='gist_earth',
                              alpha=1, ax=ax1, zorder=1)
            self._format_coordinate(ax1, m)

            lon1, lon0, lat1, lat0 = coordinate
            lon0, lat0 = m(lon0, lat0)
            lon1, lat1 = m(lon1, lat1)
            ax1.plot([lon1, lon0], [lat1, lat0], 'ro-')

            # Profile
            print(coordinate)
            z_interpolated = self.get_profile('lola', coordinate, num_points)
            ax2.plot(z_interpolated, lw=2, marker='o')
            ax2.set_ylabel('Topographic profile (m)', fontsize=24)
            ax2.tick_params(labelsize=18)

        if save == True:
            fig.savefig(name)

    def lola_image(self, save=False, name='BaseLola.png'):
        ''' Draw the topography of the region of interest

        Args:
            save (Optional[bool]): Weither or not to save the image.
                Defaults to False.
            name (Optional[str]): Absolut path to save the resulting
                image. Default to 'BaseLola.png' in the working
                directory.

        Returns:
            An image correponding to the region tography. Realized
            from the data taken by the LOLA instrument on board of LRO.

        Note:
            Nice to use in a jupyter notebook with ``%matplotib inline``
            activated.

            Feel free to modify this method to plot exactly what you need.

        '''

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)

        lon_m, lon_M, lat_m, lat_M = self.lambert_window(
            self.size_window, self.lat0, self.lon0)
        m = Basemap(llcrnrlon=lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i', projection='laea', rsphere=1734400, lat_0=self.lat0, lon_0=self.lon0)

        Xl, Yl, Zl = self.get_arrays('Lola')
        Xl, Yl = m(Xl, Yl)

        CS = m.pcolormesh(Xl, Yl, Zl, cmap='gist_earth',
                          alpha=.5, ax=ax1, zorder=1)
        # m.contour(Xl,Yl,Zl,20, colors = 'black', alpha = 1.0 , zorder=2)

        xc, yc = m(self.lon0, self.lat0)
        ax1.scatter(xc, yc, s=200, marker='v', zorder=2)

        self._add_scale(m, ax1)
        self._add_colorbar(m, CS, ax1, 'Topography')

        if save == True:
            fig.savefig(name, rasterized=True, dpi=50,
                        bbox_inches='tight', pad_inches=0.1)

    def wac_image(self, save=False, name='BaseWac.png'):
        ''' Draw the wide angle image of the region of interest

        Args:
            save (Optional[bool]): Weither or not to save the image.
                Defaults to False.
            name (Optional[str]): Absolut path to save the resulting
                image. Default to 'BaseWac.png' in the working
                directory.

        Returns:
            An image corresponding to the region wide angle image. Realized
            from the data taken by the LROC instrument on board of LRO.

        Note:
            Nice to use in a jupyter notebook with ``%matplotib inline``
            activated.

            Feel free to modify this method to plot exactly what you need.

        '''

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)

        lon_m, lon_M, lat_m, lat_M = self.lambert_window(
            self.size_window, self.lat0, self.lon0)
        m = Basemap(llcrnrlon=lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i', projection='laea', rsphere=1734400, lat_0=self.lat0, lon_0=self.lon0)

        Xw, Yw, Zw = self.get_arrays('Wac')
        Xw, Yw = m(Xw, Yw)
        grid = m.pcolormesh(Xw, Yw, Zw, cmap=cm.gray, ax=ax1, zorder=1)

        xc, yc = m(self.lon0, self.lat0)
        ax1.scatter(xc, yc, s=200, marker='v', zorder=2)

        self._add_scale(m, ax1)

        if save == True:
            fig.savefig(name, dpi=50, bbox_inches='tight', pad_inches=0.1)

    def overlay(self, save=False, name='Baseoverlay.png'):
        ''' Draw the topography over a wide angle image  of the region

        Args:
            save (Optional[bool]): Weither or not to save the image.
                Defaults to False.
            name (Optional[str]): Absolut path to save the resulting
                image. Default to 'Baseoverlay.png' in the working
                directory.

        Returns:
            An image corresponding to an overaly of the topography
            and a wide angle image. Realized from the data taken
            by the LOLA and LROC instrument on board of LRO.

        Note:
            Nice to use in a jupyter notebook with ``%matplotib inline``
            activated.

            Feel free to modify this method to plot exactly what you need.

        '''

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)

        lon_m, lon_M, lat_m, lat_M = self.lambert_window(
            self.size_window, self.lat0, self.lon0)
        m = Basemap(llcrnrlon=lon_m, llcrnrlat=lat_m, urcrnrlon=lon_M, urcrnrlat=lat_M,
                    resolution='i', projection='laea', rsphere=1734400, lat_0=self.lat0, lon_0=self.lon0)

        Xw, Yw, Zw = self.get_arrays('Wac')
        Xw, Yw = m(Xw, Yw)
        m.pcolormesh(Xw, Yw, Zw, cmap=cm.gray, ax=ax1, zorder=1)

        Xl, Yl, Zl = self.get_arrays('Lola')
        Xl, Yl = m(Xl, Yl)
        CS = m.contourf(Xl, Yl, Zl, 100, cmap='gist_earth',
                        alpha=0.4, zorder=2, antialiased=True)

        xc, yc = m(self.lon0, self.lat0)
        ax1.scatter(xc, yc, s=200, marker='v', zorder=2)

        self._add_scale(m, ax1)
        self._add_colorbar(m, CS, ax1, 'Topography')

        if save == True:
            fig.savefig(name, dpi=50, bbox_inches='tight', pad_inches=0.1)

    def _Deg(self, radius):
        return radius * 360 / (2 * np.pi * 1734.4)


class Crater(Area):
    '''A class which gathers information on impact crater.

    It is particularly useful to study a particular impact crater at
    the lunar surface. For the moment, it can gather information about
    its topography (from LRO LOLA experiment) and texture (from the
    LRO WAC experiment). More information about the Lunar
    Reconnaissance Orbiter mission (LRO) can be found `here`_

    Args:
        ide (str): ``"name"`` if you use the crater name or
            ``"index"`` if you use its index in the table.
        idx: Name of the crater if you fill ``"name"`` as a first parameter or
            its index in the table if you fill ``"index"`` as a first parameter.
        path_pdsfiles (Optional[str]): Path where the pds files are stored.
            Defaults, the path is set to the folder ``PDS_FILES`` next to
            the module files where the library is install.

            See ``defaut_pdsfile`` variable of the class.

    Attributes:
        path_pdsfiles: Path where the pds_files are stored.
        ppdlola (int): Resolution for the topography
        ppdwac (int): Resolution for the WAC image
        racine (str): Path where information about the impact crater
            dataset is stored as a table. Defaults to the folder Tables
            in the installation folder of the library.
        craters: Pandas dataframes containing the information of
            all impact craters.
        name (str): Name of the crater considered.
        lat0 (float): Latitude of the crater center (degree)
        lon0 (float): Longitude of the crater center (degree)
        diameter (float): Crater diameter (km)
        type (int): 1 if the crater is a Floor-fractured crater, 0 otherwise
        radius (float): Radius of the crater (km)
        index (str): Index of the crater in the table
        size_window (float): Radius of the region of interest (km).
            Defaults to 80 % of the crater diameter.
        window (float,float,float,float): ``(longll, longtr, latll, lattr)``
            with:

            - ``longll`` the longitude of the lower left corner
            - ``longtr`` the longitude of the top right corner
            - ``latll`` the latitude of the lower left corner
            - ``lattr`` the latitude of the top right corner

    Note:
        It is important to respect the structure of the PDS_FILES folder. It
        should contain 2 subfolder called ``LOLA`` and ``LROC_WAC`` where the
        corresponding images should be download.

        The abreviations correspond to:

        - **LRO** Lunar Reconnaissance Orbiter
        - **LOLA** Lunar Orbiter Laser Altimeter
        - **LROC** Lunar Reconnaissance Orbiter Camera
        - **WAC** Wide Angle Camera

    Example:
        For instance, say we want to get an overlay, the topography
        drawn over a wide angle camera image, of the famous crater
        Copernicus

        >>> C = Crater('name','Copernicus')
        >>> C.overlay()

    .. _here:
        http://www.nasa.gov/mission_pages/LRO/spacecraft/#.VpOMDpMrKL4

    '''

    def __init__(self, ide, idx, path_pdsfile=Area.defaut_pdsfile):

        self.path_pdsfiles = path_pdsfile
        self.ppdlola = 512
        self.ppdwac = 128
        self.racine = os.path.join(
            '/'.join(os.path.abspath(__file__).split('/')[:-1]), 'Table')
        self.craters = pd.read_csv(
            os.path.join(self.racine, 'Data_Crater.csv'))

        df = self.craters[self.craters[ide] == idx]
        if len(df) == 0:
            raise ValueError("The tuple (%s,%s) does not correspond\n \
                             to any structure in the dataset. " % (ide, idx))

        def switchtype(x):
            try:
                return float(x)
            except:
                return str(x)

        [setattr(self, f, switchtype(df[f])) for f in df.columns]

        assert (self.lon0 > 0.0) & (
            self.lon0 < 360.0), 'Longitude has to span 0-360 !!!'
        self.name = df.name.iloc[0]
        self.change_window(0.8 * self.diameter)


class Dome(Area):

    '''A class which gathers information on lunar low-slope dome.

    It is particularly useful to study a particular low-slope dome at
    the lunar surface. For the moment, it can gather information about
    its topography (from LRO LOLA experiment) and texture (from the
    LRO WAC experiment). More information about the Lunar
    Reconnaissance Orbiter mission (LRO) can be found `here`_

    Args:
        ide (str): ``"name"`` if you use the dome name or
            ``"index"`` if you use its index in the table.
        idx: Name of the dome if you fill ``"name"`` as a first parameter or
            its index in the table if you fill ``"index"`` as a first parameter.
        path_pdsfiles (Optional[str]): Path where the pds files are stored.
            Defaults, the path is set to the folder ``PDS_FILES`` next to
            the module files where the library is install.

            See ``defaut_pdsfile`` variable of the class.

    Attributes:
        path_pdsfiles: Path where the pds_files are stored.
        ppdlola (int): Resolution for the topography
        ppdwac (int): Resolution for the WAC image
        racine (str): Path where information about the low-slope dome
            dataset is stored as a table. Defaults to the folder Tables
            in the installation folder of the library.
        domes: Pandas dataframes containing the information about
            the low-slope domes.
        name (str): Name of the crater considered.
        lat0 (float): Latitude of the dome center (degree)
        lon0 (float): Longitude of the dome center (degree)
        diameter (float): Dome diameter (km)
        radius (float): Radius of the crater (km)
        diameter_err (float): Error on the diameter (km)
        thickness (float): Dome thickness (km)
        thickness_err (float): Error on the dome thickness (km)
        index (str): Index of the dome in the table
        size_window (float): Radius of the region of interest (km).
            Defaults to 80 % of the crater diameter.
        window (float,float,float,float): ``(longll, longtr, latll, lattr)``
            with:

            - ``longll`` the longitude of the lower left corner
            - ``longtr`` the longitude of the top right corner
            - ``latll`` the latitude of the lower left corner
            - ``lattr`` the latitude of the top right corner

    Note:
        It is important to respect the structure of the PDS_FILES folder. It
        should contain 2 subfolder called ``LOLA`` and ``LROC_WAC`` where the
        corresponding images should be download.

        The abreviations correspond to:

        - **LRO** Lunar Reconnaissance Orbiter
        - **LOLA** Lunar Orbiter Laser Altimeter
        - **LROC** Lunar Reconnaissance Orbiter Camera
        - **WAC** Wide Angle Camera

    Example:
        For instance, say we want to get an overlay, the topography
        drawn over an wide angle camera image, of the famous dome
        M13

        >>> C = Dome('name','M13')
        >>> C.overlay()

    .. _here:
        http://www.nasa.gov/mission_pages/LRO/spacecraft/#.VpOMDpMrKL4

    '''

    def __init__(self, ide, idx, path_pdsfile=Area.defaut_pdsfile):

        self.path_pdsfiles = path_pdsfile
        self.ppdlola = 512
        self.ppdwac = 128
        self.racine = os.path.join(
            '/'.join(os.path.abspath(__file__).split('/')[:-1]), 'Table')
        self.domes = pd.read_csv(os.path.join(self.racine,
                                              'Data_Dome.csv'))
        df = self.domes[self.domes[ide] == idx]
        if len(df) == 0:
            raise ValueError("The tuple (%s,%s) does not correspond\n \
                             to any structure in the dataset. " % (ide, idx))

        def switchtype(x):
            try:
                return float(x)
            except:
                return str(x)

        [setattr(self, f, switchtype(df[f])) for f in df.columns]

        assert (self.lon0 > 0.0) & (
            self.lon0 < 360.0), 'Longitude has to span 0-360 !!!'
        self.name = df.name.iloc[0]
        self.change_window(0.8 * self.diameter)
