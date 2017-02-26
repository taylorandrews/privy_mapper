from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # 22181 coordinates
    lat_0_22181 = 38.908
    lon_0_22181 = -77.293
    llcrnrlon_22181 = 38.872
    llcrnrlat_22181 = -77.329
    urcrnrlon_22181 = 38.937
    urcrnrlat_22181 = -77.261

    map = Basemap(projection='merc',
                  resolution='c',
                  lat_0=lat_0_22181,
                  lon_0=lon_0_22181,
                  llcrnrlon=llcrnrlon_22181,
                  llcrnrlat=llcrnrlat_22181,
                  urcrnrlon=urcrnrlon_22181,
                  urcrnrlat=urcrnrlat_22181)

    # map.drawcoastlines(linewidth=0.25)
    # map.drawcountries(linewidth=0.25)
    # map.fillcontinents(color='coral',lake_color='aqua')
    # map.drawmapboundary(fill_color='aqua')

    map.readshapefile('../virginia-latest-free/osm_roads_free_1', 'roads')
    # map.drawmeridians(np.arange(0,360,30))
    # map.drawparallels(np.arange(-90,90,30))
    # nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
    # lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
    # lons = (delta*np.indices((nlats,nlons))[1,:,:])
    # wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
    # mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
    # x, y = map(lons*180./np.pi, lats*180./np.pi)
    # cs = map.contour(x,y,wave+mean,15,linewidths=1.5)
    plt.title('hi')
    plt.show()
