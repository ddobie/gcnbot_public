import healpy as hp

import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import numpy as np

import datetime
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')

import logging
import logging.config

def downsample_skymap(skymap, nside_out=64):
  npix = len(skymap)
  nside = hp.npix2nside(npix)
  
  skymap_ds = hp.ud_grade(skymap, nside_out, power=-2)
  logging.debug('Downgraded skymap to nside=%d'%(nside_out))
  
  return skymap_ds
  


def observable_probability_calc(skymap, location, start_time, radecs, delta_T, horizon=20*u.deg):

  probs = np.zeros(shape=delta_T.shape)

  for j,offset in enumerate(delta_T):
    time = start_time + offset
    observatory_frame = astropy.coordinates.AltAz(obstime=time, location=location)
    altaz = radecs.transform_to(observatory_frame)
    prob = skymap[altaz.alt >= horizon].sum()
    probs[j] = prob
    
  return probs
  
  
def radio_telescope_info():
  ATCA = EarthLocation(lat=-30.31277778*u.deg, lon=149.55000000*u.deg, height=236.87*u.m)
  ASKAP = EarthLocation(lat=-26.70416667*u.deg, lon=116.65888889*u.deg, height=100*u.m)
  VLA = EarthLocation(lat=34.07874917*u.deg, lon=-107.61777778*u.deg, height=100*u.m)
  MeerKAT = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=100*u.m)
  Apertif = EarthLocation(lat=52.9145*u.deg, lon=6.6027*u.deg, height=100*u.m)
  GMRT = EarthLocation(lat=19.093495*u.deg, lon=74.050333*u.deg, height=100*u.m)
  
  return locals()  

def observable_probabilty_plot(skymap, nside_ud=64, filename='test.png'):
  delta_T = np.linspace(0, 24, 24*3)*u.hour

  npix = len(skymap)
  nside = hp.npix2nside(npix)

  theta, phi = hp.pix2ang(nside, np.arange(npix))
  radecs = astropy.coordinates.SkyCoord(ra=phi*u.rad, dec=(0.5*np.pi - theta)*u.rad)
  
  npix = len(skymap)
  nside = hp.npix2nside(npix)
  
  if nside > nside_ud:
    skymap = observability.downsample_skymap(skymap)  
  
  # Look up (celestial) spherical polar coordinates of HEALPix grid.
  theta, phi = hp.pix2ang(nside, np.arange(npix))
  radecs = astropy.coordinates.SkyCoord(ra=phi*u.rad, dec=(0.5*np.pi - theta)*u.rad)
  
  current_time = datetime.datetime.now()
  midnight_ut = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
  
  start_time = Time(midnight_ut, scale='utc')
  start_time_str = datetime.datetime.strftime(midnight_ut,'%Y-%m-%d')
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  radio_telescope_dict = radio_telescope_info()
  
  for obsname, obscoord in radio_telescope_dict.items():
    probs = observable_probability_calc(skymap, obscoord, start_time, radecs, delta_T)
  
    ax.plot(delta_T, probs, label=obsname)
  
  ax.set_xlim(0,24)
  ax.set_ylim(bottom=0,top=1.0)
  
  ax.set_ylabel('Probability observable')
  ax.set_xlabel('Hours from midnight UT on %s'%(start_time_str))
  
  plt.legend()
  plt.savefig(filename,dpi=200)
  logging.debug('Saved observability plot')
  
  #time_midnight = 
  #print(time.date)
