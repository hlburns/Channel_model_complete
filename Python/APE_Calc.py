# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:24:11 2016
Adapted from Ryan Abernathey's APE scripts
@author: hb1g13
"""

""" Python module for Energy calculations
    Assumes following file structure:
    outputdir/
              grid.nc (grid)
              Tav.nc  (Timeav U, V,W,T)
              PSI.nc (layers package output)
              Tav_VT.nc  (Timeav UT, VT, WT)
    where *.nc are flies glued together with gluemncbig
    Usage:
    append path to incl this script and:
    c = utils.ChannelSetup(output_dir=str(fname))
    c.var or c.func()
"""

import numpy.ma as ma
import numpy as np


#from scipy.interpolate import interp1d
class APEcomputer:

    def __init__(self, channelsetup, b=None):
        c = channelsetup
        self.c = c
        self.__dict__.update(locals())
        self.vol = c.HFacC*c.rac*np.tile(c.dzf,(c.Nx,c.Ny,1)).T
        # this definies the volume of the z grid
        # (indez of z goes down from the surface)
        self.zf_vol = np.hstack([0,np.cumsum(self.vol.sum(axis=2).sum(axis=1))])

        if b is None:
            T = self.c.mnc('Tav.nc', 'THETA')
            b = self.c.g * self.c.tAlpha * T#.mean(axis=2)

            self.b = b
        self.compute_Zstar()

    def new_b(self,b):
        self.b = b
        self.compute_Zstar()

    def calc_PE(self):
        return sum( -self.b * self.c.zc[:,np.newaxis,np.newaxis] * self.vol )

    def calc_BPE(self):
        return sum( -self.Bstar[:,np.newaxis,np.newaxis] *
                    self.c.zc[:,np.newaxis,np.newaxis] * self.vol )

    def calc_BPE_via_Zstar(self):
        return sum(self.vol * self.Zstar * -self.b)

    def calc_APE(self):
        return sum( (-self.b + self.Bstar[:,np.newaxis,np.newaxis])*
                    self.c.zc[:,np.newaxis,np.newaxis] * self.vol )

    def calc_APE_3d(self):
        return ( (-self.b + self.Bstar[:,np.newaxis,np.newaxis])*
                    self.c.zc[:,np.newaxis,np.newaxis] * self.vol )


    def Zstar_of_b(self,b):
        """Takes b as an input, returns the corresponding Z*"""
        return ma.masked_array(np.interp(-b,-self.Bstar_f,self.c.zf),1-self.c.HFacC)

    def calc_Bstar_f(self,b,vol,zf_vol):
        maskidx = np.where(vol.ravel()>0)[0]
        b_flat = b.ravel()[maskidx]
        vol_flat = vol.ravel()[maskidx]
        sortidx = np.argsort(-b_flat)
        cumvol = np.cumsum(vol_flat[sortidx])
        return np.interp(zf_vol, cumvol, b_flat[sortidx])

    def compute_Zstar(self):
        """Creates the Zstar profile through sorting"""
        self.Bstar_f = self.calc_Bstar_f(self.b,self.vol,self.zf_vol)
        self.Bstar = 0.5 * (self.Bstar_f[1:] + self.Bstar_f[:-1])
        self.Zstar = self.Zstar_of_b(self.b)
