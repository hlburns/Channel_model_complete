# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:01:17 2016

@author: hb1g13
"""
from numba import jit 
import numpy.ma as ma
import numpy as np
import sys
sys.path.append('/noc/users/hb1g13/Python/python_functions/MITgcmUtils/')
import utils
#import netCDF4
#from scipy.interpolate import interp1d

class LayersComputer:
    """Layers calulater:.layers_calc.LayersComputer(self,
    ChannelSetup, fine_grid_fac=10)
    """
  
    def __init__(self, channelsetup, fine_grid_fac=10):
        """A class to facilitate computation of layer transport."""
        # automatically make set instance variables from arguments
        self.__dict__.update(locals())
        c = channelsetup
        self.c = c
        nz = self.c.Nz
        # set up global vars for interpolation
        #(copy this from MITgcm LAYERS pacakge)
        # print self.c.output_dir
        layers_g = c.layers_bounds
        #layers_g = np.arange(-1,11,.1)+0.1
        self.fine_grid_fac = fine_grid_fac
        self.layers_g = layers_g
        self.ng = len(self.layers_g)-1
        self.glvl = 0.5 *(self.layers_g[:-1] + self.layers_g[1:])
        self.nzz = nz * self.fine_grid_fac
        self.dzzf = np.empty(self.nzz)
        self.map_fac = np.empty(self.nzz)
        self.map_idx = np.empty(self.nzz)
        # for quickly breaking down z-grid
        self.break_idx = np.tile(np.arange(nz),
                                 [self.fine_grid_fac, 1]).T.flatten()

        self.dzzf = self._break_into_zz(self.c.dzf)/self.fine_grid_fac
        #for k in np.arange(nz):
        #   self.dzzf[(k*self.fine_grid_fac):((k+1)*self.fine_grid_f
        #ac) ] = self.c.dzf[k]/self.fine_grid_fac
        self.zzf = np.empty(self.nzz+1)
        self.zzf[1:] = -np.cumsum(self.dzzf)
        self.zzc = 0.5 *(self.zzf[:-1] + self.zzf[1:])

        # It would probably be possible to use a built-in scipy
        #interpolation routine,
        # but instead I have tried to follow the spirit of the layers
        # code as closely as possible.
        raw_map_idx = np.interp(-self.zzc, -self.c.zc, np.arange(nz))
        self.map_idx = np.floor(raw_map_idx).astype('int')
        self.map_idx[self.map_idx == nz-1] = nz-2
        self.map_fac = 1 -(raw_map_idx - self.map_idx)

        # figure out if we can use the fortran module for speedy interpolation
   
    @jit
    def transform_g_to_z(self, f, h, zpoint='C'):
        if f.shape != h.shape:
            raise ValueError('f and h must have the same shape')
        if zpoint == 'C':
            z = self.c.zc
        elif zpoint == 'F':
            z = self.c.zf
        else:
            raise ValueError('Only zpoints C or F are allowed.')

        # Assume that the first index of h and f is at the bottom of the domain.
        # To determine the depth, we have to sum from the top
        zf = -np.cumsum(h[::-1], axis=0)
        f = f[::-1]
        # handle multiple dimensions
        sh = f.shape
        if len(sh) == 2:
            fz = np.empty((len(z), sh[1]))
        elif len(sh) == 3:
            fz = np.empty((len(z), sh[1], sh[2]))
        for j in np.arange(sh[1]):
            if len(sh) == 3:
                for i in np.arange(sh[2]):
                    fz[:, j, i] = np.interp(-z, -zf[:, j, i], f[:, j, i])
            elif len(sh) == 2:
                fz[:, j] = np.interp(-z, -zf[:, j], f[:, j])
        return fz
        
  
    @jit
    def compute_uflux(self):
        """Does the zonal layers calculation at a particular iteration"""
        u = self.c.mnc('Tav.nc', 'UVEL')
        g = utils.cgrid_to_ugrid(self._load_g_field())
        return self._compute_flux(u, g, self.c.HFacS)
        
  
    @jit
    def compute_vflux(self):
        """Does the zonal layers calculation at a particular iteration"""
        v = self.c.mnc('Tav.nc', 'VVEL')
        g = utils.cgrid_to_vgrid(self._load_g_field())
        return self._compute_flux(v, g)
        

    @jit
    def interp_to_g(self, q, g):
        """Interpolate c-grid variable (tracer) to isopycnal (g) coordinates"""
        hq, h = self._compute_flux(q, g)
        #return ma.masked_array(hq / h, h == 0.), h
        return hq, h
   
    @jit
    def _compute_flux(self, v, g):
        """Low-level generic function for computing layer flux, assuming
           all variables on the same grid"""
        if isinstance(v, ma.masked_array):
            v = v.filled(0.)
        if isinstance(g, ma.masked_array):
            g = g.filled(0.)
        vflux = np.empty((self.ng, self.c.Ny, self.c.Nx))
        hv = np.empty((self.ng, self.c.Ny, self.c.Nx))

        for j in np.arange(self.c.Ny):
            for i in np.arange(self.c.Nx):
                var = self.c.HFacS[:, j, i]
                vflux[:, j, i], hv[:, j, i] = self._interp_column(v[:, j, i],
                                                                  g[:, j, i],
                                                                  var)
        return vflux, hv
        
    
    @jit
    def _interp_column(self, u, g, hfac):
        """Low-level function for partitioning a field u, defined at points g,
           into layers. The other parameters (grid geometry, g_levs,etc.)
           are derived from the parent class."""
        gzz = self._value_at_zz(g)
        uzz = self._break_into_zz(u)
        layer_idx = self._assign_to_layers(gzz)
        h = np.zeros(self.ng)
        uh = np.zeros(self.ng)
        dzzf = self.dzzf * self._break_into_zz(hfac)
        for kk in np.arange(self.nzz):
            h[layer_idx[kk]] += dzzf[kk]
            uh[layer_idx[kk]] += dzzf[kk] * uzz[kk]
        return uh, h
        
    
    @jit
    def _value_at_zz(self, g):
        """Low-level function to interpolate field g to fine grid zz points"""
        return self.map_fac*g[self.map_idx] +(1-self.map_fac)*g[self.map_idx+1]
        
 
    @jit
    def _break_into_zz(self, u):
        """Low-level function to partition u onto the fine vertical grid,
        conserving total u integral."""
        return u[self.break_idx]
        

    @jit
    def _assign_to_layers(self, gzz):
        """Low-level function that does the hard work of figuring out which
        points in g belong in which layers."""
        # use the same hunting algorithm as the MITgcm code
        # layers_g index (e.g. temperature) goes from cold to warm
	   # water column goes from warm (k=1) to cold, so initialize the
        #search with the warmest value

        kg = self.ng-1
        layer_idx = np.empty(self.nzz)
        # remember that the top and bottom values of layers_g are totally
        # ignored maybe this is weird, but that is how LAYERS does it
        for kk in np.arange(self.nzz):
            g = gzz[kk] # the value of g in this box
            if g >= self.layers_g[self.ng-1]:
		     # the point is in the hottest bin or hotter
                kg = self.ng-1
            elif g < self.layers_g[1]:
		     # the point is in the coldest bin or colder
                kg = 0
            elif (g >= self.layers_g[kg]) and (g < self.layers_g[kg+1]):
	          # already in the right bin
                pass
            elif g >= self.layers_g[kg]:
                while g >= self.layers_g[kg+1]:
                    kg += 1
            elif g < self.layers_g[kg+1]:
                while g < self.layers_g[kg]:
                    kg -= 1
            else:
                raise ValueError('Couldn\'t find a bin in layers_g for g=%g'%g)
            layer_idx[kk] = kg
        return layer_idx
        
  
    @jit
    def _load_g_field(self):
        """Docstring
        """
        return self.c.mnc('Tav.nc', 'THETA')
    
    @jit
    def interp_to_g1(self, q, g):
        """Interpolate c-grid variable (tracer) to isopycnal (g) coordinates"""
        hq, h = self._compute_layer_val(q, g)
        #return ma.masked_array(hq , h == 0.), h       
        return hq, h
    @jit
    def _compute_layer_val(self, v, g):
        """Low-level generic function for computing layer flux, assuming
           all variables on the same grid"""
        if isinstance(v, ma.masked_array):
            v = v.filled(0.)
        if isinstance(g, ma.masked_array):
            g = g.filled(0.)
        vflux = np.empty((self.ng, self.c.Ny, self.c.Nx))
        hv = np.empty((self.ng, self.c.Ny, self.c.Nx))

        for j in np.arange(self.c.Ny):
            for i in np.arange(self.c.Nx):
                var = self.c.HFacS[:, j, i]
                vflux[:, j, i], hv[:, j, i] = self._interp_column_1(v[:, j, i],
                                                                  g[:, j, i],
                                                                  var)
        return vflux, hv
        
    
    @jit
    def _interp_column_1(self, u, g, hfac):
        """Low-level function for partitioning a field u, defined at points g,
           into layers. The other parameters (grid geometry, g_levs,etc.)
           are derived from the parent class."""
        gzz = self._value_at_zz(g)
        uzz = self._break_into_zz(u)
        layer_idx = self._assign_to_layers(gzz)
        h = np.zeros(self.ng)
        uh = np.zeros(self.ng)
        dzzf = self.dzzf * self._break_into_zz(hfac)
        for kk in np.arange(self.nzz):
            h[layer_idx[kk]] += dzzf[kk]
            uh[layer_idx[kk]] = uzz[kk]
        return uh, h