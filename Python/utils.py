""" Python module for working with my MITgcm  output
    Channel model: Burns PhD Theis
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

import os.path
import numpy.ma as ma
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d

class ChannelSetup:
    """Contains a general description of the GCM grid and configuration
       Load in grid, with option to load in variables, time average,
       take gradients and regrid as well as a few others functions.
    """

    # global variables
    Nx, Ny, Nz = 0, 0, 0
    Lx, Ly, H = 0, 0, 0

    def __init__(self, output_dir, deltaT=8.0, N=1000):

        self.output_dir = output_dir
        self.f0 = -1e-4
        self.beta = 1e-11
        self.g = 9.8
        self.tAlpha = 2e-4
        self.N = N
        self.deltaT = deltaT
        #self.layers_bounds = np.arange(-1.,11.,0.03)
        self.layers_bounds = np.genfromtxt(str(self.output_dir)+'/Temp',delimiter=',')
        #np.genfromtxt(str(self.output_dir)+'/Temp',delimiter=',')
        global Nx, Ny, Nz, Lx, Ly, H

        file2read = netCDF4.Dataset(self.output_dir+'/grid.nc', 'r')
        self.xc = file2read.variables['XC'][0, :]
        self.xg = file2read.variables['XG'][0, :]
        self.yc = file2read.variables['YC'][:, 0]
        self.yg = file2read.variables['YG'][:, 0]
        self.zc = file2read.variables['RC'][:]
        self.zf = file2read.variables['RF'][:]
        self.zf = file2read.variables['RF'][:]
        # for derivatives and integrals
        self.rac = file2read.variables['rA']
        self.dyc = file2read.variables['dyC'][:, 0]
        self.dyg = file2read.variables['dyG'][:, 0]
        self.dxc = file2read.variables['dxC'][0, :]
        self.dxg = file2read.variables['dxG'][0, :]
        self.dzc = file2read.variables['drC'][:]
        self.dzf = file2read.variables['drF'][:]
        self.Depth = file2read.variables['Depth'][:]
        # masks
        self.HFacC = file2read.variables['HFacC'][:]
        self.HFacS = file2read.variables['HFacS'][:]
        self.HFacW = file2read.variables['HFacW'][:]

        Nx, Ny, Nz = len(self.xc), len(self.yc), len(self.zc)
        Lx, Ly, H = self.dxg.sum(), self.dyg.sum(), self.dzf.sum()

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.H = Lx, Ly, H

        self.ksign = 1
        # layer bounds
        self.Nlayers = len(self.layers_bounds)-1
        # w-cell bounds
        self.layers_bounds_w = 0.5 * (self.layers_bounds[1:] +
                                      self.layers_bounds[:-1])
        # top points (where w is computed)
        self.layers_top = self.layers_bounds[:-1]
        # Cell Volume array
        self.CellVol = self.rac*np.tile(self.dzf,(self.Nx,self.Ny,1)).T

    def mnc(self, fname, varname, mask=None):
        """import var with  masking capability"""
        file2read = netCDF4.Dataset(self.output_dir+fname, 'r')

        var = file2read.variables[varname][:].mean(axis=0)
        if mask is not None:
            out = ma.masked_array(var, 1-mask)
        else:
            out = var
        return out

    def mnc_t(self, fname, varname, tt=0, mask=None):
        """import var with  masking capability"""
        file2read = netCDF4.Dataset(self.output_dir+fname, 'r')

        var = file2read.variables[varname][tt,:,:,:]
        if mask is not None:
            out = ma.masked_array(var, 1-mask)
        else:
            out = var
        return out

    def mnc_tl(self, fname, varname, mask=None):
        """import var with  masking capability, NO TIMESLICE"""
        file2read = netCDF4.Dataset(self.output_dir+fname, 'r')

        var = file2read.variables[varname][:]
        if mask is not None:
            out = ma.masked_array(var, 1-mask)
        else:
            out = var
        return out


    def wgrid_to_cgrid(self, q, neumann_bc=False):
        """Interpolate a w-grid variable (cell top) vertically
        down to the c-grid (cell center)"""
        q_up = q.copy()[0:-1] # the current point
        q_dn = q.copy()[1::]#[np.r_[1:Nz, Nz-1]] # the point below
        # apply masks
        if neumann_bc:
            mask = (self.HFacC == 0.)
            q_dn[mask[np.r_[1:Nz, Nz-1]]] = q_up[mask[np.r_[1:Nz, Nz-1]]]
        # otherwise the point will be naturally interpolated assuming w=0 at
        # the boundary (doesn't account for partial cells!)

        return 0.5 * (q_dn + q_up)

    def wgrid_to_cgridnz(self, q, nz, neumann_bc=False):
        """Interpolate a w-grid variable (cell top) vertically
        down to the c-grid (cell center)"""

        q_up = q.copy()[0:-1] # the current point
        q_dn = q.copy()[1::] # the point below

        if neumann_bc:
            mask = (self.HFacC == 0.)
            q_dn[mask[np.r_[1:nz, nz-1]]] = q_up[mask[np.r_[1:nz, nz-1]]]
        # otherwise the point will be naturally interpolated assuming w=0 at
        # the boundary (doesn't account for partial cells!)

        return 0.5 * (q_dn + q_up)


    def vgrid_to_cgrid(self, q, neumann_bc=False):
        """Interpolate v-grid variable (cell southern bdry) hortizontally
        to the c-grid (cell center)"""
        # make sure to mask land properly
        # values inside land are zero, so don't try to interpolate through them
        # apply masks --> implement for Full topo runs
        q_south = q.copy()[:, 0:-1] # the current point
        q_north = q.copy()[:, np.r_[1:Ny, Ny-1]] # the point north
        if neumann_bc:
            mask = (self.HFacS == 0.)
            q_south[mask] = q_north[mask]
            q_north[mask[:, np.r_[1:Ny, Ny-1]]] = q_south[mask[:, np.r_[1:Ny, Ny-1]]]
        return 0.5 * (q_north + q_south)

    def vgrid_to_cgridall(self, q, neumann_bc=False):
        """Interpolate v-grid variable (cell southern bdry) hortizontally
        to the c-grid (cell center)"""
        # make sure to mask land properly
        # values inside land are zero, so don't try to interpolate through them
        # apply masks --> implement for Full topo runs
        q_south = q.copy()[:, :, 0:-1, :] # the current point
        q_north = q.copy()[:, :, np.r_[1:Ny, Ny-1], :] # the point north
        if neumann_bc:
            mask = (self.HFacS == 0.)
            q_south[mask] = q_north[mask]
            q_north[mask[:, np.r_[1:Ny, Ny-1]]] = q_south[mask[:, np.r_[1:Ny, Ny-1]]]
        return 0.5 * (q_north + q_south)

    def ugrid_to_cgrid(self, q, neumann_bc=False):
        """Interpolate v-grid variable (cell southern bdry) hortizontally
        to the c-grid (cell center)"""
        # make sure to mask land properly
        # values inside land are zero, so don't try to interpolate through them
        q_east = q.copy()[:, :, 0:-1] # the current point
        q_west = q.copy()[:, :, 1::] # the point north
        # apply masks --> implement for Full topo runs
        if neumann_bc:
            mask = (self.HFacS == 0.)
            q_east[mask] = q_east[mask]
            q_west[mask[:, :, np.r_[1:Nx, Nx-1]]] = q_east[mask[:, :, np.r_[1:Nx, Nx-1]]]

        return 0.5 * (q_east + q_west)

    def ugrid_to_cgridall(self, q, neumann_bc=False):
        """Interpolate v-grid variable (cell southern bdry) hortizontally
        to the c-grid (cell center)"""
        # make sure to mask land properly
        # values inside land are zero, so don't try to interpolate through them
        q_east = q.copy()[:, :, :, 0:-1] # the current point
        q_west = q.copy()[:, :, :, 1::] # the point north
        # apply masks --> implement for Full topo runs
        if neumann_bc:
            mask = (self.HFacS == 0.)
            q_east[mask] = q_east[mask]
            q_west[mask[:, :, np.r_[1:Nx, Nx-1]]] = q_east[mask[:, :, np.r_[1:Nx, Nx-1]]]

        return 0.5 * (q_east + q_west)

    def ddz_cgrid_centered(self, q):
        """Vertical second-order centered difference on the c grid"""
        dzf = np.tile(self.dzf, (Ny, 1)).T
        if len(q.shape) == 3:
            dzf = np.tile(dzf.T, (Nx, 1, 1)).T
        out = np.zeros(q.shape)
        # second order for interior
        out[1:Nz-1, :] = ma.divide((q[1:-1] - q[2::]), (dzf[1:-1] + dzf[2::]))
        # first order for the top and bottom
        out[0, :] = ma.divide((q[0, :] - q[1, :]), dzf[0, :])
        out[-1, :] = ma.divide((q[-2, :] - q[-1, :]), dzf[-1, :])

        return out

    def ddy_cgrid_centered(self, q):
        """Merdional second-order centered difference on the c grid"""

        dyg = np.tile(self.dyg, (Nz, 1))
        if len(q.shape) == 3:
            dyg = np.tile(dyg.T, (Nx, 1, 1)).T
        out = np.zeros(q.shape)
        out[:, 1:-1] = ma.divide((q[:, 2::] - q[:, 1:-1]),
                                 (dyg[:, 1:-1] + dyg[:, 2::]))
        out[:, 0] = ma.divide((q[:, 1] - q[:, 0]), dyg[:, 0])
        out[:, -1] = ma.divide((q[:, -1] - q[:, -2]), dyg[:, -1])

        #if isinstance(q, ma.masked_array):
        #    mask = q.mask
        #    mask[:,1:Ny-1] = q.mask[:,2:] | q.mask[:,:Ny-2]
         #   out = ma.masked_array(out, mask)

        return out

    def ddy_cgrid_centered_1D(self, q):
        """Merdional second-order centered difference on the c grid"""

        dyg = self.dyg
        out = np.zeros(q.shape)
        out[ 1:-1] = ma.divide((q[ 2::] - q[ 1:-1]),
                                 (dyg[ 1:-1] + dyg[ 2::]))
        out[ 0] = ma.divide((q[ 1] - q[ 0]), dyg[ 0])
        out[ -1] = ma.divide((q[ -1] - q[ -2]), dyg[ -1])

        #if isinstance(q, ma.masked_array):
        #    mask = q.mask
        #    mask[:,1:Ny-1] = q.mask[:,2:] | q.mask[:,:Ny-2]
         #   out = ma.masked_array(out, mask)

        return out

    def ddx_cgrid_centered(self, q):
        dxc = np.tile(self.dxc, (Nz, Ny, 1))
        out = np.zeros(q.shape)
        out[:, :, 1:Nx-1] = ma.divide((q[:, :, 2::] - q[:, :, :Nx-2]),
                                      (dxc[:, :, 1:Nx-1] + dxc[:, :, 3::]))
        out[:, :, 0] = ma.divide((q[:, :, 1] - q[:, :, 0]), dxc[:, :, 0])
        out[:, :, -1] = ma.divide((q[:, :, -1] - q[:, :, -2]), dxc[:, :, -1])
        return out

    def get_zonal_avg(self, filename, varname, mask=None):
        """Read the file and take a zonal average"""
        if mask is not None:
            out = ma.mean(self.mnc(filename, varname, mask), axis=2)

        else:
            out = np.nanmean(self.mnc(filename, varname), axis=2)

        return out

    def get_N2(self):
        """ Docstring """
        b = (self.g * self.tAlpha *
             self.get_zonal_avg('Tav.nc', 'THETA', mask=self.HFacC[:]))
        return self.ddz_cgrid_centered(b)

    def get_Sp(self):
        """ Docstring """
        b = self.get_zonal_avg('Tav.nc', 'THETA', mask=None)

        return ma.divide(-self.ddy_cgrid_centered(b), self.ddz_cgrid_centered(b))


    def get_psi_iso(self):
        """Read output from layers package to constructe isopycnal
        streamfunction."""

        V = self.get_zonal_avg('PSI.nc', 'LaVH1TH')
        return -V.cumsum(axis=0)*self.Lx


    def get_psi_iso_z(self):
        """Put the output from psi_iso into Z coordinates."""

        psi_iso = self.get_psi_iso()
        # figure out the depth of each layer
        h = self.get_zonal_avg('PSI.nc', 'LaHs1TH')
        # psi_iso is defined at the *bottom* of each layer,
        # therefore we want the depth at the bottom of the layer
        z = np.cumsum(h, axis=0)
        # interpolate to center z points
        depth = np.mean(self.Depth, axis=1) # Must use depth for Topo runs!!
        psi_iso_z = np.zeros((Nz, Ny))
        for j in np.arange(Ny):
            layer_depth = z[:, j] - depth[j]
            psi_iso_z[:, j] = np.interp(self.zc, layer_depth[:], psi_iso[:, j])
        return psi_iso_z


    # use c instead of self, shorter to write
    def get_qgpv_grad(self, mask=None):
        """Calculate QGPV gradient from standard output fields"""
        if mask is not None:
            T = self.mnc('Tav.nc', 'THETA', mask)
        else:
            T = self.mnc('Tav.nc', 'THETA')
        # isopycnal slope
        s = ma.divide(- self.ddy_cgrid_centered(T), self.ddz_cgrid_centered(T))
        return self.beta - self.f0 * self.ddz_cgrid_centered(s)

    def get_qgpv(self, mask=None):
        """Calculate QGPV gradient from standard output fields"""
        if mask is not None:
            T = self.get_zonal_avg('Tav.nc', 'THETA', mask)
        else:
            T = self.get_zonal_avg('Tav.nc', 'THETA')
        bN2dz = self.ddz_cgrid_centered(T/ self.ddz_cgrid_centered(T))
        return self.f0 * bN2dz

    def get_psi_bar(self, V=None, zpoint='F'):
        """Doc String"""
        if V is None:
            V = self.mnc('Tav.nc', 'VVEL', mask=self.HFacS[:])
        vflux = V * self.dzf[:, np.newaxis, np.newaxis]
        Vdx = vflux*self.HFacS
        Vdx = ma.mean(Vdx, axis=2)*self.Lx
        psi = ma.cumsum(Vdx, axis=0)
        if zpoint == 'F':
            return psi
        elif zpoint == 'C':
            psi = ma.apply_along_axis(np.vstack, 1, [np.zeros(self.Ny+1), psi])
            return 0.5 * (psi[1:] + psi[:-1])


    def get_theta(self, N=1000):
        """Remove reference temps from temps """

        Tref = np.divide(self.deltaT*(np.exp(self.zc/N)-np.exp(-self.H/N)),
                         (1-np.exp(-self.H/N)))

        THETA = ma.subtract(self.mnc('Tav.nc', 'THETA', mask=self.HFacC[:]),
                            np.tile(Tref, (self.Nx, self.Ny, 1)).T)
        return THETA

    def depth_average(self, Var):
        """Depth Average a variable in C-grid with varying depth"""
        Depth_av = (ma.mean(ma.divide(Var *
                                      np.tile(self.dzf, (self.Nx, self.Ny, 1)).T,
                                      self.Depth), axis=0))
        return Depth_av

    def depth_integrate(self, Var):
        """Depth Integrate a variable in C-grid with varying depth"""
        Depth_int = ma.sum(Var*np.tile(self.dzf, (self.Nx, self.Ny, 1)).T,
                           axis=0)
        return Depth_int


    def barotropic_streamfuc(self):
        """Calculate depth integrated barotropic stream function Psi(x, y)
        in Sv"""
        U = self.mnc('Tav.nc', 'UVEL', mask=self.HFacW[:])
        Psi = ma.cumsum(ma.sum(U*np.tile(self.dzf, (self.Nx+1, self.Ny, 1)).T,
                               axis=0)[::-1, :]*5000, axis=0)[::-1, :]
        Psi_sv = Psi/10**6
        return Psi_sv

#    def _setup_bottom_idx(self):
#        mask = self.mask[np.r_[:self.Nz, self.Nz-1]]
#        mask[-1] = True
#        isbot = ~mask[:self.Nz] & mask[1:]
#        # deal with the surface
#        isbot[0] = mask[0]
#        isbot_mask = np.zeros_like(isbot)
#        isbot_mask[0] = mask[0]
#        # use fortran byte order to do ravel
#        self._bot_idx = np.where(np.ravel(isbot, order='F'))[0]
#        self._bot_idx_mask = np.ravel(isbot_mask, order='F')[self._bot_idx]

#    def value_at_bottom(self, q, point='C'):
#        qb = ma.masked_array( np.ravel(q, order='F')[self._bot_idx], self._bot_idx_mask)
#        qb.shape = (self.Nx, self.Ny)
#        return np.copy(np.transpose(qb))
    def diapycnal_velocity(self, d):
        """Calculate diapycnal velocity, given the appropriate
        layers output."""
        # Ts, Th, Tr, Tha, Tra, Ss, Sh, Sr, Sha, Sra =
        # m.rdmds('DiagLAYERS-diapycnal')[:, ::ksign]
        wflux = d[:, ::self.ksign] * self.rac
        return wflux

    def advective_flux_divergence(self, uh, vh):
        """Calculate the advective diapycnal velocity resulting
        from advective flux divergence."""
        uflux = uh[::self.ksign] * self.dyg
        vflux = vh[::self.ksign] * self.dxg
        div_uflux = (np.roll(uflux, -1, axis=-1)-uflux).cumsum(axis=0)
        div_vflux = (np.roll(vflux, -1, axis=-2)-vflux).cumsum(axis=0)

        # don't include top point because we don't have a
        # diapycnal velocity there
        wflux = -(div_uflux[:-1] + div_vflux[:-1])
        return wflux

    def calc_KEt(self):
        """Calculates domain KE NO MASK"""
        return 0.5 * ((self.ugrid_to_cgridall(self.mnc_tl('VSQ1.nc','UVELSQ'))
                       + self.vgrid_to_cgridall(self.mnc_tl('VSQ1.nc','VVELSQ')
                       )*self.CellVol*self.HFacC).sum(axis=3).sum(axis=2).sum(axis=1))


    def calc_EKE(self):
        """Calculates EKE from reynolds decomp"""
        return 0.5 * ((self.mnc('VSQ.nc','UVELSQ',mask=self.HFacW[:])
                       - self.mnc('Tav.nc','UVEL',
                                  mask=self.HFacW[:])**2).mean(axis=2)
                      + (self.vgrid_to_cgrid(self.mnc('VSQ.nc','VVELSQ',
                                                      mask=self.HFacS[:]) -
                                             self.mnc('Tav.nc','VVEL',
                                                      mask=self.HFacS[:])**2).mean(axis=2)))


    def calc_EKE_all(self):
        """Calculates EKE from reynolds decomp"""
        return 0.5 * (self.ugrid_to_cgrid(self.mnc('VSQ.nc','UVELSQ',
                                                   mask=self.HFacW[:])
                       - self.mnc('Tav.nc','UVEL',
                                  mask=self.HFacW[:])**2)
                      + (self.vgrid_to_cgrid(self.mnc('VSQ.nc','VVELSQ',
                                                      mask=self.HFacS[:]) -
                                             self.mnc('Tav.nc','VVEL',
                                                      mask=self.HFacS[:])**2)))


    def calc_MKE(self):
        """Calculates mean KE """
        return 0.5 * ((self.mnc('Tav.nc','UVEL',mask=self.HFacW[:])**2).mean(axis=2)
                      + self.vgrid_to_cgrid((self.mnc('Tav.nc','VVEL',
                                                       mask=self.HFacS[:])**2).mean(axis=2)))

    def calc_MKE_all(self):
        """Calculates mean KE """
        return 0.5 * (self.ugrid_to_cgrid(self.mnc('Tav.nc','UVEL',mask=self.HFacW[:])**2)
                      + self.vgrid_to_cgrid((self.mnc('Tav.nc','VVEL',
                                                       mask=self.HFacS[:])**2)))

    def avg_cross_stream(self, q):
        """ q = masked array
        """
        X,Y = np.meshgrid(self.xc,self.yc)
        T0 = np.squeeze(self.depth_average(self.mnc('Tav.nc','THETA')))
        Tbnds = np.squeeze(T0).mean(axis=1)[([self.Ny*2/3, self.Ny/2])]
        T00 = Tbnds.mean()
        mask = (T0<Tbnds[0])|(T0>Tbnds[1])

        c = plt.contour(self.xc, self.yc,T0, [T00], colors='k')
        p = c.collections[0].get_paths()[0].vertices
        Np = p.shape[0]
        dx,dy = np.diff(p[:,0]),np.diff(p[:,1])
        s = np.hstack([0,np.cumsum((dx**2 + dy**2)**0.5)])
        s = ma.masked_array(s, mask)
        S = np.empty((self.Ny,self.Nx)).flatten()
        for i in mask[~mask]:
            k = anp.rgmin( (X.ravel()[i]-p[:,0])**2 + (Y.ravel()[i]-p[:,1])**2 )
            S[i] = s[k]
        S.shape = X.shape
        S = ma.masked_array(S, mask)
        Ns = 800
        S0 = np.linspace(0,s[-1],Ns)
        Ns = len(S0)
        qsum = np.zeros(Ns)
        area = np.zeros(Ns)
        for n in np.arange(Ns):
            fullmask = s.maks | (s>S0[n])
            area[n] = (1-fullmask).sum()
            qsum[n] = np.sum(ma.masked_array(q,fullmask))

        Dqsum = np.diff(qsum)
        Darea = np.diff(area)
        qS = Dqsum/Darea
        # interpolate nans
        mask = np.isnan(qS)
        qS[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), qS[~mask])
        return ma.masked_array(np.hstack([qS, 0]), S0>s.max())






# Gridding:
def cgrid_to_vgrid(q, neumann_bc=True):
    """Interpolate c-grid variable (cell center) hortizontally
    to the v-grid (cell southern boundary)"""
    # make sure to mask land properly
    # values inside land are zero, so don't try to interpolate through them
    # interpolate
    q_north = q.copy() # the current point
    q_south = q.copy()[:, np.r_[0, :Ny-1]] # the point below
    # apply masks
    if neumann_bc:
        if len(q.shape) < 3:
            #mask = (self.HFacC==0.).mean(axis=2)
            npad = ((0, 0), (1, 0))
        else:
            #mask = (self.HFacC==0.)
            npad = ((0, 0), (1, 0), (0, 0))
    out = 0.5 * (q_north + q_south)
    out = np.pad(out, pad_width=npad, mode='constant', constant_values=0)
    return out

def cgrid_to_wgrid(q, neumann_bc=True):
    """Interpolate a c-grid variable (cell center) vertically
    to the f-grid (cell top boundary)"""

    q_dn = q.copy() # the current point
    q_up = q.copy()[np.r_[0, 0:Nz-1]] # the point above
    # apply masks --> implement for Full topo runs
    if neumann_bc:
        if len(q.shape) < 3:
            # mask = (self.HFacC==0.).mean(axis=2)
            npad = ((1, 0), (0, 0))
        else:
            # mask = (self.HFacC==0.)
            npad = ((1, 0), (0, 0), (0, 0))
        out = 0.5 * (q_up + q_dn)
        out = np.pad(out, pad_width=npad, mode='constant', constant_values=0)
    return out

def moving_average(a, n=3):
    """ Docstring """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def ddT_Lgrid_centered(q, th):
    """Vertical second-order centered difference on the layers grid"""

    out = np.zeros(q.shape)
    # second order for interior
    out[1:-1, :] = ma.divide((q[1:-1] - q[2::]), (th[1:-1] + th[2::]))
    # first order for the top and bottom
    out[0, :] = ma.divide((q[0, :] - q[1, :]), th[0, :])
    out[-1, :] = ma.divide((q[-2, :] - q[-1, :]), th[-1, :])

    return out

def ddy_Lgrid_centered(q, dyg):
    """Merdional second-order centered difference on the layers grid"""

    dyg = np.tile(dyg, (len(q[:,1]), 1))
    out = np.zeros(q.shape)
    out[:, 1:-1] = ma.divide((q[:, 2::] - q[:, 1:-1]),
                             (dyg[:, 1:-1] + dyg[:, 2::]))
    out[:, 0] = ma.divide((q[:, 1] - q[:, 0]), dyg[:, 0])
    out[:, -1] = ma.divide((q[:, -1] - q[:, -2]), dyg[:, -1])

    #if isinstance(q, ma.masked_array):
    #    mask = q.mask
    #    mask[:,1:Ny-1] = q.mask[:,2:] | q.mask[:,:Ny-2]
     #   out = ma.masked_array(out, mask)

    return out

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
