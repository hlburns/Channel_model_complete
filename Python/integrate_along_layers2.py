# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:18:07 2016

@author: hb1g13

Remapping into temperature space in Parallel. 

Start by running:

ipcluster start --profile=thalassa -n 10

this starts 10 processors.

This is a little more complex (even though simple) than internet examples
I am using my own modules to calculate things in parallel with in a for loop.
dv.apply only works on top level functions - not my nested modules and classes
therefore you must execute in parallel after passing on the necessary info to 
the processors. loading in data 6 times - check RAM use!


Write the results to netcdf4 so this will not be repeated!
"""


# Normal imports
from IPython import parallel
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import netCDF4
import numpy.ma as ma
from pylab import *
sys.path.append('/noc/users/hb1g13/Python/python_functions/')
import SG as SG
import layers_calc_numba
sys.path.append('/noc/users/hb1g13/Python/python_functions/MITgcmUtils/')
import utils

# Set up processors
rc = parallel.Client('/noc/users/hb1g13/.ipython/profile_maelstrom/security/ipcontroller-client.json')
dv = rc[:]
rc.ids

# Now each processor needs to know where my modules are:
dv.execute('import sys')
dv.execute('sys.path.append("/noc/users/hb1g13/Python/python_functions/")')
dv.execute('import layers_calc_numba')
dv.execute('sys.path.append("/noc/users/hb1g13/Python/python_functions/MITgcmUtils/")')
dv.execute('import utils')

# Some parameteres to ensure right files are picked up:
Full = 'N'  # 9 Pannels isn't ideal for presentations N option give 4 plots
if Full == 'N':
    tau = ['3', '300', '3000', 'Closed']
elif Full == 'Extremes':
    tau = ['3', 'Closed']
else:
    tau = ['3', '10', '30', '100', '300',
           '1000', '3000', '10000', 'Closed']
Figletter = ['a) ','b) ','c) ','d) ','e)','f)','g)','h)','j)']
# Path root
x = '/noc/msm/scratch/students/hb1g13/Mobilis'
# Now Make file structure
check = 0
runs = []
for i in range(len(tau)):
    flist = x+'/'+str(tau[i])+'daynokpp/PSI.nc'
    if not os.path.exists(flist):
        print ' WARNING: '+flist+' does not exist! (skipping this tau...)'
        check += 0
    else:
        check += 1
        runs.append(i)
Runs=np.array(runs)


for i in range(len(Runs)):
    i = 3
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    # Calculate cartesian diabatic eddies 
    # PLOT PANNELS
    CellVol = c.rac * np.tile(c.dzf, (c.Nx, c.Ny, 1)).T
    g = layers_calc_numba.LayersComputer(c)
    # Set up on processors:
    dv['fname'] = fname
    dv.execute('c = utils.ChannelSetup(output_dir=str(fname))',block='True')
    dv.execute('g = layers_calc_numba.LayersComputer(c)',block='True')
    # load V, W, T bar
    # put everything on the C Grid
    dv['VT'] = c.vgrid_to_cgrid(c.mnc('Tav_VT.nc', 'VVELTH'))
    WT = (c.mnc('Tav_VT.nc', 'WVELTH'))
    dv['V'] = c.vgrid_to_cgrid(c.mnc('Tav.nc', 'VVEL'))
    W = (c.mnc('Tav.nc', 'WVEL'))
    dv['T'] = c.mnc('Tav.nc', 'THETA')
    npad = ((0, 1), (0, 0), (0, 0))
    dv['W'] = c.wgrid_to_cgrid(np.pad(W, pad_width=npad, mode='constant', constant_values=0))
    dv['WT'] = c.wgrid_to_cgrid(np.pad(WT, pad_width=npad, mode='constant', constant_values=0))
    
    # Now remap in parallel
    dv.execute('A_local=g.interp_to_g(VT,T)',block=True)
    VT_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(WT,T)',block=True)
    WT_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(V,T)',block=True)
    V_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(W,T)',block=True)
    W_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(W,T)',block=True)
    th = dv.gather('A_local').get()[1]
    dv.execute('A_local=g.interp_to_g(T,T)',block=True)
    T_l  = dv.gather('A_local').get()[0]

    dv['Ty'] = c.ddy_cgrid_centered(c.mnc('Tav.nc', 'THETA'))
    dv['Tz'] = c.ddz_cgrid_centered(c.mnc('Tav.nc', 'THETA'))
    dv.execute('A_local=g.interp_to_g(Ty,T)',block=True)
    Ty_l  = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(Tz,T)',block=True)
    Tz_l  = dv.gather('A_local').get()[0]
    print W_l.shape
    # ADD in netcdf write #########
    f = netCDF4.Dataset(str(fname)+'/layer_int2/Remapped.nc','w')
    f.createDimension('T',g.ng)
    f.createDimension('Y',c.Ny)
    f.createDimension('X',c.Nx)
    h2=f.createVariable('W_l','float',('T','Y','X'))
    h2[:] = W_l
    h2.standard_name = 'W remamped to T space'
    h3=f.createVariable('th','float',('T','Y','X'))
    h3[:] = th
    h3.standard_name = 'Layer thickness'
    h4=f.createVariable('V_l','float',('T','Y','X'))
    h4[:] = V_l
    h4.standard_name = 'V remamped to T space'
    h5=f.createVariable('VT_l','float',('T','Y','X'))
    h5[:] = VT_l
    h5.standard_name = 'VTremamped to T space'
    h6=f.createVariable('WT_l','float',('T','Y','X'))
    h6[:] = WT_l
    h6.standard_name = 'WT remamped to T space'
    h7=f.createVariable('T_l','float',('T','Y','X'))
    h7[:] = T_l
    h7.standard_name = 'T remamped to T space'
    h8=f.createVariable('Tz_l','float',('T','Y','X'))
    h8[:] = Tz_l
    h8.standard_name = 'Tz remamped to T space'
    h9=f.createVariable('Ty_l','float',('T','Y','X'))
    h9[:] = Ty_l
    h9.standard_name = 'Ty remamped to T space'
    f.close()

    #Clear vars
    dv['Ty_l']=0
    dv['Tz_l']=0
    dv['V_l']=0
    dv['W_l']=0
    dv['WT_l']=0
    dv['VT_l']=0
    # Eddy components                     
    VT = (c.mnc('Tav_VT.nc','VVELTH'))
    WT = (c.mnc('Tav_VT.nc','WVELTH'))
    Tv = utils.cgrid_to_vgrid(c.mnc('Tav.nc','THETA'))
    Tw = utils.cgrid_to_wgrid(c.mnc('Tav.nc','THETA'))
    V = (c.mnc('Tav.nc','VVEL'))
    W = (c.mnc('Tav.nc','WVEL'))
    npad = ((1, 0), (0, 0), (0, 0))
    W = np.pad(W, pad_width=npad, mode='constant', constant_values=0)
    WT = np.pad(WT, pad_width=npad, mode='constant', constant_values=0)
    dv['VTbar'] = V*Tv
    dv['WTbar'] = W*Tw
    dv['VpTp'] = c.vgrid_to_cgrid(VT - V*Tv)
    dv['WpTp'] = c.wgrid_to_cgrid(WT -  W*Tw)
    dv.execute('A_local=g.interp_to_g(VpTp,T)',block=True)
    VpTp_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(WpTp,T)',block=True)
    WpTp_l = dv.gather('A_local').get()[0]
    f = netCDF4.Dataset(str(fname)+'/layer_int2/Eddy_L.nc','w')
    f.createDimension('T',g.ng)
    f.createDimension('Y',c.Ny)
    f.createDimension('X',c.Nx)
    h2=f.createVariable('WpTp_l','float',('T','Y','X'))
    h2[:] = WpTp_l
    h2.standard_name = 'WT(y,T) - Wbar(y,T)Tbar'
    h3=f.createVariable('VpTp_l','float',('T','Y','X'))
    h3[:] = VpTp_l
    h3.standard_name = 'VT(y,T) - Vbar(y,T)Tbar'
    f.close()
   
    
    # Extra mean remapped to T space
    dv.execute('A_local=g.interp_to_g(WTbar,T)',block=True)
    WbarTbar_l = dv.gather('A_local').get()[0]
    dv.execute('A_local=g.interp_to_g(VTbar,T)',block=True)
    VbarTbar_l = dv.gather('A_local').get()[0]
    
    f = netCDF4.Dataset(str(fname)+'/layer_int2/Remapped_bar.nc','w')
    f.createDimension('T',g.ng)
    f.createDimension('Y',c.Ny)
    f.createDimension('X',c.Nx)
    h2=f.createVariable('WbarTbar_l','float',('T','Y','X'))
    h2[:] = WbarTbar_l
    h2.standard_name = 'WbarTbar(y,T)'
    h3=f.createVariable('VbarTbar_l','float',('T','Y','X'))
    h3[:] = VbarTbar_l
    h3.standard_name = 'VbarTbar(y, T)'
    f.close()