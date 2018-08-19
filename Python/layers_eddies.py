#! /usr/bin/env ipython
################################################################
##                                                            ##
##        Script to create V'T' time average variable         ##
##                        Helen Burns                         ##
################################################################
from scipy.io import netcdf
import numpy as np
import os
import csv
import sys
import netCDF4
from numba import autojit
import glob
from my_functions import *
from pylab import *
###
OP = sys.argv[1]
x = os.getcwd()
lists = glob.glob(x+'/'+str(OP)+'/*alled.nc')
###
# Read in the timeaveraged files
print 'Loading Tav and grid fields...'
file2 = netCDF4.Dataset(x+'/'+str(OP)+"/Tav.nc",'r') 
file3 = netCDF4.Dataset(x+'/'+str(OP)+"/grid.nc",'r') 
Temp = file2.variables['THETA'][:].squeeze()
Yc = file2.variables['Y'][:]
X = file2.variables['X'][:]
Zp = file3.variables['Zp1'][:]
dz = Zp[0:len(Zp)-1]-Zp[1:len(Zp)]
Z = file3.variables['Z'][:]
V = file2.variables['VVEL'][:].squeeze()
W = file2.variables['WVEL'][:].squeeze()
Yp1 = file2.variables['Yp1'][:]

# Regrid V to center pointsVC = numba_regrid(V[:]*1)
npad = ((0, 1), (0, 0), (0, 0))
W = np.pad(W, pad_width=npad,
             mode='constant', constant_values=0)
WC = numba_regridz(W[:])
V = numba_regridy(V)
# LAYERS STYLE ITERPOLATION
# Split cells into 10
FineGridFact = 10
ZFF = np.zeros((300))
for kk in range(len(Z)-1):
    ZFF[10*kk:10*kk+10] = np.linspace(Z[kk],Z[kk+1],10)
ZFF[-10::] = np.linspace(Z[-1],Zp[-1],10)

TTavff = np.zeros((300,400,200))
VTavff = np.zeros((300,401,200))
WTavff = np.zeros((300,400,200))
for ii in range(len(X)):
    for jj in range(len(Yc)):
        TTavff[:,jj,ii]=interp(ZFF,Z[::-1], Temp[::-1,jj,ii])
        VTavff[:,jj,ii]=interp(ZFF,Z[::-1], V[::-1,jj,ii])
        WTavff[:,jj,ii]=interp(ZFF,Z[::-1], WC[::-1,jj,ii])
VTavff = numba_regridy(VTavff)
# Bin Temps layer
Rho = np.arange(-2,11,0.1) 
TTavffbin = zeros_like(TTavff)
for ii in range(len(X)):
    for jj in range(len(Yc)):
        for kk in range(len(ZFF)):
            TTavffbin[kk,jj,ii] = find_nearest(Rho,TTavff[kk,jj,ii])

# Load in each file and find V'T' and timeaverage it!
lists = glob.glob(x+'/'+str(OP)+'/*alled.nc')
VTprimetav20 = 0
VTbar20 = 0
WTprimetav20 = 0
WTbar20 = 0
total=len(lists)
# Possible memory error so print some info"

Rho = np.arange(-2,11,0.1)
Tpff = np.zeros((100,300,400,200))
Vpff = np.zeros((100,300,400,200))
print np.shape(Tpff)
def eddyfluxbin(i,VTprimetav20, VTbar20, VEL):
    file2 = netCDF4.Dataset(i,'r')
    Temp = file2.variables['THETA'][:]
    V = file2.variables[VEL][:]
    if VEL == 'WVEL':
        npad = ((0, 0), (0, 1), (0, 0), (0, 0))
        V = np.pad(V, pad_width=npad,
                       mode='constant', constant_values=0)
        Vc = numba_regridz(V)
    else:
        Vc = numba_regridy(V)
    for yr in range((6)):
        yr1 = 100*yr
        yr2 = yr1+100 
        Vc = Vc[yr1:yr2]
        Temp = Temp[yr1:yr2]
        for tt in range((100)):
            for ii in range(len(X)):
                for jj in range(len(Yc)):
                    Tpff[tt,:,jj,ii]=np.apply_along_axis(interp,
                                         0, ZFF, Z[::-1], Temp[tt,::-1,jj,ii])
                    Vpff[tt,:,jj,ii]=np.apply_along_axis(interp,
                                         0, ZFF, Z[::-1],Vc[tt,::-1,jj,ii])
                    for kk in range(len(ZFF)):
                        Tpff[tt,kk,jj,ii] = find_nearest(Rho,
                                                             Tpff[tt,kk,jj,ii])
        Vprime = Vpff-VTavff
        Tprime = Tpff[:]-TTavffbin[:]
        VTprime = Vprime*Tprime
        VTprimetav = np.mean(VTprime,axis=0)
        VTprimetav20 = (VTprimetav20 + VTprimetav/(6*total))
        VTbar = np.mean(Vpff*Tpff,axis=0)
        VTbar20 = VTbar20+VTbar/(6*total)    
    return VTprimetav20, VTbar20
numba_eddyfluxbin = autojit()(eddyfluxbin)
numba_eddyfluxbin.func_name = "eddyfluxbin"
# Here we go...
print 'Setup done... starting flux calc'
for file in lists:
    print file # Where am i?
    #VTprimetav20, VTbar20 = numba_eddyfluxbin(file, VTprimetav20, 
    #                                          VTbar20, 'VVEL')
    file2 = netCDF4.Dataset(file,'r')
    Temp = file2.variables['THETA'][:]
    V = file2.variables['VVEL'][:]
    Vc = numba_regridy(V)
    for yr in range((6)):
        yr1 = 100*yr
        yr2 = yr1+100 
        Vc = Vc[yr1:yr2]
        Temp = Temp[yr1:yr2]
        for tt in range((100)):
            for ii in range(len(X)):
                for jj in range(len(Yc)):
                    Tpff[tt,:,jj,ii]=np.apply_along_axis(interp,
                                      0, ZFF, Z[::-1], Temp[tt,::-1,jj,ii])
                    Vpff[tt,:,jj,ii]=np.apply_along_axis(interp,
                                         0, ZFF, Z[::-1],Vc[tt,::-1,jj,ii])
                    for kk in range(len(ZFF)):
                        Tpff[tt,kk,jj,ii] = find_nearest(Rho,
                                                        Tpff[tt,kk,jj,ii])
        Vprime = Vpff-VTavff
        Tprime = Tpff[:]-TTavffbin[:]
        VTprime = Vprime*Tprime
        VTprimetav = np.mean(VTprime,axis=0)
        VTprimetav20 = (VTprimetav20 + VTprimetav/(6*total))
        VTbar = np.mean(Vpff*Tpff,axis=0)
        VTbar20 = VTbar20+VTbar/(6*total)    

    print "done"
# Write to nc format
print "writing..."
f = netcdf.netcdf_file(x+'/'+str(OP)+'/VTprimebar.nc','w')
f.createDimension('X',len(VTprimetav20[1,1,:]))
f.createDimension('Y',len(VTprimetav20[1,:,1]))
f.createDimension('Z',len(VTprimetav20[:,1,1]))
VT = f.createVariable('VT','double',('ZFF','Y','X'))
VT[:] = VTprimetav20
f.close()
# Write to nc format                                            
f = netcdf.netcdf_file(x+'/'+str(OP)+'/VTbar.nc','w')
f.createDimension('X',len(VTbar20[1,1,:]))
f.createDimension('Y',len(VTbar[1,:,1]))
f.createDimension('Z',len(VTbar[:,1,1]))
VT = f.createVariable('VT','double',('ZFF','Y','X'))
VT[:] = VTbar20
f.close()
print "now for W..."
for file in lists:
    print file
    WVTprimetav20, WTbar20 = numba_eddyfluxbin(file, WTprimetav20,
                                               WTbar20, 'WVEL')    
    print "done"
f = netcdf.netcdf_file(x+'/'+str(OP)+'/WTbar.nc','w')
f.createDimension('X',len(WTbar20[1,1,:]))
f.createDimension('Y',len(WTbar[1,:,1]))
f.createDimension('Z',len(WTbar[:,1,1]))
WT = f.createVariable('WT','double',('ZFF','Y','X'))
WT[:] = WTbar20
f.close()
f = netcdf.netcdf_file(x+'/'+str(OP)+'/WTprimebar.nc','w')
f.createDimension('X',len(WTprimetav20[1,1,:]))
f.createDimension('Y',len(WTprimetav20[1,:,1]))
f.createDimension('Z',len(WTprimetav20[:,1,1]))
WT = f.createVariable('WT','double',('ZFF','Y','X'))
WT[:] = WTprimetav20
f.close()
