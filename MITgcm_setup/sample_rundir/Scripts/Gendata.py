import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
import math
from pylab import *
from IPython.display import display, Math, Latex
from numba import jit
import glob
# Use writeFile to write files in the correct format!
from Writebin import *

Topo = "flat"  # Please Choose ridge, slope or flat
Wind = "Standard"  # Sine bell 0.2N/m$^2$
Heat = "nonetQ"  # Please Choose Abernathey or nonetQ
BC = "Sponge"  # Please Choose Sponge or Diffusion
Name = "500m"  # Give Experiment Name
N = 0.5e3 # Set stratification
deltaT0 = 8.0 # top to bottom temp dif
N0 = 1.0e3 # Natural stratification
Res = 5000
Ly = 2000e3
Lx = 1000e3 #Full domain = 4000km otherwise 1000km
H =  3000 #  Sponge = 3000m
nz = 30 #  Sponge= 30 levels
print Topo, ' ', BC, ' ', Name , ' created'
#Dimensions
nx = np.int(np.round(Lx/Res))
ny = np.int(np.round(Ly/Res))
dx = np.ones(nx)*Res
dy = np.ones(ny)*Res
#Write binary output
writeFile('delY',dy)
writeFile('delX',dx)
# Create c-grid with grid points in dead center
x = (np.cumsum(dx)-dx/2)-Lx/2
y = (np.cumsum(dy)-dy/2)-Ly/2
[Y, X] = np.meshgrid(y, x) 
h= -H*np.ones((nx,ny)) # Flat bottom
if Topo == "ridge":#2500 and 2000 for full depth
        h= h+((2500 + 300*np.sin(10*pi*Y/Ly) + 400*np.sin(8*pi*Y/Ly)
               + 300*sin(25*pi*Y/Ly) )*(1/np.cosh(((X)-0.2*Y+3e5)/1.2e5)))
        h= h+((2000 + 600*np.sin(11*pi*Y/Ly) + 300*np.sin(7*pi*Y/Ly)+
               500*sin(21*pi*Y/Ly) )*(1/np.cosh(((X)+0.1*Y+1.5e6)/1.2e5)))
if Topo == "slope" or Topo=="ridge":
    for i in range(int(nx)):
        slope= np.transpose(H*(np.divide((Y[i,0:round(0.2*ny)]
                                          -Y[i,0]),(Y[i,0]-Y[i,round(0.2*ny)]))))
        h2 = h[:,0:round(0.2*ny)]
        h[:,0:round(0.2*ny)]=np.maximum(slope,h2)
# Close both ends
h[:,0] = 0
h[:,-1] = 0
# Write to binary
writeFile('topog',np.transpose(h))
# netcdf check
f = netCDF4.Dataset('topog.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
h2=f.createVariable('h','float',('X','Y'))
h2[:] = h
f.close()

#MITgcm opposite way round
Q_0 = 10
Q = Q_0*(np.sin(Y*(3*pi/Ly)))
Q[:,ny-(np.round(ny/6)):ny]=0
x1 = (np.cumsum(dx)-dx/2)
y1 = (np.cumsum(dy)-dy/2)
[Y1, X1] = np.meshgrid(y1, x1) 
if Heat=="nonetQ":
    Q[:,0:int(5*ny/36)]=Q_0*(np.cos(Y1[:,0:int(5*ny/36)]*(pi/(Y1[1,int(10*ny/36)]))))
    Q[:,int(5*ny/36):int(20*ny/36)]= -Q_0*(np.sin((Y1[:,int(5*ny/36):int(20*ny/36)]
                                                 -Y1[1,int(5*ny/36)])*(pi/(Y1[1,int(15*ny/36)]))))
    Q[:,int(20*ny/36):int(30*ny/36)]= Q_0*(np.sin(Y1[:,0:int(10*ny/36)]*(pi/(Y1[1,int(10*ny/36)]))))
    Q[:,0:int(30*ny/36)] = Q[:,0:int(30*ny/36)]+(-sum(Q)/(5*ny*nx/6)) 
# Write to binary
writeFile('Qsurface',np.transpose(Q))
# netcdf check
f = netCDF4.Dataset('Qsurface.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
Q2=f.createVariable('Q','float',('X','Y'))
Q2[:]=Q
f.close()

tau = 0.2*((np.sin((Y+Ly/2)*(pi/Ly)))) #Y is centred at 0 so put that back!
if BC=='Diffusion':
    Taunew = tau + 2e-3 * (np.random.random((nx,ny)) - 0.5)
    tau=Taunew
# Write to binary
writeFile('Wind',np.transpose(tau))
# netcdf check
f=netCDF4.Dataset('Wind.nc','w')
f.createDimension('Xp1',nx+1)
f.createDimension('Y',ny)
tau3=np.zeros((ny,nx+1))
tau3[:,1:]=np.transpose(tau)
tau2=f.createVariable('tau','double',('Xp1','Y'))
tau2[:]=np.transpose(tau3)
f.close()

#Parameters
deltaT = deltaT0 
Tref = np.zeros(nz)
zp = np.linspace(1,0,31)
zp = H+H*(tanh(-pi*zp))
zp = zp - zp[0]-(zp[1]-zp[0])
zp = np.round(zp,2)
dz = zp[0:-1] - zp[1::]
z = zeros((nz))
z[1::] = (zp[1:-1]+zp[2::])/2
z[0]= -zp[0]/2

Tref = deltaT*(exp(-z/N)-exp(-H/N))/(1-exp(-H/N))+((N-N0)/N0)
if N > H:
    deltaT = 8 -((H-N0)/N0)
    Tref = deltaT*(exp(-z/N)-exp(-H/N))/(1-exp(-H/N))+1+((H-N0)/N0)
print 'max: ', np.max(Tref), '\n min: ', np.min(Tref)
#Make a 3D array of it
T=np.ones((nz,ny,nx))
Temp_field=np.zeros(np.shape(T))
for i in range(int(nx)):
    for j in range(int(ny)):
        Temp_field[:,j,i]=np.multiply(Tref,T[:,j,i])

Tnew = transpose(tile(Temp_field.mean(axis=2),(nx,1,1)),[1,2,0])
Tnew[:,-1] = Tnew[:,-2]
#Maybe add more 
if BC=='Diffusion':
    Tnew = Tnew + 2e-3 * (np.random.random((nz,ny,nx)) - 0.5)
else:
    Tnew = Tnew + 1e-3 * (np.random.random((nz,ny,nx)) - 0.5)

# Write to binary
writeFile('T_Sponge',Temp_field)
writeFile('T.init',Tnew)
# netcdf check
f=netCDF4.Dataset('TSponge.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Temp=f.createVariable('Temp','double',('Z','Y','X'))
Temp[:]=Temp_field
f.close()

#Make 3D mask
#Must vary between 0 (no Relaxation) and 1 (full relaxtion)
#I have gone for a parabolic decay in x and linear decay in z (from playing around)
msk=np.zeros(np.shape(T))
for k in range(0,len(z)):
    for i in range(len(x)):  
        msk[k,ny-20:ny,i]=((np.divide((Y[i,ny-21:ny-1]-Y[i,ny-21]),(Y[i,ny-1]-Y[i,ny-21]))))
# Write to binary
writeFile('T.msk',msk)
# netcdf check
f=netCDF4.Dataset('Mask.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Mask=f.createVariable('Mask','double',('Z','Y','X'))
Mask[:]=(msk)
f.close()
