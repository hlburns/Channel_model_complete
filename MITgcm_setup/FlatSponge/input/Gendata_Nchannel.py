
# coding: utf-8

## Generate model inputs

### This will set up the forcing for Abernathey Style Forcing 

# In[94]:

from scipy.io import netcdf
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


# When writing in python it is very important to note reverse dimensions!!
# MITgcm assumes column major order (as does matlab) Python, uses row major order.
# Mosty it's fine just to write straight to binary, but to absolutely specific of the format for MITgcm the WriteFile fuction (taken from the MITgcm csv gendata.py):

# In[95]:

# Use writeFile to write files in the correct format!
sys.path.append('/noc/users/hb1g13/Python/python_functions/')
from Writebin import *


### Decide parameters:

                Resolution 
Depth
Domain
Boundary Condition
Topography
Forcing
                
# In[96]:

Topo="Ridge" #Please Choose ridge, slope or flat
Wind="Standard" # Sine bell 0.2N/m$^2$
Heat="nonetQ" # Please Choose Abernathey or nonetQ
BC="Diffusion" # Please Choose Sponge or Diffusion
Name="Full_diffusion" # Give Experiment Name


# In[97]:

# Adjust accordingly
Res=5000
Ly=2000e3
Lx=4000e3 #Full domain = 4000km otherwise 1000km
H=2985 # Diffusion = 3800m, Sponge = 2985m
nz=30 # Diffusion = 24 level, Sponge= 30 levels


# In[98]:

x="/noc/users/hb1g13/MITgcm/Mobilis/"+Name+"/input/" 
os.chdir(x)


### Set up grid:

# In[99]:

#Dimensions
nx=np.round(Lx/Res)
ny=np.round(Ly/Res)
dx=np.ones(nx)*Res
dy=np.ones(ny)*Res
#Write binary output
writeFile('delY',dy)
writeFile('delX',dx)
# Create c-grid with grid points in dead center
x=(np.cumsum(dx)-dx/2)-Lx/2
y=(np.cumsum(dy)-dy/2)-Ly/2
[Y, X]=np.meshgrid(y,x) 


### Now Create topography:

# Start with flat, then add slope and ridges

# In[100]:

h= -H*np.ones((nx,ny)) # Flat bottom
if Topo=="ridge":#2500 and 2000 for full depth
         h= h+(1500 + 300*np.sin(10*pi*Y/Ly) + 400*np.sin(8*pi*Y/Ly)+ 300*sin(25*pi*Y/Ly) )*(1/np.cosh(((X)-0.2*Y+3e5)/1.2e5))
         h= h+((1000 + 600*np.sin(11*pi*Y/Ly) + 300*np.sin(7*pi*Y/Ly)+ 500*sin(21*pi*Y/Ly) )*(1/np.cosh(((X)+0.1*Y+1.5e6)/1.2e5)))
if Topo=="slope" or Topo=="ridge":
    for i in range(int(nx)):
      slope= np.transpose(H*(np.divide((Y[i,0:round(0.2*ny)]-Y[i,0]),(Y[i,0]-Y[i,round(0.2*ny)]))))
      h2=h[:,0:round(0.2*ny)]
      h[:,0:round(0.2*ny)]=np.maximum(slope,h2)
# Close both ends
h[:,0]=0
h[:,-1]=0
# Write to binary
writeFile('topog',np.transpose(h))


# In[101]:

if Topo=="flat" or Topo=="slope":
    plt.plot(y/1000,h[nx/2,:])
    plt.title('Topography')
    plt.ylabel('Depth (m)')
    plt.xlabel('Y (km)')
if Topo=='ridge':
    plt.contourf(x/1000,y/1000,np.transpose(h),30)
    cb=plt.colorbar()
    plt.title('Topography')
    plt.ylabel('Y (km)')
    plt.xlabel('X (km)')
    cb.set_label('Depth (m)')
    


### Surface Heat Forcing

# Now for the surface heat forcing:
# Must have bouyancy gain in the south and bouyancy loss over maximum wind sress to allow overturning
# 
# 
# $Q=-Q_{0}\,cos(\frac{3\pi y}{Ly})\quad \quad \quad \quad y \le \frac{5Ly}{6}$

# In[102]:

#Itegrate!
Q=10*(np.sin(Y*(3*pi/Ly)))
Q[:,ny-(np.round(ny/6)):ny]=0
if Heat=="nonetQ":
   Q[:,np.round(ny/2):ny]=0.5*Q[:,np.round(ny/2):ny]
# Write to binary
writeFile('Qsurface',np.transpose(Q))
# netcdf check
f=netcdf.netcdf_file('Qsurface.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
Q2=f.createVariable('Q','float',('X','Y'))
Q2[:]=Q
f.close()


# In[103]:

plt.plot(y/1000,Q[100,:])
plt.title('Surface Heat Flux $W/m^2$')
plt.ylabel('Heat Flux ($W/m^2$)')
plt.xlabel('Meridional Distance (m)')


### Windstress

# Plus the Windress with $\tau_o$ set to $0.2Nm^-2$
# 
# $\tau_s(y)=\tau_0 sin(\frac{\pi y}{Ly})$

# In[104]:

tau=0.2*((np.sin((Y+Ly/2)*(pi/Ly)))) #Y is centred at 0 so put that back!
# Write to binary
writeFile('Wind',np.transpose(tau))
# netcdf check
f=netcdf.netcdf_file('Wind.nc','w')
f.createDimension('Xp1',nx+1)
f.createDimension('Y',ny)
tau3=np.zeros((ny,nx+1))
tau3[:,1:]=np.transpose(tau)
tau2=f.createVariable('tau','double',('Xp1','Y'))
tau2[:]=np.transpose(tau3)
f.close()


# In[105]:

plt.plot(y/1000,tau[100,:])
plt.title('Surface Wind Stress $N/m^2$')
plt.ylabel('$\tau$ ($N/m^2$)')
plt.xlabel('Meridional Distance (m)')


### Generate Sponge

                Now creat a Sponge mask and a reference profile to relax to:
                
# In[106]:

#Parameters
N=1e3 # Natural stratification
deltaT=8
Tref=np.zeros(nz)
#Create depth array:
a=5,22.5,60
b=np.linspace(135,2535,25)
c=2685,2885
z=np.concatenate([a,b,c])


# \begin{equation*} T^*(z)=\Delta T\frac{(e^{z/h}-e^{-H/h})}{1-e^{-H/h}} \end{equation*}

# In[107]:

Tref = deltaT*(exp(-z/N)-exp(-H/N))/(1-exp(-H/N))


# In[108]:

plt.plot(Tref,z)
plt.gca().invert_yaxis()
plt.title('Temperature Profile')
plt.ylabel('Depth (m)')
plt.xlabel('Temperature $^oC$')


# In[109]:

#Make a 3D array of it
T=np.ones((nz,ny,nx))
Temp_field=np.zeros(np.shape(T))
for i in range(int(nx)):
    for j in range(int(ny)):
        Temp_field[:,j,i]=np.multiply(Tref,T[:,j,i])


# In[110]:

Tnew = transpose(tile(Temp_field.mean(axis=2),(nx,1,1)),[1,2,0])
Tnew[:,-1] = Tnew[:,-2]
Tnew = Tnew + 1e-3 * (np.random.random((nz,ny,nx)) - 0.5)


# In[111]:

# Write to binary
writeFile('T_Sponge',Temp_field)
writeFile('T.init',Tnew)
# netcdf check
f=netcdf.netcdf_file('TSponge.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Temp=f.createVariable('Temp','double',('Z','Y','X'))
Temp[:]=Temp_field
f.close()


# In[113]:

#Make 3D mask
#Must vary between 0 (no Relaxation) and 1 (full relaxtion)
#I have gone for a parabolic decay in x and linear decay in z (from playing around)
msk=np.zeros(np.shape(T))
for k in range(0,len(z)):
    for i in range(len(x)):  
        msk[k,ny-20:ny,i]=((np.divide((Y[i,ny-21:ny-1]-Y[i,ny-21]),(Y[i,ny-1]-Y[i,ny-21])))**2)        *(z[k]/H)    
# Write to binary
writeFile('T.msk',msk)
# netcdf check
f=netcdf.netcdf_file('Mask.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Mask=f.createVariable('Mask','double',('Z','Y','X'))
Mask[:]=(msk)
f.close()


# In[117]:

plt.contourf(y/1000,z,msk[:,:,100],24,cm=cm.Spectral)
cbar = plt.colorbar()
plt.gca().invert_yaxis()
plt.title('Mask Matrix')
plt.ylabel('Depth (m)')
plt.xlabel('Meridional Distance (km)')


# In[118]:

if BC=="Diffusion":
        #Background
        diffusi=(1e-5)*np.ones((nz,ny,nx))
        # Linear ramp
        for k in range(0,nz):
           for i in range(0,int(nx)):
               diffusi[k,ny-20:ny,i]=0.00025+500*(np.divide((Y[i,ny-21:ny-1]-Y[i,ny-21]),                                                            (Y[i,ny-1]-Y[i,ny-21])))                **2*diffusi[k,ny-21:ny-1,i]#*(1-exp(-z[k]/(N))-exp(-(H)/(N)))/(1-exp(-(H)/(N)))
               # Enhance at the surface
        for k in range(0,3):
            for i in range(0,int(nx)):
                diffusi[k,:,i]=np.maximum(0.002*((z[nz-1-k]/H)**2)                                          *(1-np.divide(2*abs(Y[i,:]),(2*Ly))),diffusi[k,:,i])
        # Write to binary
        writeFile('diffusi.bin',diffusi)
        # netcdf check
        f=netcdf.netcdf_file('diffusi.nc','w')
        f.createDimension('Z',nz)
        f.createDimension('Y',ny)
        f.createDimension('X',nx)
        Diff=f.createVariable('Diffusi','double',('Z','Y','X'))
        Diff[:]=diffusi
        f.close()
        plt.contourf(y/1000,z,diffusi[:,:,150],24,cm=cm.Spectral)
        cbar = plt.colorbar()
        plt.gca().invert_yaxis()
        


# In[ ]:



