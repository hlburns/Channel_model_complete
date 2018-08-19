#! /usr/env ipython
from numba import autojit
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy import interpolate

def movingaverage(values,window):
    '''Calculate Moving Average'''
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def find_nearest(array,value):
    '''Tell me where the nearest value - used in remapping '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def regridz(q):
    """Interpolate a w-grid variable (cell top) vertically
       down to the c-grid (cell center)"""
    Nz = len(q)-1
    q_up = q.copy()[0:-1] # the current point
    q_dn = q.copy()[np.r_[1:Nz,Nz-1]] # the point below
    # otherwise the point will be
    # naturally interpolated assuming w=0 at the
    # boundary (doesn't account for partial cells!)

    return 0.5 * (q_dn + q_up)
                                                                 
#def regridz(Variable):
#    '''Regrids Zp1 to Z (Time averaged) '''
#    if len(np.shape(Variable))==4:
#        Vc=(Variable[:,0:-1]+Variable[:,1::])/2
#    else:
#        Vc=(Variable[0:-1]+Variable[1::])/2
#    return Vc

def regridy(Variable):
    '''Regrids Yp1 to Y (Time averaged)
       Future : check if 4 dimentions for 
       non timeaveraged?
    '''
    if len(np.shape(Variable))==4:
        Vc=(Variable[:,:,0:-1]+Variable[:,:,1::])/2
    else:
        Vc=(Variable[:,0:-1]+Variable[:,1::])/2
    return Vc

def regridx(Variable):
    '''Regrids Xp1 to X (Time averaged)
       Future : check if 4 dimentions for 
       non timeaveraged?
    '''
    if len(np.shape(Variable))==4:
        Vc=(Variable[:,:,:,0:-1]+Variable[:,:,:,1::])/2
    else:
        Vc=(Variable[:,:,0:-1]+Variable[:,:,1::])/2
    return Vc

# Numba them
numba_regridy = autojit()(regridy)
numba_regridy.func_name = "numba_regridy"
numba_regridz = autojit()(regridz)
numba_regridz.func_name = "numba_regridz"
numba_regridx = autojit()(regridx)
numba_regridx.func_name = "numba_regridx"

def maxmag(var):
    m1=np.max(var,axis=0)
    m2=np.min(var,axis=0)
    if m1<abs(m2):
        val=m2
    else:
        val=m1
        
    return(val)

def mld(Tfield,Z,DZ,criterion=1):
    '''Function to calculate meridional
       mixed layer depth
       Uses Time averaged Temp field,
       Depth steps uneven use Zmatrix or set
       Even steps
       
       Criterion can be set to establish 
       mixed layer depth

       Returns hml(y) 
     
    '''
    if len(np.shape(Tfield))==3:
        Tfield=np.mean(Tfield,axis=2)
    TTavz, TTavy, = np.gradient(Tfield)
    TTavz=TTavz
    hml=np.zeros(np.shape(Tfield[1,:]))
    Ind=np.ones(np.shape(Tfield[1,:]))
    for i in range(len(Tfield[1,:])):
        I=find_nearest(TTavz[:,i], criterion)
        b=np.nonzero(np.cumsum(TTavz[:,i],axis=0)<criterion)[0]
        if len(b)==0:
            b=[0]
        Ind[i]=b[0]
        hml[i]=Z[b[0]]
    return [hml, Ind]

def MOC(Vtav,dz,dx=5000):
    '''MOC '''
    Vtav=np.nansum(Vtav*dx,axis = 2)
    # No more super slow forloop!           
    psi2=np.apply_along_axis(np.multiply,0,Vtav,dz)
    psi3=np.cumsum(-psi2[::-1,:],axis=0)
    npad = ((0,1), (0,0))
    psi4 = np.pad(psi3, pad_width=npad, mode='constant', constant_values=0)
    Psi=psi4/10**6
    return Psi

def ROC(lvrho,dx=5000):
    '''Calc ROC in Temp layers
       dx is set to 5000 by default
    '''
    VT=np.sum(lvrho*dx,axis=3) #integrate Vdx along x
    VTfdz=np.cumsum(VT[:,::-1,:],axis=1) #sum up the water column
    psi=np.mean(VTfdz[:,::-1,:],axis=0)/10**6 #Time average and put into Sv and put back in right ord
    return psi

def ROCY(lvrho,Z,Rho,Tfield,dx=5000,lvls=168,All='n'):
    ''' '''
    if len(np.shape(Tfield))==3:
        Tfield=np.mean(Tfield,axis=2)
    psi=ROC(lvrho)
    Psi=numba_regridy(psi)                                                                          
    #Expand temperature co-ordinates (30 lvls to 168 lvls)                                 
    Z2=interp1d(Z,Z,axis=0)
    Znew=np.linspace(int(Z[0]),int(Z[-1]),lvls)
    Zexp=Z2(Znew)
    T2=interp1d(Z,Tfield,axis=0)
    Tnew=np.linspace(int(Z[0]),int(Z[-1]),lvls)
    Texp=T2(Tnew)
    #Mixed layer depth
    [hmlexp, Iexp] = mld(Texp,Zexp,np.diff(-Zexp)[0])
    R2=interp1d(Rho,Rho,axis=0)
    Rnew=np.linspace(Rho[0],Rho[-1],lvls)
    Rexp=R2(Rnew)
    P2=interp1d(Rho,Psi,axis=0)
    Pnew=np.linspace(Rho[0],Rho[-1],lvls)
    Pexp=P2(Pnew)
    Psimap=np.zeros(np.shape(Texp))
    for g in range(len(Tfield[1,:])):
        for k in range(len(Zexp)):
            DT=Texp[k,g]
            if np.isnan(DT):
               Psimap[k,g]=np.nan
            else:
               P=Pexp[:,g]
               I=find_nearest(Rexp, DT)
               b=np.nonzero(Rexp==I)[0][0]
               Psimap[k,g]=P[b]
    if All=='y':
        return [Psimap, Zexp, Texp, hmlexp, Iexp, Pexp, Rexp]
    else: 
        return [Psimap, Zexp, Texp] 
numba_ROCY = autojit()(ROCY)
numba_ROCY.func_name = "numba_ROCY"

def Var_ml(Var,hml,Z):
    '''Find Variable value at mixed layer depth'''
    Varml=np.zeros(np.shape(Var[1,:]))
    mli=np.zeros(np.shape(Var[1,:]))
    for i in range(len(Varml)):
        I=find_nearest(Z, hml[i])
        b=np.nonzero(Z==I)[0]
        if b<1:
            b=0
        else:
            b=b
        mli[i]=int(b)
        Varml[i] = Var[int(b),i]
    return Varml, mli

def Var_mlav(Var,I):
    '''Find Variable average over mixed layer depth'''
    Varml=np.zeros(np.shape(Var[1,:]))
    for i in range(len(Varml)):
        Varml[i] = np.mean(Var[0:I[i],i])
    return Varml
def Var_mlint(Var,I):
    '''Find Variable average over mixed layer depth'''
    Varml=np.zeros(np.shape(Var[1,:]))
    for i in range(len(Varml)):
        Varml[i] = np.nansum(Var[0:I[i],i])
    return Varml

numba_Var_ml = autojit()(Var_ml)
numba_Var_ml.func_name = "numba_Var_ml"
numba_Var_mlav = autojit()(Var_mlav)
numba_Var_mlav.func_name = "numba_Var_mlav"
numba_Var_mlint = autojit()(Var_mlint)
numba_Var_mlint.func_name = "numba_Var_mlint"

def fill_holes(Var):
    sh = Var.shape
    Var_new = Var
    # Cant fill in masked values need to remove mask
    for kk in range(sh[0]):
        indx = np.where(Var[kk,:].mask==False)[0]
        if len(indx)==0:
            continue
        start = np.where(Var[kk,:].mask==False)[0][0]
        space=np.diff(indx) 
        for jj in range(len(indx)-1):
            if jj < start or space[jj] == 1 :
               continue
            Var_new[kk,indx[jj]] = ma.mean(Var[kk,indx[jj]:indx[jj+1]])
    return Var_new
