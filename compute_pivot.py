import numpy as np
from mpmath import mp

mp.dps = 500

def compute_pivot(etajTy,z_interval,tn_mu,tn_sigma):
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        if each_interval != None:
            nu_minus = each_interval[0]
            nu_plus = each_interval[1]

            denominator += (mp.ncdf((nu_plus - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))
            #numerator += (mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))

            if etajTy >= nu_plus:
                numerator += (mp.ncdf((nu_plus - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))
            elif (etajTy >= nu_minus) and (etajTy < nu_plus):
                numerator += (mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))
    
    if denominator != 0:
        return float(numerator/denominator)
    else:
        pass

def compute_pivot_naive(etaj,etajTy,cov,tn_mu,Vplus,Vminus):
    numerator = 0
    denominator = 0
    
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T,cov),etaj))[0][0]

    denominator += (mp.ncdf((Vplus - tn_mu)/tn_sigma) - mp.ncdf((Vminus - tn_mu)/tn_sigma))
    
    if etajTy >= Vplus:
        numerator += (mp.ncdf((Vplus - tn_mu)/tn_sigma) - mp.ncdf((Vminus- tn_mu)/tn_sigma))
    elif (etajTy >= Vminus) and (etajTy < Vplus):
        numerator += (mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((Vminus - tn_mu)/tn_sigma))
        
    if denominator != 0:
        return float(numerator/denominator)

    
    
    
        


