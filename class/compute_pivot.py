import numpy as np
from mpmath import mp

mp.dps = 500

def compute_pivot(etajTy,z_interval,tn_mu,tn_sigma):
    numerator = 0
    denominator = 0

    z_interval = convert_to_nested_list(z_interval)
    
    for nu_minus,nu_plus in z_interval:
        denominator += (mp.ncdf((nu_plus - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))

        if etajTy >= nu_plus:
            numerator += (mp.ncdf((nu_plus - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))
        elif (etajTy >= nu_minus) and (etajTy < nu_plus):
            numerator += (mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((nu_minus - tn_mu)/tn_sigma))
    
    if denominator != 0:
        return float(numerator/denominator)
    else:
        pass

def convert_to_nested_list(interval):
    if isinstance(interval[0], list):  # 既に二重のリストである場合
        return interval
    else:  # 単一のリストである場合
        return [interval]