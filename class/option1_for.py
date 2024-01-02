import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from scipy import stats
from tqdm import tqdm
from sicore import tn_cdf_mpmath

import common_func
import missing_imputation
import feature_selection
import feature_selection_si
import outlier_removal
import outlier_removal_si
import compute_pivot

mp.dps = 500
eps = 1e-7

p_value_list = []

for each_seed in tqdm(range(0,1000)):

    n = 100
    d = 10
    
    # ハイパラの設定
    lamda_cook = 3
    k_ms = 5
    k_sfs = 3
    lamda_lasso = 0.08

    # 初期条件
    M_obs = list(range(d))
    O_obs = []

    # データの生成
    sigma = 1
    beta_vec = np.zeros(d)
    mu_vec = np.zeros((n))
    s_list = [1,2,3,4,5]
    mu_vec[s_list] = 4

    X,y,true_y = common_func.data_generation_seed(each_seed,n,d,beta_vec,mu_vec)
    
    # 欠損値を加える(ランダムに)，(n / 10)個
    missing_index = np.random.choice(list(range(n)), size= int(n/10), replace=False) 
    y[missing_index] = np.nan

    # 欠損値補完
    X,y,cov = missing_imputation.mean_value_imputation(X,y,sigma)

    # 外れ値除去(cook)
    O_obs = outlier_removal.cook_distance(X,y,M_obs,O_obs,lamda_cook)

    # 特徴選択(ms)
    M1_obs = feature_selection.ms(X,y,M_obs,O_obs,k_ms)

    # 特徴選択(lasso)
    M2_obs = feature_selection.lasso(X,y,M1_obs,O_obs,lamda_lasso) 
    
    # 特徴選択(sfs)
    M3_obs = feature_selection.sfs(X,y,M1_obs,O_obs,k_sfs)
    
    M_obs = common_func.union(M2_obs,M3_obs)

    if len(M_obs) != 0:

        rand_value = np.random.randint(len(M_obs))
        j_selected = M_obs[rand_value]
        
        a,b,z_obs,var = common_func.compute_teststatistics(X,y,M_obs,O_obs,j_selected,cov)
        std = np.sqrt(var) # 標準偏差

        z_min,z_max = -10 * std - np.abs(z_obs), 10 * std + np.abs(z_obs)
        z = z_min
        # リスト
        interval = []
        while z < z_max:
            # 初期設定(特徴，外れ値，切断区間)
            M = list(range(d))
            O = []
            l = np.NINF
            u = np.Inf

            # cook
            M,O,l,u = outlier_removal_si.cook_si(a,b,z,X,M,O,l,u,lamda_cook)

            # ms
            M1,O,l,u = feature_selection_si.ms_si(a,b,z,X,M,O,l,u,k_ms)
            
            # lasso 
            M2,O,l,u = feature_selection_si.lasso_si(a,b,z,X,M1,O,l,u,lamda_lasso)
            
            # sfs
            M3,O,l,u = feature_selection_si.sfs_si(a,b,z,X,M1,O,l,u,k_sfs)
            
            M = common_func.union(M2,M3)
            
            if set(O_obs) == set(O) and set(M_obs) == set(M):
                interval.append([l,u])
            
            z = u + eps

        if len(interval) != 0:
            pivot = compute_pivot.compute_pivot(z_obs,interval,0,std)
            if pivot is not None:
                p_value = min(pivot,1-pivot) * 2
                p_value_list.append(p_value)

ks = stats.kstest(p_value_list, "uniform")
print("ks p-value (naive) =", ks.pvalue)
if ks.pvalue > 0.05:
    print("一様分布")
else:
    print("一様分布ではない")

plt.hist(p_value_list)
plt.xlabel('P_value',fontsize = 15)
plt.ylabel('Frequency',fontsize = 15)
plt.title(f"Option1 for(ks test:{round(ks.pvalue,2)})",fontsize = 15)
plt.xlim([0,1]);
plt.show()