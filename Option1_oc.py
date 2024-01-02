import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from scipy import stats
mp.dps = 500
from tqdm import tqdm

import common_func
import missing_imputation
import feature_selection
import feature_selection_si
import outlier_removal
import outlier_removal_si

from sicore import tn_cdf_mpmath

p_list = []

# def option1(n,d):
# cook - ms - lasso and/or sfs
for i in tqdm(range(10000)):
    n = 100
    d = 10
    #each_seed = 1

    # ハイパラの設定
    lamda_cook = 3

    k_ms = 5
    k_sfs = 3
    lamda_lasso = 0.08

    # 初期条件
    M = list(range(d))
    O = []

    # データの生成
    sigma = 1
    beta_vec = np.zeros(d)
    mu_vec = np.zeros((n))
    s_list = [1,2,3,4,5]
    mu_vec[s_list] = 4

    X,y,true_y = common_func.data_generation(n,d,beta_vec,mu_vec)
    #X,y,true_y = common_func.data_generation_seed(each_seed,n,d,beta_vec,mu_vec)
    
    # 欠損値を加える(ランダムに)，今回は10個
    missing_index = np.random.choice(list(range(n)), size= int(n/10), replace=False) 
    y[missing_index] = np.nan

    # 欠損値補完
    X,y,cov = missing_imputation.mean_value_imputation(X,y,sigma)

    # 外れ値除去(cook)
    O1 = outlier_removal.cook_distance(X,y,M,O,lamda_cook)
    # print(f'cook = {O1}')

    # 特徴選択(ms)
    M1 = feature_selection.ms(X,y,M,O1,k_ms)
    # print(f'ms = {M1}')

    # 特徴選択(lasso)
    M2 = feature_selection.lasso(X,y,M1,O1,lamda_lasso) 

    # 特徴選択(sfs)
    M3 = feature_selection.sfs(X,y,M1,O1,k_sfs)
    
    A = common_func.intersect(M2,M3)

    # print(A,O1)

    if len(A) != 0:

        rand_value = np.random.randint(len(A))
        j_selected = A[rand_value]
        
        a,b,z_obs,var = common_func.compute_teststatistics(X,y,A,O1,j_selected,cov)

        l = np.NINF
        u = np.Inf

        # siについて
        O1,l,u = outlier_removal_si.cook_si(a,b,z_obs,X,M,O,l,u,lamda_cook)
        # print(f'cook = {O1}')
        # print(l,u)

        M1,l,u = feature_selection_si.ms_si(a,b,z_obs,X,M,O1,l,u,k_ms)

        # print(l,u)
        # print(f'ms = {M1}')

        M2,l,u = feature_selection_si.lasso_si(a,b,z_obs,X,M1,O1,l,u,lamda_lasso)
    
        # print(l,u)
        # print(f'lasso = {M2}')

        M3,l,u = feature_selection_si.sfs_si(a,b,z_obs,X,M1,O1,l,u,k_sfs)

        # print(l,u)
        # print(f'sfs = {M3}')

        interval = [l,u]

        if len(interval) != 0:
            
            # 区間と検定統計量の標準化
            std = np.sqrt(var)
            z_obs = z_obs / std
            interval = interval / std

            # p値の計算
            p = tn_cdf_mpmath(z_obs,interval,absolute=True)
            p_value = min(1-p, p) * 2
            p_list.append(p_value)

ks = stats.kstest(p_list, "uniform")
print("ks p-value (naive) =", ks.pvalue)
if ks.pvalue > 0.05:
    print("一様分布")
else:
    print("一様分布ではない")

plt.hist(p_list)
plt.xlabel('P_value',fontsize = 15)
plt.ylabel('Frequency',fontsize = 15)
plt.title(f"p OC MS -> Lasso AND SFS(ks test:{round(ks.pvalue,2)})",fontsize = 15)
#plt.title(f"naive uv p(ks test:{round(ks.pvalue,2)})",fontsize = 15)
#plt.title(f"naive paper p(ks test:{round(ks.pvalue,2)})",fontsize = 15)
plt.xlim([0,1]);
plt.show()