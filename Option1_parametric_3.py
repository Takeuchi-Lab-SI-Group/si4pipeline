import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from scipy import stats
mp.dps = 500
eps = 1e-4
from tqdm import tqdm

import common_func
import missing_imputation
import feature_selection
import feature_selection_si
import feature_selection_si_2
import outlier_removal
import outlier_removal_si
import outlier_removal_si_2
import compute_pivot

from sicore import tn_cdf_mpmath

p_list = []

# def option1(n,d):
# cook - ms - lasso and/or sfs
for i in tqdm(range(10)):
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

    X,y,true_y = common_func.data_generation(n,d,beta_vec,mu_vec)
    #X,y,true_y = common_func.data_generation_seed(each_seed,n,d,beta_vec,mu_vec)
    
    # 欠損値を加える(ランダムに)，今回は10個
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
    
    M_obs = common_func.intersect(M2_obs,M3_obs)

    if len(M_obs) != 0:

        rand_value = np.random.randint(len(M_obs))
        j_selected = M_obs[rand_value]
        
        a,b,z_obs,var = common_func.compute_teststatistics(X,y,M_obs,O_obs,j_selected,cov)
        std = np.sqrt(var) # 標準偏差

        z_min,z_max = -20 * var,20 * var

        z = z_min

        # リスト(O_list：外れ値,M_list：特徴，I_list：切断区間)
        interval = []

        while z < z_max:
            # 初期設定(特徴，外れ値，切断区間)
            M = list(range(d))
            O = []
            Iz = [np.NINF,np.inf]

            # cook
            O,Iz = outlier_removal_si_2.cook_si(a,b,z,X,M,O,Iz,lamda_cook)

            # ms
            M1,Iz = feature_selection_si_2.ms_si(a,b,z,X,M,O,Iz,k_ms)
            
            # lasso
            M2,Iz = feature_selection_si_2.lasso_si(a,b,z,X,M1,O,Iz,lamda_lasso)
            
            # sfs
            M3,Iz = feature_selection_si_2.sfs_si(a,b,z,X,M1,O,Iz,k_sfs)
            
            M = common_func.intersect(M2,M3)

            if set(O_obs) == set(O) and set(M_obs) == set(M):
                interval += Iz

            z = Iz[-1][-1] + eps

        z_min,z_max = -20 * var,20 * var
        z = z_min
        # リスト(O_list：外れ値,M_list：特徴，I_list：切断区間)
        interval2 = []
        while z < z_max:
            # 初期設定(特徴，外れ値，切断区間)
            M = list(range(d))
            O = []
            l = np.NINF
            u = np.Inf

            # cook
            O,l,u = outlier_removal_si.cook_si(a,b,z,X,M,O,l,u,lamda_cook)

            # ms
            M1,l,u = feature_selection_si.ms_si(a,b,z,X,M,O,l,u,k_ms)

            # lasso
            M2,l,u = feature_selection_si.lasso_si(a,b,z,X,M1,O,l,u,lamda_lasso)

            # sfs
            M3,l,u = feature_selection_si.sfs_si(a,b,z,X,M1,O,l,u,k_sfs)

            M = common_func.intersect(M2,M3)

            if set(O_obs) == set(O) and set(M_obs) == set(M):
                interval2.append([l,u])

            z = u + eps
        print(interval)
        print(interval2)
#         if len(interval) != 0:
#             # # 区間と検定統計量の標準化
#             # z_obs = z_obs / std
#             # interval = interval / std
#             # print(interval)
#             # # p値の計算
#             # p = tn_cdf_mpmath(z_obs,interval,absolute=True)
#             # p_value = min(1-p, p) * 2
#             # p_list.append(p_value)

#             pivot = compute_pivot.compute_pivot(z_obs,interval,0,std)
#             if pivot != None:
#                 p_value = (min(pivot,1-pivot)) * 2
#                 p_list.append(p_value)




# ks = stats.kstest(p_list, "uniform")
# print("ks p-value (naive) =", ks.pvalue)
# if ks.pvalue > 0.05:
#     print("一様分布")
# else:
#     print("一様分布ではない")

# plt.hist(p_list)
# plt.xlabel('P_value',fontsize = 15)
# plt.ylabel('Frequency',fontsize = 15)
# plt.title(f"p OC MS -> Lasso AND SFS(ks test:{round(ks.pvalue,2)})",fontsize = 15)
# #plt.title(f"naive uv p(ks test:{round(ks.pvalue,2)})",fontsize = 15)
# #plt.title(f"naive paper p(ks test:{round(ks.pvalue,2)})",fontsize = 15)
# plt.xlim([0,1]);
# plt.show()