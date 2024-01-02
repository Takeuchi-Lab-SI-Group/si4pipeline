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

# def option1(n,d):

for i in range(1):
    n = 100
    d = 10
    each_seed = 1

    # ハイパラの設定
    lamda_cook = 3
    lamda_dffits = 3

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

    #X,y,true_y = common_func.data_generation(n,d,beta_vec,mu_vec)
    X,y,true_y = common_func.data_generation_seed(each_seed,n,d,beta_vec,mu_vec)
    
    # 欠損値を加える(ランダムに)，今回は10個
    missing_index = np.random.choice(list(range(n)), size= int(n/10), replace=False) 
    y[missing_index] = np.nan

    # 欠損値補完
    X,y,cov = missing_imputation.regression_probabilistic_imputation(X,y,sigma)

    # 外れ値除去(cook)
    M,O1 = outlier_removal.cook_distance(X,y,M,O,lamda_cook)

    # 外れ値除去(soft)
    lamda_soft = outlier_removal.soft_IPOD_lambda(X)
    M,O2 = outlier_removal.soft_ipod(X,y,M,O1,lamda_soft)

    # 外れ値除去(dffits)
    M,O3 = outlier_removal.dffits(X,y,M,O,lamda_dffits)

    O = common_func.intersect(O2,O3)

    # 特徴選択(ms)
    M1,O = feature_selection.ms(X,y,M,O,k_ms)
    print(M1)

    # 特徴選択(lasso)
    M2,O = feature_selection.lasso(X,y,M1,O,lamda_lasso)
    print(M2)
    
    # 特徴選択(sfs)
    M3,O = feature_selection.sfs(X,y,M1,O,k_sfs)
    print(M3)
    
    M = common_func.intersect(M2,M3)

    print(M,O)

    if len(M) != 0:

        rand_value = np.random.randint(len(M))
        j_selected = M[rand_value]

        a,b,z_obs = common_func.compute_teststatistics(X,y,M,O,j_selected,cov)

        # siについて
        O1,l1,u1 = outlier_removal_si.cook_si(a,b,z_obs,X,)
    

    