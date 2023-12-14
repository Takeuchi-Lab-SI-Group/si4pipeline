import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from scipy import stats
mp.dps = 500
from tqdm import tqdm

import common_func
import missing_imputation
import feature_selection
import outlier_removal

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

    print(y)

    # 欠損値補完
    X,y,cov = missing_imputation.regression_probabilistic_imputation(X,y,sigma)

    # 外れ値除去
    M,O = outlier_removal.cook_distance(X,y,M,O,lamda_cook)

    print(O)
    

    