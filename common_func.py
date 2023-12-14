import numpy as np
from mpmath import mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
mp.dps = 500

# seedの指定なし
def data_generation(n,p,beta_vec,mu_vec):
    X = np.random.randn(n,p)
    true_y = np.zeros(n)
    y = X @ beta_vec + np.random.randn(n) + mu_vec
    
    y = y.reshape((n,1))
    true_y = true_y.reshape((n,1))

    return X,y,true_y

# seedの指定あり
def data_generation_seed(i,n,p,beta_vec,mu_vec):
    np.random.seed(i)
    X = np.random.randn(n,p)
    true_y = np.zeros(n)
    y = X @ beta_vec + np.random.randn(n) + mu_vec
    
    y = y.reshape((n,1))
    true_y = true_y.reshape((n,1))

    return X,y,true_y

# 選択された特徴，外れ値集合を考慮した検定統計量の計算
def compute_eta(j_selected,A,X,y,outlier):
    n = y.shape[0]
    ej = []
    ej = [1 if j_select == j_selected else 0 for j_select in A]
    ej = np.array(ej).reshape((len(A),1))
    Im = np.eye(n)
    Im = np.delete(Im,[outlier],0)
    etaj = (np.linalg.inv(X.T @ X) @ X.T @ Im).T @ ej
    etajTy = np.dot(etaj.T,y)[0][0]

    return etaj,etajTy

# 標準化
def Standardization(X,y):
    mean_X = np.mean(X,axis = 0)
    std_X = np.std(X, axis= 0)

    X_stand = (X - mean_X) / std_X

    y_stand = (y - np.mean(y)) / np.std(y)

    return X_stand,y_stand

# 正規化
def Normalization(X,y):
    max_X = np.max(X,axis=0)
    min_X = np.min(X,axis= 0)
    max_y = np.max(y)
    min_y = np.min(y)

    X_new = (X - min_X) / (max_X - min_X)
    y_new = (y - min_y) / (max_y - min_y)

    return X_new,y_new

# a,bの計算
def compute_a_b(n,etaj,y):
    eta_norm = np.linalg.norm(etaj)**2
    In = np.identity(n)
    c = np.dot(etaj,np.linalg.inv(np.dot(etaj.T,etaj)))
    cetaT = np.dot(c,etaj.T)
    # alpha,betaの計算でreshapeしている
    a = (np.dot(In - cetaT,y)).reshape(-1)
    b = (etaj / eta_norm).reshape(-1)

    return a,b  
