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

    return X,y,true_y

# seedの指定あり
def data_generation_seed(i,n,p,beta_vec,mu_vec):
    np.random.seed(i)
    X = np.random.randn(n,p)
    true_y = np.zeros(n)
    y = X @ beta_vec + np.random.randn(n) + mu_vec

    return X,y,true_y

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

# 検定に関わるパラメータの計算
def compute_teststatistics(X,y,M,O,j_selected,cov):

    n = y.shape[0]

    # Xの変更
    X = np.delete(X,[O],0) # 外れ値の除去
    X = X[:,M] # 特徴の除去

    # ejの設定
    ej = [1 if j_select == j_selected else 0 for j_select in M]
    ej = np.array(ej)
    
    # Imの設定
    Im = np.eye(n)
    Im = np.delete(Im,[O],0)
    
    # etaj,etajTyの計算
    etaj = (np.linalg.inv(X.T @ X) @ X.T @ Im).T @ ej
    z_obs = np.dot(etaj.T,y)

    var = etaj.T @ cov @ etaj

    b = (cov @ etaj) / var
    a = (np.identity(n) - b.reshape(-1,1) @ etaj.reshape(1,-1)) @ y
    
    return a,b,z_obs,var

# AND
def intersect(*lists):

    # 与えられたリスト(集合)が2個未満の場合
    if len(lists) < 2:
        raise Exception("エラー: 無効な条件です。")
    
    # listsはタプルとしてまとめられる
    # 全部取り出してANDを取るのは難しい
    # .intersection_updateを使って，基準を設けて積集合を取る

    result_set = set(lists[0])

    # result_setを基準に他の集合との積集合を取る
    for each_list in lists[1:]:
        result_set.intersection_update(each_list)

    return list(result_set)

# OR
def union(*lists):

    # 与えられたリスト(集合)が2個未満の場合
    if len(lists) < 2:
        raise Exception("エラー: 無効な条件です。")
    
    # listsはタプルとしてまとめられる
    # 全部取り出してORを取るのは難しい
    # .updateを使って，基準を設けて積集合を取る

    result_set = set(lists[0])

    # result_setを基準に他の集合との積集合を取る
    for each_list in lists[1:]:
        result_set.update(each_list)

    return list(result_set)

# 共通区間を取り出す
def convert_to_nested_list(interval):
    if isinstance(interval[0], list):  # 既に二重のリストである場合
        return interval
    else:  # 単一のリストである場合
        return [interval]

def interval_disassembly(interval1,interval2):
    interval_list = []
    i, j = 0, 0

    while i < len(interval1) and j < len(interval2):
        interval1 = convert_to_nested_list(interval1)
        interval2 = convert_to_nested_list(interval2)
        
        L1, U1 = interval1[i]
        L2, U2 = interval2[j]

        # インターバルが重なっている場合
        if U2 > L1 and U1 > L2:
            new_L = max(L1,L2)
            new_U = min(U1,U2)
            interval_list.append([new_L,new_U])

        # 次のインターバルに進みます
        if U1 < U2:
            i += 1
        else:
            j += 1
    
    return interval_list

def check_interval(interval1,interval2):
    interval_list = []
    i, j = 0, 0

    while i < len(interval1) and j < len(interval2):
        interval1 = convert_to_nested_list(interval1)
        interval2 = convert_to_nested_list(interval2)
        
        L1, U1 = interval1[i]
        L2, U2 = interval2[j]

        # インターバルが重なっている場合
        if U2 > L1 and U1 > L2:
            new_L = max(L1,L2)
            new_U = min(U1,U2)
            interval_list.append([new_L,new_U])

        # 次のインターバルに進みます
        if U1 < U2:
            i += 1
        else:
            j += 1
    
    return len(interval_list) > 0