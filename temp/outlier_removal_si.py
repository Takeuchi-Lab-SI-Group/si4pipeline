import numpy as np
from sklearn.linear_model import Lasso
from sicore import polytope_to_interval, intersection

def cook_si(a,b,z,X,M,O,l,u,lamda):

    # yzの作成
    yz_flatten = a + b * z
    y = yz_flatten.reshape(-1,1)

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,M]

    # cook's distance
    non_outlier = []
    outlier = []
    n,p = X.shape

    hat_matrix =  X @ np.linalg.inv(X.T @ X) @ X.T
    Px = np.identity(n) - hat_matrix
    threads = lamda / n #しきい値の設定

    # 外れ値の除去
    for i in range(n):
        ej = np.zeros((n,1))
        ej[i] = 1
        hi = hat_matrix[i][i] #Pxの対角成分
        Di_1 = (y.T @ (Px @ ej @ ej.T @ Px) @ y) / (y.T @ Px @ y) # Diの1項目
        Di_2 = ((n - p) * hi) / (p * (1 - hi)**2) # Diの2項目
        Di = Di_1 * Di_2

        if Di < threads:
            non_outlier.append(i)
        else:
            outlier.append(i)
    
    # 切断区間
    interval = [-np.inf, np.inf]
    B = np.zeros(n)
    C = 0 

    for i in range(n):

        ej = np.zeros((n,1))
        ej[i] = 1
        hi = hat_matrix[i][i]
        H_1 = ((n - p) * hi) * Px @ ej @ ej.T @ Px
        H_2 = ((lamda * p * (1 - hi)**2) / n ) * Px
        H = H_1 - H_2

        if i in outlier:
            H = -H

        intervals = polytope_to_interval(a,b,H,B,C)
        interval = intersection(intervals,interval)
    
    # 切断区間の共通区間
    for lz,uz in interval:
        if lz < z < uz:
            l = max(lz,l)
            u = min(uz,u)
        
    assert l < z < u
    
    # 元の特徴に基づいた結果
    O2 = [num_outlier_data[i] for i in outlier]
    O += O2
    
    return M,O,l,u


def dffits_si(a,b,z,X,M,O,l,u,lamda):

    # yzの作成
    yz_flatten = a + b * z
    y = yz_flatten.reshape(-1,1)

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,M]

    # DFFITS
    non_outlier = []
    outlier = []
    n,p = X.shape

    hat_matrix =  X @ np.linalg.inv(X.T @ X) @ X.T
    Px = np.identity(n) - hat_matrix
    threads = (lamda * p) / (n - p) #しきい値の設定

    # 外れ値の除去
    for i in range(n):
        ej = np.zeros((n,1))
        ej[i] = 1
        hi = hat_matrix[i][i] #Pxの対角成分
        DFFITSi_1 = np.sqrt(hi * (n - p - 1)) / (1 - hi) # DFFITSの片側
        DFFITSi_2_denominator = y.T @ Px @ y - ((y.T @ Px @ ej @ ej.T @ Px @ y) / (1 - hi))
        DFFITSi_2 = (ej.T @ Px @ y) / np.sqrt(DFFITSi_2_denominator )
        DFFITSi = DFFITSi_1 * DFFITSi_2

        if DFFITSi**2 < threads:
            non_outlier.append(i)
        else:
            outlier.append(i)
    
    # 切断区間
    interval = [-np.inf, np.inf]
    B = np.zeros(n)
    C = 0 

    for i in range(n):
        ej = np.zeros((n,1))
        ej[i] = 1
        hi = hat_matrix[i][i]
        H_1_1 = ((hi * (n - p - 1)) / (1 - hi)**2) + ((lamda * p) / ((n - p)*(1 - hi)))
        H_1 = H_1_1 * Px @ ej @ ej.T @ Px 
        H_2 = ((lamda * p)/(n - p)) * Px
    
        H = H_1 - H_2
        if i in outlier:
            H = - H

        intervals = polytope_to_interval(a,b,H,B,C)
        interval = intersection(intervals,interval)
    
    # 切断区間の共通区間
    for lz,uz in interval:
        if lz < z < uz:
            l = max(lz,l)
            u = min(uz,u)

    # 元の特徴に基づいた結果
    O2 = [num_outlier_data[i] for i in outlier]
    O += O2

    assert l < z < u
    
    return M,O,l,u

# soft-IPODのハイパーパラメータは外部から入力する感じ
def soft_ipod_si(a,b,z,X,M,O,l,u,lamda):
    
    # yzの作成
    yz_flatten = a + b * z
    y = yz_flatten.reshape(-1,1)

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,M]

    # soft-IPODの準備
    n,p = X.shape
    I = np.identity(n)

    hat_matrix =  X @ np.linalg.inv(X.T @ X) @ X.T
    PXperp = I - hat_matrix
    PXperpy = PXperp @ y

    # soft-IPODの実行
    clf = Lasso(alpha=lamda,fit_intercept=False,max_iter=5000,tol=1e-10)
    clf.fit(PXperp,PXperpy)
    coef = clf.coef_
    outlier = np.where(coef!=0)[0].tolist() #外れ値
    non_outlier = np.where(coef==0)[0].tolist() #非外れ値
    s = np.sign(coef[outlier])

    # 切断区間
    soft_IPOD_condition = []
    # PXperpを計算
    X_caron = PXperp

    X_caron_M = X_caron[:,non_outlier]
    X_caron_Mc = X_caron[:,outlier]

    PX_caron_Mc_perp = I - X_caron_Mc @ np.linalg.inv(X_caron_Mc.T @ X_caron_Mc) @ X_caron_Mc.T
    X_caron_Mc_plus = X_caron_Mc @ np.linalg.inv(X_caron_Mc.T @ X_caron_Mc)

    # 以下はy_caronに対する係数
    A0_plus = (1 / (lamda * n)) * (X_caron_M.T @ PX_caron_Mc_perp @ X_caron)
    A0_minus = -1 *  (1 / (lamda * n)) * (X_caron_M.T @ PX_caron_Mc_perp @ X_caron)

    b0_plus = np.ones(len(non_outlier)) - X_caron_M.T @ X_caron_Mc_plus @ s
    b0_minus = np.ones(len(non_outlier)) + X_caron_M.T @ X_caron_Mc_plus @ s

    A1 = -1 * np.diag(s) @ np.linalg.inv(X_caron_Mc.T @ X_caron_Mc) @ X_caron_Mc.T @ X_caron
    b1 = -1 * n * lamda * np.diag(s) @ np.linalg.inv(X_caron_Mc.T @ X_caron_Mc) @ s

    soft_IPOD_condition = [[A0_plus,b0_plus],[A0_minus,b0_minus],[A1,b1]]

    list_u = []
    list_v = []

    nu_plus = np.Inf
    nu_minus = np.NINF

    for j in soft_IPOD_condition:
        Aj,bj = j
        uj = ((Aj @ b).reshape(-1)).tolist()
        vj = ((bj - Aj @ a).reshape(-1)).tolist()
        list_u.extend(uj)
        list_v.extend(vj)

    for m in range(len(list_u)):
        left = list_u[m]
        right = list_v[m]

        if np.around(left,5) == 0:
            if right <= 0:
                print("ERROR")
                
            continue

        temp = right / left

        if left > 0:
            nu_plus = min(temp,nu_plus)
        else:
            nu_minus = max(temp,nu_minus)

    assert nu_minus < nu_plus

    # 切断区間の共通区間
    l = max(l,nu_minus)
    u = min(u,nu_plus)
    
    assert l < z < u

    # 元の特徴に基づいた結果
    O2 = [num_outlier_data[i] for i in outlier]
    O += O2

    return M,O,l,u