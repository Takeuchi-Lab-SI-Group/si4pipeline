import numpy as np
from sklearn.linear_model import Lasso

def cook_distance(X,y,M,O,lamda):

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

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

    # 元の特徴に基づいた結果
    O2 = [num_outlier_data[i] for i in outlier]
    O += O2

    return O

def dffits(X,y,M,O,lamda):

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

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

    # 元の特徴に基づいた結果
    O2 = [num_outlier_data[i] for i in outlier]
    O += O2

    return O

# soft-ipodで用いるハイパーパラメータの計算
def soft_IPOD_lambda(X,M,O):
    X = np.delete(X,[O],0)
    X = X[:,M]
    
    nsim = 5000
    lamda_list = np.array([])
    
    n = X.shape[0]
    hat_matrix =  X @ np.linalg.inv(X.T @ X) @ X.T
    PXperp = np.identity(n) - hat_matrix

    for i in range(nsim):
        eps = np.random.randn(n)
        tXeps = np.abs(PXperp.T @ eps)
        imax = np.max(tXeps)
        lamda_list = np.append(lamda_list, imax)
    
    lamda = 0.7*(np.mean(lamda_list/n))

    return lamda

def soft_ipod(X,y,M,O,lamda):

    # 最後に最終的に得られる外れ値集合を求めるのに使用する
    num_data = list(range(X.shape[0]))
    num_outlier_data = [x for x in num_data if x not in O]

    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    # 特徴の除去
    X = X[:,M]

    # soft-IPODの準備
    n = X.shape[0]

    hat_matrix =  X @ np.linalg.inv(X.T @ X) @ X.T
    PXperp = np.identity(n) - hat_matrix
    PXperpy = PXperp @ y

    # soft-IPODの実行
    clf = Lasso(alpha=lamda,fit_intercept=False,max_iter=5000,tol=1e-10)
    clf.fit(PXperp,PXperpy)
    coef = clf.coef_
    outlier = np.where(coef!=0)[0].tolist() #外れ値

    O2 = [num_outlier_data[i] for i in outlier]
    O += O2

    return O