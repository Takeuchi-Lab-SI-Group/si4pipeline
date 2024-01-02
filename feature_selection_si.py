import numpy as np
from sklearn.linear_model import Lasso

def ms_si(a,b,z,X,A,O,l,u,k):

    # yzの作成
    yz_flatten = a + b * z
    yz = yz_flatten.reshape(-1,1)

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    #yz = np.delete(yz,[O]).reshape(-1,1)
    yz = np.delete(yz,[O])

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,A]

    # Marginal Screening
    XTyz_abs = np.abs(X.T @ yz).flatten()
    sort_XTyz_abs = np.argsort(XTyz_abs)[::-1]

    Az = sort_XTyz_abs[:k]
    Acz = sort_XTyz_abs[k:]

    # 切断区間算出
    list_u = []
    list_v = []

    for i in Az:
        xj = X[:,i]
        sj = np.sign(np.dot(xj.T,yz))

        e1 = sj * np.dot(xj.T,a)
        e2 = sj * np.dot(xj.T,b)

        for k in Acz:
            xk = X[:,k]
            
            e3 = np.dot(xk.T,a)
            e4 = np.dot(xk.T,b)

            e5 = -np.dot(xk.T,a)
            e6 = -np.dot(xk.T,b)

            list_u.append(e4 - e2)
            list_u.append(e6 - e2)

            list_v.append(e1 - e3)
            list_v.append(e1 - e5)
    
    nu_plus = np.Inf
    nu_minus = np.NINF

    for left,right in zip(list_u,list_v):
        if np.around(left, 5) == 0:
            if right <= 0:
                raise Exception("エラー: 無効な条件です。")
            continue

        temp = right / left

        if left > 0: #論文の条件式より
            nu_plus = min(temp,nu_plus) 
        else:
            nu_minus = max(temp,nu_minus)
    
    assert nu_minus < nu_plus

    # 共通した切断区間
    l = max(l,nu_minus)
    u = min(u,nu_plus)

    # 元の特徴に基づいた選択結果
    Az = [A[i] for i in Az]

    return Az,l,u

def lasso_si(a,b,z,X,A,O,l,u,lamda):

    # yzの作成
    yz_flatten = a + b * z
    yz = yz_flatten.reshape(-1,1)

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    yz = np.delete(yz,[O]).reshape(-1,1)

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,A]

    # lasso
    clf = Lasso(alpha=lamda,fit_intercept=False,max_iter=5000,tol=1e-10)
    clf.fit(X,yz)
    coef = clf.coef_

    # lassoによる結果(インデックス表示)
    Az = np.where(coef != 0)[0].tolist() #coefが0でないものをリストとして表示
    Acz = [i for i in list(range(X.shape[1])) if i not in Az]
    s = np.sign(coef[Az])

    # XA,XAcの算出
    XA = X[:,Az]
    XAc = X[:,Acz]

    # 切断区間
    lasso_condition = []
    Pm = XA @ np.linalg.inv(XA.T @ XA) @ XA.T
    XA_plus = np.linalg.inv(XA.T @ XA) @ XA.T

    A0_plus = 1 / (lamda*X.shape[0]) * (XAc.T @ (np.identity(X.shape[0]) - Pm) ) 
    A0_minus = 1 / (lamda*X.shape[0]) * (-1 * XAc.T @ (np.identity(X.shape[0]) - Pm))
    b0_plus = np.ones(XAc.shape[1]) - XAc.T @ XA_plus.T @ s
    b0_minus = np.ones(XAc.shape[1]) + XAc.T @ XA_plus.T @ s

    A1 = -1 * np.diag(s) @ np.linalg.inv(XA.T @ XA) @ XA.T
    b1 = -1 * (lamda*X.shape[0]) * np.diag(s) @ np.linalg.inv(XA.T @ XA) @ s

    lasso_condition = [[A0_plus,b0_plus],[A0_minus,b0_minus],[A1,b1]]

    list_u = []
    list_v = []

    nu_plus = np.Inf
    nu_minus = np.NINF

    for Aj,bj in lasso_condition:
        uj = np.dot(Aj,b).reshape(-1).tolist()
        vj = (bj - np.dot(Aj, a)).reshape(-1).tolist()
        list_u.extend(uj)
        list_v.extend(vj)
    
    for left,right in zip(list_u,list_v):

        if np.round(left,5) == 0:
            if right <= 0:
                raise Exception("エラー: 無効な条件です。")
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

    # 元の特徴に基づいた結果
    Az = [A[i] for i in Az]

    return Az,l,u

def sfs_si(a,b,z,X,A,O,l,u,k):
    
    # yzの作成
    yz_flatten = a + b * z
    yz = yz_flatten.reshape(-1,1)

    # 外れ値の除去(X,y,a,bに対して)
    X = np.delete(X,[O],0)
    yz = np.delete(yz,[O]).reshape(-1,1)

    a = np.delete(a,[O])
    b = np.delete(b,[O])

    # 特徴の除去
    X = X[:,A]

    # sfs
    Az = []
    Acz = list(range(X.shape[1]))
    s = []

    for i in range(k):
        XA = X[:,Az]
        r = yz - XA @ np.linalg.pinv(XA.T @ XA) @ XA.T @ yz
        correlation = X[:,Acz].T @ r 

        index = np.argmax(np.abs(correlation)) #何番目の要素が最大か？

        s.append(np.sign(correlation[index]))

        Az.append(Acz[index])
        Acz.remove(Acz[index])

    # 切断区間
    list_u = []
    list_v = []
    Acz = list(range(X.shape[1])) # 選択されなかった特徴，最初は全特徴を含む

    for i in range(k):
        XA = X[:,Az[0:i]] # 残差r_t 計算用
        x_jt = X[:,Az[i]] # 選択イベント x_\hat{j}_{t} 
        s_t = s[i] # \hat{s}_{t} 
        F = np.identity(X.shape[0]) - XA @ np.linalg.pinv(XA.T @ XA) @ XA.T
        Acz.remove(Az[i])

        for j in Acz:
            x_j = X[:,j]

            u1 = np.dot((x_j - s_t * x_jt).T,np.dot(F,b))
            v1 = np.dot(-(x_j - s_t * x_jt).T,np.dot(F,a))
            u2 = np.dot((- x_j - s_t * x_jt).T,np.dot(F,b))
            v2 = np.dot( -(- x_j - s_t * x_jt).T,np.dot(F,a))

            list_u.append(u1)
            list_u.append(u2)
            list_v.append(v1)
            list_v.append(v2)
    
    nu_plus = np.Inf
    nu_minus = np.NINF

    for m in range(len(list_u)):
        left = list_u[m]
        right = list_v[m]

        if np.around(left, 5) == 0:
            if right <= 0:
                print("ERROR")
                
            continue

        temp = right / left

        if left > 0: #論文の条件式より
            nu_plus = min(temp,nu_plus) 
        else:
            nu_minus = max(temp,nu_minus)
    
    assert nu_minus < nu_plus

    # 切断区間の共通区間
    l = max(l,nu_minus)
    u = min(u,nu_plus)

    # 元の特徴に基づいた結果
    Az = [A[i] for i in Az]

    return Az,l,u