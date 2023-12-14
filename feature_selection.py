import numpy as np
from sklearn.linear_model import Lasso

# M：選択された特徴，O：検出された外れ値集合，最後：ハイパーパラメータ
def ms(X,y,M,O,k):
    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    # 特徴の除去
    X = X[:,M]

    # Marginal Screening
    XTy_abs = np.abs(X.T @ y).flatten()
    sort_XTy_abs = np.argsort(XTy_abs)[::-1]

    A = sort_XTy_abs[:k]
    Ac = sort_XTy_abs[k:]

    M = [M[i] for i in A]

    return M,O

def lasso(X,y,M,O,lamda):
    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    # 特徴の除去
    X = X[:,M]

    # lasso
    clf = Lasso(alpha=lamda,fit_intercept=False,max_iter=5000,tol=1e-10)
    clf.fit(X,y)
    coef = clf.coef_

    # lassoによる結果(インデックス表示)
    A = np.where(coef != 0)[0].tolist() #coefが0でないものをリストとして表示
    Ac = [i for i in X.shape[1] if i not in A]
    s = np.sign(coef[A])

    # 元の特徴に基づいた結果
    M = [M[i] for i in A]

    return M,O

def sfs(X,y,M,O,k):
    # 外れ値の除去(X,yに対して)
    X = np.delete(X,[O],0)
    y = np.delete(y,[O]).reshape(-1,1)

    # 特徴の除去
    X = X[:,M]

    # sfs
    A = []
    Ac = list(range(X.shape[1]))
    s = []

    for i in range(k):
        XA = X[:,A]
        r = y - XA @ np.linalg.pinv(XA.T @ XA) @ XA.T @ y
        correlation = X[:,Ac].T @ r 

        index = np.argmax(np.abs(correlation)) #何番目の要素が最大か？

        s.append(np.sign(correlation[index]))

        A.append(Ac[index])
        Ac.remove(Ac[index])

    # 元の特徴に基づいた結果
    M = [M[i] for i in A]

    return M,O