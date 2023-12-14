import numpy as np
from sklearn.linear_model import Lasso

# 平均値補完
def mean_value_imputation(X,y,sigma):

    # データ数の取得
    n = y.shape[0]

    # 欠損箇所を取得
    missing_index = np.where(np.isnan(y))[0]

    # index_random番目の以外のyを取得
    y_delete = np.delete(y, missing_index)

    # 平均値を計算
    y_mean = np.mean(y_delete)

    # 欠損値補完，(100,1)に変換
    y[missing_index] = y_mean
    y = y.reshape((n,1))

    # yの分散共分散行列の変更
    cov = np.identity(n)

    # 該当する箇所の分散共分散を変更していく
    # (1 / (n - 1)) * (y_1 + y_2 + ... + y_n-1)
    each_var_cov_value = sigma**2 / (n - len(missing_index))

    cov[:,missing_index] = each_var_cov_value
    cov[missing_index,:] = each_var_cov_value

    return X,y,cov

# Hot-deck法

# ユークリッド距離
def euclidean_imputation(X,y,sigma):

    # 欠損箇所を取得
    missing_index = np.where(np.isnan(y))[0]

    idx_list = []

    for index_random in missing_index:
        # ユークリッド距離の計算，二種類
        X_euclidean = np.sqrt(np.sum((X - X[index_random])**2, axis=1))

        # 欠損箇所を除いて，ユークリッド距離が最小になる
        X_euclidean_deleted = np.delete(X_euclidean, missing_index)
        idx_deleted = np.argmin(X_euclidean_deleted) # missing_indexを除いたindex

        # idx_deletedは欠損値を除いたデータセットに対するものなので、元のデータセットに対する正しいインデックスを得るためには、欠損値のインデックスが現在のインデックスより小さい場合には+1します。
        # idx_deletedを求めた時と同じ条件(欠損箇所を除いた)で，idxを求める
        original_indices = np.arange(len(X_euclidean))  # 元のデータセットのインデックス
        valid_indices = np.delete(original_indices, missing_index)  # 欠損値を除いたインデックス
        idx = valid_indices[idx_deleted]  # 欠損値を除いたインデックスから元のインデックスを取得

        y[index_random] = y[idx]

        idx_list.append(idx)

    y = y.reshape((X.shape[0],1))

    # 6. 分散共分散行列の作成
    cov = np.identity(X.shape[0])

    for index_random,idx in zip(missing_index,idx_list):
        cov[index_random,idx] = sigma**2
        cov[idx,index_random] = sigma**2

    return X,y,cov

# マンハッタン距離
def manhattan_imputation(X,y,sigma):

    # 欠損箇所を取得
    missing_index = np.where(np.isnan(y))[0]

    idx_list = []

    for index_random in missing_index:

        # マンハッタン距離の計算
        X_manhattan = np.sum(np.abs(X - X[index_random]), axis=1)

        # 欠損箇所を除いて，マンハッタン距離が最小になる
        X_manhattan_deleted = np.delete(X_manhattan, missing_index)
        idx_deleted = np.argmin(X_manhattan_deleted) # missing_indexを除いたindex

        # idx_deletedは欠損値を除いたデータセットに対するものなので、元のデータセットに対する正しいインデックスを得るためには、欠損値のインデックスが現在のインデックスより小さい場合には+1します。
        # idx_deletedを求めた時と同じ条件(欠損箇所を除いた)で，idxを求める
        original_indices = np.arange(len(X_manhattan))  # 元のデータセットのインデックス
        valid_indices = np.delete(original_indices, missing_index)  # 欠損値を除いたインデックス
        idx = valid_indices[idx_deleted]  # 欠損値を除いたインデックスから元のインデックスを取得

        y[index_random] = y[idx]

        idx_list.append(idx)

    y = y.reshape((X.shape[0],1))

    # 6. 分散共分散行列の作成
    cov = np.identity(X.shape[0])

    for index_random,idx in zip(missing_index,idx_list):
        cov[index_random,idx] = sigma**2
        cov[idx,index_random] = sigma**2

    return X,y,cov

# チェビシェフの距離
def chebyshev_imputation(X,y,sigma):

    # 欠損箇所を取得
    missing_index = np.where(np.isnan(y))[0]

    idx_list = []

    for index_random in missing_index:
        # チェビシェフの距離を求める
        X_chebyshev = np.max(np.abs(X - X[index_random]), axis=1)

        # 欠損箇所を除いて，チェビシェフの距離が最小になる
        X_chebyshev_deleted = np.delete(X_chebyshev, missing_index)
        idx_deleted = np.argmin(X_chebyshev_deleted) # missing_indexを除いたindex

        # idx_deletedは欠損値を除いたデータセットに対するものなので、元のデータセットに対する正しいインデックスを得るためには、欠損値のインデックスが現在のインデックスより小さい場合には+1します。
        # idx_deletedを求めた時と同じ条件(欠損箇所を除いた)で，idxを求める
        original_indices = np.arange(len(X_chebyshev))  # 元のデータセットのインデックス
        valid_indices = np.delete(original_indices, missing_index)  # 欠損値を除いたインデックス
        idx = valid_indices[idx_deleted]  # 欠損値を除いたインデックスから元のインデックスを取得

        y[index_random] = y[idx]

        idx_list.append(idx)

    y = y.reshape((X.shape[0],1))

    # 6. 分散共分散行列の作成
    cov = np.identity(X.shape[0])

    for index_random,idx in zip(missing_index,idx_list):
            cov[index_random,idx] = sigma**2
            cov[idx,index_random] = sigma**2

    return X,y,cov

# 回帰代入法

# 確定的
def regression_definite_imputation(X,y,sigma):
    # 欠損箇所を取得
    n = y.shape[0]
    missing_index = np.where(np.isnan(y))[0]

    # missing_index番目の以外のX,yを取得
    X_delete = np.delete(X, missing_index, 0)
    y_delete = np.delete(y, missing_index, 0).reshape(-1,1)

    cov = np.identity(n)

    # 回帰係数の推定
    beta_hat = np.linalg.inv(X_delete.T @ X_delete) @ X_delete.T @ y_delete

    data_list = list(range(n))

    # 各欠損箇所に対して
    for index_random in missing_index:

        # 欠損しているデータを取得
        X_missing = X[index_random]

        # beta_hatにより欠損していた値を補完
        y_new = X_missing @ beta_hat

        # 欠損箇所の補完
        y[index_random] = y_new

        # 分散の計算
        # 列ごとに分散，共分散を計算しているイメージ
        for i in data_list:
            # データの抽出
            each_x = X[i]
    
            # 分散の計算
            var_missing = sigma**2 * each_x @ np.linalg.inv(X_delete.T @ X_delete) @ X_missing.T
            #var_missing = (sigma * each_x.T @ np.linalg.inv(X_delete.T @ X_delete) @ X_missing)[0,0]

            # 分散共分散行列に追加
            cov[i,index_random] = var_missing
            cov[index_random,i] = var_missing

    y = y.reshape((n,1))

    return X,y,cov

# 確率的
def regression_probabilistic_imputation(X,y,sigma):
    # 欠損箇所を取得
    n = y.shape[0]
    missing_index = np.where(np.isnan(y))[0]

    # missing_index番目の以外のX,yを取得
    X_delete = np.delete(X, missing_index, 0)
    y_delete = np.delete(y, missing_index, 0).reshape(-1,1)

    cov = np.identity(n)

    # 回帰係数の推定
    beta_hat = np.linalg.inv(X_delete.T @ X_delete) @ X_delete.T @ y_delete

    data_list = list(range(n))

    # 各欠損箇所に対して
    for index_random in missing_index:

        # 欠損しているデータを取得
        X_missing = X[index_random]

        # beta_hatにより欠損していた値を補完
        rng = np.random.default_rng()
        noise = rng.standard_normal()

        y_new = X_missing @ beta_hat + noise

        # 欠損箇所の補完
        y[index_random] = y_new

        # 分散の計算
        # 列ごとに分散，共分散を計算しているイメージ
        for i in data_list:
            # データの抽出
            each_x = X[i]

            if i == index_random or i in missing_index:
                var_missing = sigma**2 * each_x @ np.linalg.inv(X_delete.T @ X_delete) @ X_missing.T + 1
                cov[i,index_random] = var_missing
                cov[index_random,i] = var_missing
            
            else:
                var_missing = sigma**2 * each_x @ np.linalg.inv(X_delete.T @ X_delete) @ X_missing.T
                cov[i,index_random] = var_missing
                cov[index_random,i] = var_missing

            #var_missing = (sigma * each_x.T @ np.linalg.inv(X_delete.T @ X_delete) @ X_missing)[0,0]

    y = y.reshape((n,1))

    return X,y,cov