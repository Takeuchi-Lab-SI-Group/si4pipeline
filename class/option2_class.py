import numpy as np
from tqdm import tqdm
from scipy import stats
from sicore import SelectiveInferenceNorm

import feature_selection
import feature_selection_si
import missing_imputation
import outlier_removal
import outlier_removal_si
import common_func

class PipiLineOption1:

    def __init__(self, X, y, sigma, **kwargs):
        self.X = X
        self.y = y
        self.var = sigma ** 2

        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self):

        X = self.X
        y = self.y
        n,d = X.shape

        M = list(range(d))
        O = []

        # 欠損値補完
        X,self.y,self.cov = missing_imputation.mean_value_imputation(X,y,self.var)

        y = self.y

        # 外れ値除去
        O1 = outlier_removal.cook_distance(X,y,M,O,self.lamda_cook)

        self.lamda_soft = outlier_removal.soft_IPOD_lambda(X,M,O)
        O2 = outlier_removal.soft_ipod(X,y,M,O,self.lamda_soft)

        O = common_func.intersect(O1,O2)

        # 特徴選択
        M1 = feature_selection.ms(X,y,M,O,self.k_ms)

        M2 = feature_selection.lasso(X,y,M1,O,self.lamda_lasso)

        M3 = feature_selection.sfs(X,y,M1,O,self.k_sfs)

        M = common_func.union(M2,M3)

        self.M = M
        self.O = O

        test_index = np.random.choice(M)

        # Xの特徴，外れ値除去
        X = np.delete(X,[O],0) # 外れ値の除去
        X = X[:,M] # 特徴の除去

        # ejの設定
        ej = [1 if j_select == test_index else 0 for j_select in M]
        ej = np.array(ej)
        
        # Imの設定
        Im = np.eye(n)
        Im = np.delete(Im,[O],0)
        
        self.eta = (np.linalg.inv(X.T @ X) @ X.T @ Im).T @ ej
        self.max_tail = 20 * np.sqrt(self.eta.T @ self.cov @ self.eta)

    def model_selector(self, M_and_O):
        M, O = M_and_O
        return (set(M) == set(self.M)) and (set(O) == set(self.O))

    def algorithm(self, a, b, z):
        
        X = self.X

        #   初期設定
        l = np.NINF
        u = np.Inf
        M = list(range(X.shape[1]))
        O = []

        M,O1,l,u = outlier_removal_si.cook_si(a,b,z,X,M,O,l,u,self.lamda_cook)

        M,O2,l,u = outlier_removal_si.soft_ipod_si(a,b,z,X,M,O,l,u,self.lamda_soft)

        O = common_func.intersect(O1,O2)
        
        M1,O,l,u = feature_selection_si.ms_si(a,b,z,X,M,O,l,u,self.k_ms)

        M2,O,l,u = feature_selection_si.lasso_si(a,b,z,X,M1,O,l,u,self.lamda_lasso)

        M3,O,l,u = feature_selection_si.sfs_si(a,b,z,X,M1,O,l,u,self.k_sfs)

        M = common_func.union(M2,M3)
        
        return (M, O),[l, u]
    
    def inference(self,**kwargs):
        self.si = SelectiveInferenceNorm(self.y, self.cov, self.eta)
        result = self.si.inference(self.algorithm, self.model_selector, max_tail=self.max_tail,**kwargs)
        return result.p_value