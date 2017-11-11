# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import random as rd


class AdversarialPredictionClassifier(object):
    def __init__(self,base_estimator=None):
        if base_estimator == None:
            self.base_estimator = LogisticRegression()
        else:
            self.base_estimator = base_estimator
        
    def fit(self,X,y):
        self.base_estimator.fit(X,y)
        self.X = pd.DataFrame(X)
        self.y = y
        self._n_features = len(self.X.iloc[0])
        self._columns = self.X.columns.values
        self._X_std = [np.std(self.X.iloc[:,i], ddof=1) for i in xrange(self._n_features)]

    def _predict_single(self,X_pred,adv_estimator=None):
        if adv_estimator == None:
            self.m_adv = LogisticRegression()
        else:
            self.m_adv = adv_estimator
        
        y_pred = self.base_estimator.predict(X_pred)[0]
        
        X_obs = self.X.loc[self.y == y_pred]
        
        n_adv_sample = len(X_obs.iloc[:,0])
        
        X_cand = pd.DataFrame()
        for i in xrange(self._n_features):
            X_cand[self._columns[i]] = [rd.gauss(X_pred[0][i], 
                   self._X_std[i]) for j in xrange(n_adv_sample)]
        
        self.X_cand = X_cand
        
        X_adv = X_obs.append(X_cand)
        
        y_adv = [0 for i in xrange(n_adv_sample)]

        for i in xrange(n_adv_sample):
            y_adv.append(1)
            
        
        conf = min(1,(1-np.mean(cross_val_score(self.m_adv, X_adv, y_adv, cv = 10,
                                                scoring = "accuracy")))/0.5)
        
        self.m_adv.fit(X_adv,y_adv)
        
        pred_prb = self.base_estimator.predict_proba(X_pred)
        pred = [list(pred_prb[0]), conf]

        return pred
    
    def predict_conf(self, X_pred, adv_estimator=None):
        arr_pred = []
        for i in xrange(len(X_pred)):    
            arr_pred.append(self._predict_single(X_pred[i].reshape(1,self._n_features)))
        return arr_pred
          

def main():
    return -1


if __name__ == "__main__":
    main()           
