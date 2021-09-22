import argparse
from typing import Counter

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import os, sys
import datasets
from algorithm import NonLinear4

from imblearn.over_sampling import SMOTE

import imblearn.over_sampling as overer
import imblearn.combine as combiner
from smote_variants import KernelADASYN
from smote_variants import MWMOTE
import time

from sklearn.neighbors import KNeighborsClassifier

DEFAULT_DATA_PATH = Path(__file__).parent / 'datasets' 
classifiers = {
    "LSVM":SVC(kernel="linear", C=0.025),
    "DT":DecisionTreeClassifier(max_depth=5),
    "NB":GaussianNB(),
    "NN":KNeighborsClassifier(3)
}

def metric_performance(y_test, predictions):
    g_mean = 1.0
    for label in np.unique(y_test):
        idx = (y_test == label)
        g_mean *= metrics.accuracy_score(y_test[idx], predictions[idx])
    g_mean = np.sqrt(g_mean)
    auc = metrics.roc_auc_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions)
    return [round(g_mean,4), round(auc,4), round(f1,4)]

def fun(res:dict, score:list, name:str):
    if name not in res.keys():
        res[name] = np.zeros(len(score))
    for i in range(len(score)):
        res[name][i] += score[i]

RANDOM_STATE = 12
oversampler = {
    "SM": overer.SMOTE(random_state=RANDOM_STATE),
    "BSM": overer.BorderlineSMOTE(random_state=RANDOM_STATE),
    "SMENN": combiner.SMOTEENN(random_state=RANDOM_STATE),
    "KADA": KernelADASYN(random_state=RANDOM_STATE),
    "MW": MWMOTE(k1=5, k2=3, cf_th=5, cmax=2, random_state=RANDOM_STATE)
}

def pipeline(data_name, folder, min_size=10, n_fold=5, n_rep_fold=10):
    X, y = datasets.read_data(data_name, folder) #1-positive,0-negtive
    X = StandardScaler().fit_transform(X)
    kf = RepeatedKFold(n_splits=n_fold, n_repeats=n_rep_fold,random_state=RANDOM_STATE) # n_repeats * n_splits次
    res = dict()

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if min_size == 0:
            min_size = len(np.where(y_train==1)[0])

        x_train_maj = X_train[np.where(y_train==0)[0], :]
        x_train_min = X_train[np.random.choice((np.where(y_train==1)[0]),30), :]
        x_train_min = X_train[np.random.choice((np.where(y_train==1)[0]),min_size), :]
        x_trainImb = np.append(x_train_maj, x_train_min,axis=0)
        y_trainImb = np.append(np.zeros(x_train_maj.shape[0]), np.ones(min_size))

        numSamples = np.sum(y_trainImb==0)-np.sum(y_trainImb==1)
        np.random.seed(seed=1234)

        # ========================SM variants=============================
        flag = 0
        if flag:
            for sampler_name, smp in oversampler.items():
                X_res, y_res = smp.fit_resample(x_trainImb, y_trainImb)
                for clfname, clf in classifiers.items():
                    y_pred = clf.fit(X_res, y_res).predict(X_test)
                    score = metric_performance(np.array(y_test), np.array(y_pred))
                    fun(res, score, "%s_%s" % (clfname, sampler_name))
        
        # =========================Partd ImpovedSRBF======================
        flag = 0
        if flag:
            print("new4")
            irbo = NonLinear4()
            X_res, y_res = irbo.fit_sample(x_trainImb, y_trainImb, numSamples)
            for clfname, clf in classifiers.items():
                y_pred = clf.fit(X_res, y_res).predict(X_test)
                score = metric_performance(np.array(y_test), np.array(y_pred))
                fun(res, score, "%s_%s" % (clfname, "new4"))
    return res

if __name__ == '__main__':
    min_size = 0
    n_fold = 5
    n_rep_fold = 2

    name = "rbo"
    fp = open("%d_%d_%s.txt" % (n_fold, n_rep_fold, name), 'a+')
    time_str = time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time()))
    fp.write(time_str)

    for dirs in ["KEEL2"]:
        for data_name in os.listdir(dirs):
            if data_name not in ['winequality-red-8_vs_6.dat']:
                continue
            print(data_name)
            res = pipeline(data_name, folder=dirs, min_size=min_size, n_fold=n_fold, n_rep_fold=n_rep_fold)  
            for model, score in res.items():
                score = [round(sc / float(n_fold * n_rep_fold), 4) for sc in score]
                print("%s %s %s %s %s" % (data_name, model, score[0], score[1], score[2]))
                line = "%s %s %s %s %s" % (data_name, model, score[0], score[1], score[2])
                fp.write(line+'\n')
    fp.close()
