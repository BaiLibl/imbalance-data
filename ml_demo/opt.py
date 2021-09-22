from hyperopt.pyll_utils import hp_choice
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
# 这里的warnings实在太多了，我们加入代码不再让其显示
import warnings
from sklearn.utils import resample
warnings.filterwarnings("ignore")

import datasets
from new4 import NonLinear4
from new5 import NonLinear5
from sklearn.preprocessing import StandardScaler
from utils import demo5_best_nn


def search_best_param(data_name, folder):
    clf_name = 'SVM'
    X, y = datasets.read_data(data_name, folder)
    X = StandardScaler().fit_transform(X)

    def hyperopt_model_score_dtree(params):
        X_ = X[:]
        resample = NonLinear5(**params)
        numSamples = np.sum(y==0)-np.sum(y==1)
        X_res, y_res = resample.fit_sample(X_, y, numSamples)
        clf = SVC(kernel="linear", C=0.025)
        return cross_val_score(clf, X_res, y_res).mean()

    def fn_dtree(params):
        acc = hyperopt_model_score_dtree(params)
        return -acc

    region = {
    'scale1':[4.0, 5.0, 6.0, 7.0],
    'scale2':[5.0, 6.0, 7.0],
    'k_neighbors':[3,5,7],
    'k_neighbors2':[3,5,7],
    'cluster_num':[4,6,8],
    'cluster_num2':[4,6,8],
    'dth':[0.5,0.6,0.8],
    'dth2':[0.5,0.8,0.9]
    }

    space_dtree = {}
    for key, value  in region.items():
        space_dtree[key] = hp.choice(key, value)

    # 为可视化做准备
    trials = Trials()
    best = fmin(fn=fn_dtree, space=space_dtree, algo=tpe.suggest, max_evals=500, trials=trials)
    # print('best:', best) 

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    info = "%s %s-%s:" % (time_str, data_name.split('.')[0], clf_name)
    res = {k:region[k][v] for k, v in best.items()}
    print(info, res)

    # parameters = list(region.keys())  # decision tree
    # cols = len(parameters)
    # f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
    # cmap = plt.cm.jet
    # for i, val in enumerate(parameters):
    #     xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    #     ys = [-t['result']['loss'] for t in trials.trials]
    #     ys = np.array(ys)
    #     axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
    #     axes[i].set_title(val)
    # plt.savefig(data_name+".pdf")
    # plt.close()

if __name__ == '__main__':
    for dirs in ["KEEL2"]:
        for data_name in os.listdir(dirs):
            if data_name in demo5_best_nn.keys():
                continue
            print(data_name)
            search_best_param(data_name, dirs)

