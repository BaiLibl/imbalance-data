#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reference: https://github.com/ahazxm/DPC

import sys
import matplotlib
import scipy.cluster.hierarchy as sch
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from multiprocessing import Process, Queue, Pool
import matplotlib.font_manager as f
from scipy.spatial.kdtree import distance_matrix
matplotlib.rcParams['axes.unicode_minus'] = False #支持正常显示负号

class DPC(object):
    """
    1. 读取点
    2. 计算距离
    3. 计算截断距离 dc
    4. 计算局部密度 rho
    5. 计算 delta
    6. 确定 聚类中心
    7. 聚类
    8. 绘图
    """

    def __init__(self, X, data_name="demo", n=0, dc_percent=1, use_halo=False, plot=None, dis_matrix=None):
        self.X = X
        self.data_name = data_name
        self.cluster_num = n
        self.dc_percent = dc_percent
        self.use_halo = use_halo
        self.plot = plot
        self.dis_matrix = dis_matrix
    
    def fit(self):
        # print("CFS begins ...")
        # 读取点，计算距离
        # print("calc_distance")
        d_dist, d_list, min_dis, max_dis, X_len = self.calc_distance()
        # 计算截断聚类 dc
        # print("get_dc")
        dc = self.get_dc_static(d_list,2)
        # print("dc = %f" % dc)
        # dc = self.get_dc(d_dist, d_list, min_dis, max_dis, X_len, self.dc_percent)
        # print("dc = %f" % dc)
        # 计算 rho
        # print("get_rho")
        rho = self.get_rho(d_dist, X_len, dc)
        # 计算 delta
        # print("get_delta")
        delta = self.get_delta(d_dist, X_len, rho)
        # 确定聚类中心
        # print("get_center")
        center, gamma = self.get_center(rho, delta, self.cluster_num) #取前n个作为聚类中心
        # 聚类
        cluster = self.assign(d_dist, center, X_len)
        if self.plot is None:
            print('here')
            # 单数据集绘制图分布
            fig, axes = plt.subplots(1, 2, figsize=(18.6, 6.2))
            fig.subplots_adjust(left=0.05, right=0.95)
            axes[0].set_title('dc-' + str(round(dc,4)))
            self.draw_roh_delta(rho, delta, center, axes[0])
            self.draw_gamma(rho, delta, axes[1])
            plt.show()
        return cluster
        '''
        halo = []
        if use_halo:
            # halo
            cluster, halo = self.get_halo(d_dist, rho, cluster, center, dc, X_len)
        if plot is None:
            # 单数据集绘制图分布
            fig, axes = plt.subplots(1, 3, figsize=(18.6, 6.2))
            fig.subplots_adjust(left=0.05, right=0.95)
            axes[0].set_title('dc-' + str(round(dc,4)))
            self.draw_roh_delta(rho, delta, center, axes[0])
            self.draw_gamma(rho, delta, axes[1])
            self.draw_cluster(data_name, cluster, halo, X, axes[2])
            plt.show()
        else:
            # 全部数据集画图
            self.draw_cluster(data_name, cluster, halo, X, plot)
        '''


    def calc_distance(self):
        X_len = len(self.X)
        d = pd.DataFrame(np.zeros((X_len, X_len)))  # 距离矩阵
        if self.dis_matrix is None:
            dis = sch.distance.pdist(self.X, 'euclidean')  # 欧式距离
            n = 0
            for i in range(X_len):
                for j in range(i + 1, X_len):
                    d.at[i, j] = dis[n]
                    d.at[j, i] = d.at[i, j]
                    n += 1
        else:
            d = pd.DataFrame(self.dis_matrix)
            dis = self.dis_matrix.reshape(-1)

        min_dis = d.min().min()
        max_dis = d.max().max()

        # print(min_dis, max_dis, d[0][:10],'aaaa',X_len)

        return d, dis, min_dis, max_dis, X_len
    
    def get_dc_static(self, d, percent):
        # get per% value after sorting d in ascending order
        d_sort = sorted(list(d))
        
        for _begin in range(len(d)):
            if d_sort[_begin] > 0: break
        _index = int(percent * (len(d) - _begin) / 100) + _begin
        # print(d_sort[_begin:_begin+10])
        dc = d_sort[_index] #跳过前面=0的dis value
        return dc

    def get_dc(self, d, d_list, min_dis, max_dis, X_len, percent):
        """ 求解截断距离

        Desc:
            二分查找
        Args:
            d: 距离矩阵
            d_list: 上三角矩阵
            min_dis: 最小距离
            max_dis: 最大距离
            X_len: 点数
            percent: 占全部点的百分比数
            
        Returns:
            dc: 截断距离

        """
        # print('Get dc')
        lower = percent / 100
        upper = (percent + 1) / 100
        print(lower, upper)
        c = 1
        if True:
            while 1:
                c = c + 1
                dc = (min_dis + max_dis) / 2
                
                neighbors_percent = len(d_list[d_list < dc]) / (((X_len - 1) ** 2) / 2)  # 上三角矩阵
                print(c, dc, neighbors_percent)
                if neighbors_percent >= lower and neighbors_percent <= upper:
                    return dc
                elif neighbors_percent > upper:
                    max_dis = dc
                elif neighbors_percent < lower:
                    min_dis = dc

    def get_rho(self, d, X_len, dc):
        # print('Get rho')
        rho = np.zeros(X_len)
        for i in range(X_len):
            for j in range(X_len):
                rho[i] += math.exp(-(d.at[i, j] / dc) ** 2)
        return rho

    def get_delta(self, d, X_len, rho):
        # print('Get delta')
        delta = np.zeros(X_len)
        if True:  # 不考虑rho相同且同为最大
            for i in range(X_len):
                rho_i = rho[i]
                j_list = np.where(rho > rho_i)[0]  # rho 大于 rho_i 的点们
                if len(j_list) == 0:
                    delta[i] = d.loc[i, :].max()
                else:
                    min_dis_index = d.loc[i, j_list].idxmin()  # 密度大于i且距离最近
                    delta[i] = d.at[i, min_dis_index]
        return delta

    def get_center(self, rho, delta, n):
        """ 获取聚类中心点
        Returns:
            center: 聚类中心列表
            gamma: rho * delta

        """
        gamma = rho * delta
        gamma = pd.DataFrame(gamma, columns=['gamma']).sort_values('gamma', ascending=False)
        center = np.array(gamma.index)[:n]  # 取前n个点做中心点
        #  center = gamma[gamma.gamma > threshold].loc[:, 'gamma'].index
        return center, gamma

    def assign(self, d, center, X_len):
        # print('Assign')
        cluster = dict()  # center: X
        for i in center:
            cluster[i] = []

        # 最近中心分配
        for i in range(X_len):
            c = d.loc[i, center].idxmin()
            cluster[c].append(i)

        return cluster

    def get_halo(self, d, rho, cluster, center, dc, X_len):
        # print('Get halo')
        all_X = set(list(range(X_len)))  # 所有的点
        self.border_b = []
        for c, X in cluster.items():
            others_X = list(set(all_X) - set(X))  # 属于其他聚类的点
            border = []
            for p in X:
                if d.loc[p, others_X].min() < dc:  # 到其他聚类的点的距离小于dc
                    border.append(p)
            if len(border) != 0:
                #  rho_b = rho[border].max()  # 边界域中密度最大的值
                point_b = border[rho[border].argmax()]  # 边界域中密度最大的点
                self.border_b.append(point_b)
                rho_b = rho[point_b]  # 边界域最大密度
                filter_X = np.where(rho >= rho_b)[0]  # 筛选可靠性高的点
                X = list(set(filter_X) & set(X))  # 该聚类中可靠性高的点
                cluster[c] = X
        # halo
        cluster_X = set()
        for c, X in cluster.items():
            cluster_X = cluster_X | set(X)
        halo = list(set(all_X) - cluster_X)  # 光晕点
        return cluster, halo

    def draw_roh_delta(self, rho, delta, center, plot):
        plot.scatter(rho, delta, label='rho-delta', c='k', s=5)
        plot.set_xlabel('rho')
        plot.set_ylabel('delta')
        center_rho = rho[center]
        center_delta = delta[center]
        np.random.seed(6)
        colors = np.random.rand(len(center), 3)
        plot.scatter(center_rho, center_delta, c=colors)
        plot.legend()

    def draw_gamma(self, rho, delta, plot):
        gamma = pd.DataFrame(rho * delta, columns=['gamma']).sort_values('gamma', ascending=False)
        plot.scatter(range(len(gamma)), gamma.loc[:, 'gamma'], label='gamma', s=5)
        #  plot.hlines(avg, 0, len(gamma), 'b', 'dashed')
        plot.set_xlabel('n')
        plot.set_ylabel('gamma')
        plot.set_title('gamma')
        plot.legend()

    def draw_cluster(self, title, cluster, halo, X, plot):
        cluster_X = dict()
        colors = dict()
        np.random.seed(10)
        for k, v in cluster.items():
            cluster_X[k] = X.loc[cluster[k], :]
            colors[k] = np.random.rand(3)
        for k, v in cluster_X.items():
            plot.scatter(v.loc[:, 'x'], v.loc[:, 'y'], c=colors[k], alpha=0.5)
            plot.scatter(v.at[k, 'x'], v.at[k, 'y'], c=colors[k], s=np.pi * 10 ** 2)
        if len(halo) != 0:
            noise_pointer = X.loc[halo, :]
            plot.scatter(noise_pointer.loc[:, 'x'], noise_pointer.loc[:, 'y'], c='k')
            border_b = X.loc[self.border_b, :]
            plot.scatter(border_b.loc[:, 'x'], border_b.loc[:, 'y'], c='k', s=np.pi * 5 ** 2)
        plot.set_title(title)

    def draw_X(self, X, center=[]):
        X.plot(x='x', y='y', kind='scatter')
        plt.scatter(X.loc[:, 'x'], X.loc[:, 'y'], c='b', alpha='0.5')
        if len(center) != 0:
            center_p = X.loc[center, :]
            plt.scatter(center_p.loc[:, 'x'], center_p.loc[:, 'y'], c='r', s=np.pi * 10 ** 2)
        plt.show()


# 进程函数
# path | title | N: 聚类数 | dc method | dc per | rho method | delta method | use_halo | plot
def cluster(path, data_name, n, dc_method=0, dc_percent=1, rho_method=1, delta_method=1, use_halo=False, plot=None):
    cluster = DPC(path, data_name, n, dc_percent, use_halo, plot).fit()
    for k,v in cluster.items():
        print(k,len(v), v)

def demo():
    path = sys.path[0] + '/dataset/'
    X = pd.read_csv(path + "compound.dat", sep='\t', usecols=[0, 1])

    cluster(X, 'compound', 5, 0, 4)
    plt.show()


# if __name__ == '__main__':    
#     import datasets
#     from sklearn.preprocessing import StandardScaler
#     X, y = datasets.read_data("abalone9-18.dat", "KEEL") #1-positive,0-negtive
#     X = StandardScaler().fit_transform(X)
#     minority = X[np.where(y==1)[0], :]
#     DPC(X, "minority_samples", 2, 1, False, None).fit()
