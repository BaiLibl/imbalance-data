from DPC import DPC
import numpy as np
from scipy.spatial.distance import euclidean

def rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def distance(x, y):
    return np.sum(np.abs(x - y))

def neighbor_samples(X, Y, nn=10):
    D = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            d = euclidean(X[i], Y[j])
            D[i][j] = d
    neighbor = []
    for i in range(len(X)):
        indices = np.argsort(D[i,:])[:nn] #计算两两之间的距离，取最近邻的下标
        if len(indices) == 0:
            print(X.shape, Y.shape, i, indices, D[i,:])
        neighbor.append(indices)
    return D, neighbor

class NonLinear4:
    def __init__(self, minCls=None, min_r=0.01, cluster_num=10, scale1=4.0, scale2=5.0, k_neighbor=5, dth=0.5):
        self.min_r = min_r
        self.steps   = 50
        self.minCls  = minCls
        self.scale1 = scale1 # unsafe
        self.scale2 = scale2 #safe
        self.k_neighbor = k_neighbor
        self.cluster_num = cluster_num
        self.dth = dth

        #print(scale1, scale2, k_neighbor, cluster_num)
    
    def divide_minority(self, minority_samples, neighborMin, k_neighbors, cluster_num, dth=0.5):
        n = len(minority_samples)
        cluster = DPC(minority_samples, "minority_samples", cluster_num, 1, False, 0).fit()
        reverse_cluster = dict()
        for k,v in cluster.items():
            for _id in v:
                reverse_cluster[_id] = k # sample _id belongs to cluseter-k

        unsafe_sample = []
        safe_sample = []

        for i in range(n):
            neighbor = neighborMin[i][:k_neighbors] # i's neighbor
            cluster_id = reverse_cluster[i]
            k_xi = (len(set(neighbor).intersection(set(cluster[cluster_id]))))
            threshold = k_xi / float(k_neighbors)
            if threshold >= dth:
                safe_sample.append(i)
            else:
                unsafe_sample.append(i)
        return unsafe_sample, safe_sample
        

    def fit_sample(self, data, labels, numSamples):
        if self.minCls == None:
            self.minCls = np.argmin(np.bincount(labels.astype(int)))

        trnMajData = data[np.where(labels!=self.minCls)[0], :]
        trnMinData = data[np.where(labels==self.minCls)[0], :]
        
        synthData  = np.empty([0, data.shape[1]]) 
        disMin, neighborMin = neighbor_samples(trnMinData, trnMinData)
        disMaj, neighborMaj = neighbor_samples(trnMinData, trnMajData)

        unsafe, safe = self.divide_minority(trnMinData, neighborMin, self.k_neighbor, self.cluster_num, self.dth)
        # print(len(safe), len(unsafe))

        safe_region = []
        for j in safe:
            min_neighbor = list(neighborMin[j][:self.k_neighbor]) + [j]
            pair = [j, j]
            max_radiu = 0
            for i in min_neighbor:
                for m in min_neighbor:
                    if disMin[i][m] > max_radiu:
                        pair = [i, m]
                        max_radiu = disMin[i][m]
            center = (trnMinData[pair[0]] + trnMinData[pair[1]]) / 2.0
            h = max_radiu
            safe_region.append([j, center, h])
        
                
        unsafe_region = []
        prob = []
        for j in unsafe:
            nearMaj_id = neighborMaj[j][0]
            nearMaj_ds = disMaj[j][nearMaj_id]
            nearMin_id = neighborMin[j][0]
            nearMin_ds = disMin[j][nearMin_id]

            seed = (trnMajData[nearMaj_id] + trnMinData[nearMin_id] + 2 * trnMinData[j]) / 4
            h = nearMaj_ds + nearMin_ds #似乎很紧密
            unsafe_region.append([j, seed, h])
            c = 1.0
            for i in range(trnMajData.shape[0]):
                if disMaj[j][i] <= h:
                    c += 1
            prob.append(h / c)
        if len(prob) > 0:
            prob = prob / sum(prob)
        
        region_prob = [len(safe) / trnMinData.shape[0], len(unsafe) / trnMinData.shape[0]] # 采样概率和样本成正比

        np.random.seed(0)
        i = 0
        while synthData.shape[0] < numSamples:
            i += 1
            if i > self.steps * numSamples / 2:
                break # finish
            region = np.random.choice(len(region_prob), 1, p=region_prob)[0]
            if region == 0: # safe region 等概率生成
                j = np.random.choice(len(safe_region),1)[0]
                id, seed, h = safe_region[j]
                _std = h / self.scale2
                r = h / 2.0
            else:  # unsafe 按照权重生成
                j = np.random.choice(len(prob), 1, p=prob)[0]
                id, seed, h = unsafe_region[j]
                _std = h / self.scale1
                r = h / 4.0
            for _ in range(self.steps):
                point = np.random.normal(np.array(seed, dtype='float64'), _std, trnMinData.shape[1])
                # print(h,euclidean(point, seed))
                if euclidean(point, seed) <= r and euclidean(point, trnMinData[id]) >= self.min_r:
                    synthData = np.append(synthData, point.T.reshape((1, len(point))), axis=0)
                    break

        sampled_data = np.concatenate([np.array(synthData), data])
        sampled_labels = np.append([self.minCls]*len(synthData),labels)

        return sampled_data, sampled_labels
