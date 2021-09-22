import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from sklearn.decomposition import PCA

DATASET_PATH = Path(__file__).parent
ORIGINAL_PATH = Path(__file__).parent / 'original'
FOLDS_PATH = Path(__file__).parent / 'folds'

def read_data(name, source, sep=','):
    original_path = DATASET_PATH / source / name
    if source == "UCI":
        X, y = read_uci_data(original_path, name, sep)
    elif source == "KEEL" or source == "KEEL2":
        X, y = read_keel_data(original_path, name)
    elif source == "DL":
        X, y = read_MNIST_data(name=name)
    else:
        print("source=%s error." % source)
        X, y = [], []
    return X, y

def read_uci_data(original_path, name, sep=','):
    begin = 0
    y_index = -1
    name = name.split('.')[0]
    if name == "bands":
        begin = 20
    elif name == "wdbc":
        begin = 2
        y_index = 1
    elif name == "wisconsin":
        begin = 1
    X, y = [], []

    with open(original_path) as f:
        for line in f:
            if line.find('?') != -1:
                continue #skip
            line = [i for i in line.split(sep)]
            if len(line[begin:]) != 0:
                if name == "wdbc":
                    X.append([float(i) for i in line[begin:]])
                else:
                    X.append([float(i) for i in line[begin:-1]])
                y.append(line[y_index])
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y)
    
    if len(y[y==1]) > len(y[y==0]):
         y = 1 - y

    # print("%s: attr_len=%d, cls_num=%d, size=%d, min=%d, maj=%d, %s" % (name, len(X[0]), np.unique(y).shape[0], len(y), \
        # len(y[y==1]), len(y[y==0]), Counter(y)))
    return np.array(X), np.array(y)

def read_keel_data(original_path, name):
    metadata_lines = 0
    with open(original_path) as f:
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1

                if line.startswith('@input'):
                    inputs = [l.strip() for l in line[8:].split(',')]
                elif line.startswith('@output'):
                    outputs = [l.strip() for l in line[8:].split(',')]
            else:
                break

    df = pd.read_csv(original_path, skiprows=metadata_lines, header=None)
    df.columns = inputs + outputs
    df = pd.concat([pd.get_dummies(df[inputs]), df[outputs]], axis=1)
    # get_dummies特征提取，枚举离散属性

    matrix = df.values #df.as_matrix()
    X, y = matrix[:, :-1], matrix[:, -1]
    try:  
        y = [_y.strip() for _y in y]
        y = np.array(y)
    except:
        pass
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.array(y)
    if len(y[y==1]) > len(y[y==0]):
        y = 1 - y
    # print("%s: attr_len=%d, cls_num=%d, size=%d, min=%d, maj=%d, %s" % (name, len(inputs), np.unique(y).shape[0], len(y), \
    #     len(y[y==1]), len(y[y==0]), Counter(y)))
    return X, y

def read_MNIST_data(folder='DL', name='0-1'):
    cls = [int(c) for c in name.split('-')]
    original_path = DATASET_PATH / folder / "MNIST"
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    import gzip, os
    paths = []
    for fname in files:
        paths.append(os.path.join(original_path, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = PCA(n_components=30).fit_transform(x_test)
    X = x_test[y_test == cls[0]]
    #y = y_test[y_test == cls[0]]
    y = [0]*X.shape[0]
    for i in range(1, len(cls)):
        _x = x_test[y_test == cls[i]]
        X = np.concatenate([X, _x], axis=0)
        y = np.append(y, [1]*_x.shape[0])
    classes = np.unique(y)
    sizes = [sum(y == c) for c in classes]
    print("classes:", classes, "sizes:", sizes)
    return X, y

