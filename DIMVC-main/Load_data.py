import numpy as np
#from keras_preprocessing import image
# from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import normalize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings("ignore")

path = './data'

def _resolve_data_file(filename):
    """
    依次在以下位置查找数据文件：
    1) ./data/filename
    2) 当前脚本同目录下的 filename
    3) /mnt/data/filename（当前对话上传文件常见位置）
    """
    candidates = [
        os.path.join(path, filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join('/mnt/data', filename),
    ]
    for file_path in candidates:
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"Cannot find {filename}. Tried: {candidates}")


def _to_2d_float(array):
    """
    将每个视图统一整理成 [N, D] 形式：
    - 图像视图: [N, H, W] -> [N, H*W]
    - 向量视图: [N, D] 保持不变
    同时转换为 float32；若原始数据为整型图像，则归一化到 [0, 1]。
    """
    arr = np.asarray(array)
    original_dtype = arr.dtype

    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    arr = arr.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        arr /= 255.0

    return arr
#以上代码为适应COIL20数据集添加


def Caltech(missrate=0.5):
    # Different IMVC methods might utilize different data preprocessing functions, 
    # including but not limited to standardization, regularization, and min-max normalization, etc.
    Data = scio.loadmat(path + "/Caltech.mat")     # Upload the dataset, which is already be pre-processed.
    x1 = Data['X1']
    x2 = Data['X2']
    Y = Data['Y']
    Y = Y.reshape(Y.shape[0])
    size = Y.shape[0]
    X, Y, index = Form_Incomplete_Data(missrate=missrate, X=[x1, x2], Y=[Y, Y])
    return X, Y, size, index

def COIL20(missrate=0.5):
    """
    COIL20 = {'N': 1440, 'K': 20, 'V': 3, 'n_input': [1024, 1024, 324]}
    说明：
        - 前两个视图在 mat 文件中为 [1440, 32, 32] 图像，需要拉平成 1024 维
        - 第三个视图已经是 [1440, 324]
    """
    data_file = _resolve_data_file("COIL20.mat")
    Data = scio.loadmat(data_file)

    raw_X = Data['X']
    Y = Data['Y'].reshape(-1).astype(np.int64)

    views = []
    if isinstance(raw_X, np.ndarray) and raw_X.dtype == object:
        # MATLAB cell 常见格式: shape = (1, V)
        for i in range(raw_X.shape[1]):
            views.append(_to_2d_float(raw_X[0, i]))
    else:
        # 兜底：若不是 object/cell，则按普通数组处理
        raw_X = np.asarray(raw_X)
        if raw_X.ndim == 3:
            for i in range(raw_X.shape[0]):
                views.append(_to_2d_float(raw_X[i]))
        else:
            raise ValueError(f"Unsupported COIL20 X format: shape={raw_X.shape}, dtype={raw_X.dtype}")

    expected_dims = [1024, 1024, 324]
    actual_dims = [v.shape[1] for v in views]
    if actual_dims != expected_dims:
        raise ValueError(
            f"COIL20 input dimensions mismatch. Expected {expected_dims}, got {actual_dims}"
        )

    size = Y.shape[0]
    X, Y, index = Form_Incomplete_Data(missrate=missrate, X=views, Y=[Y] * len(views))
    return X, Y, size, index



def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    import random
    random.shuffle(t)
    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    for v in range(view_num):
        for i in range(size):
            Xtmp[v][i] = X[v][t[i]]
            Ytmp[v][i] = Y[v][t[i]]
    X = Xtmp
    Y = Ytmp

    # complete data index
    index0 = np.linspace(0, (1 - missrate) * size - 1, num=int((1 - missrate) * size), dtype=int)
    missindex = np.ones((int(missrate * size), view_num))
    print(missindex.shape)
    # incomplete data index
    index = []
    for i in range(missindex.shape[0]):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        # print(missdata)
        missindex[i, missdata] = 0
    # print(missindex)
    for i in range(view_num):
        index.append([])
    miss_begain = (1 - missrate) * size
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index[j].append(int(miss_begain + i))
    # print(index)
    maxmissview = 0
    for j in range(view_num):
        if maxmissview < len(index[j]):
            print(len(index[j]))
            maxmissview = len(index[j])
    print(maxmissview)
    # add some incomplete views' data index to equal for convenience
    for j in range(view_num):
        flag = np.random.randint(0, high=size, size=maxmissview - len(index[j]))
        index[j] = index[j] + list(flag)
    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        print(X[j].shape)
        Y[j] = Y[j][index[j]]
        print(Y[j].shape)
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, index


def load_data_conv(dataset, missrate):
    print("load:", dataset)
    if dataset == 'Caltech':                # Caltech
        return Caltech(missrate=missrate)
    elif dataset == 'COIL20':
        return COIL20(missrate=missrate)
    else:
        raise ValueError('Not defined for loading %s' % dataset)
