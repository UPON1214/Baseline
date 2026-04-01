import logging
import numpy as np
import math


def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def next_batch(X1, X2, X3, X4, X1_index, X2_index, X3_index, X4_index, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]

        batch_x1_index = X1_index[start_idx: end_idx, ...]
        batch_x2_index = X2_index[start_idx: end_idx, ...]
        batch_x3_index = X3_index[start_idx: end_idx, ...]
        batch_x4_index = X4_index[start_idx: end_idx, ...]

        '''# 创建索引数组，这里我们返回起始索引和结束索引（或者只返回起始索引，看您的需求）
        x1_index = np.arange(start_idx, end_idx)
        # 假设其他视图的索引与X1相同（如果不同，需要分别计算）
        x2_index = x1_index.copy()  # 或者根据X2的具体情况生成不同的索引
        x3_index = x1_index.copy()  # 同上
        x4_index = x1_index.copy()  # 同上

        # 批次编号，这里我们返回从0开始的批次编号（或者继续返回i+1，看您的需求）
        batch_No = i  # 如果需要从1开始，则使用i+1'''

        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x1_index, batch_x2_index, batch_x3_index, batch_x4_index, (i + 1))

def next_batch_COIL20(X1, X2, X3, X1_index, X2_index, X3_index, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]


        batch_x1_index = X1_index[start_idx: end_idx, ...]
        batch_x2_index = X2_index[start_idx: end_idx, ...]
        batch_x3_index = X3_index[start_idx: end_idx, ...]


        '''# 创建索引数组，这里我们返回起始索引和结束索引（或者只返回起始索引，看您的需求）
        x1_index = np.arange(start_idx, end_idx)
        # 假设其他视图的索引与X1相同（如果不同，需要分别计算）
        x2_index = x1_index.copy()  # 或者根据X2的具体情况生成不同的索引
        x3_index = x1_index.copy()  # 同上
        x4_index = x1_index.copy()  # 同上

        # 批次编号，这里我们返回从0开始的批次编号（或者继续返回i+1，看您的需求）
        batch_No = i  # 如果需要从1开始，则使用i+1'''

        yield (batch_x1, batch_x2, batch_x3, batch_x1_index, batch_x2_index, batch_x3_index, (i + 1))

def next_batch_handwritten(X1, X2, X3, X4, X5, X1_index, X2_index, X3_index, X4_index, X5_index, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_x5 = X5[start_idx: end_idx, ...]

        batch_x1_index = X1_index[start_idx: end_idx, ...]
        batch_x2_index = X2_index[start_idx: end_idx, ...]
        batch_x3_index = X3_index[start_idx: end_idx, ...]
        batch_x4_index = X4_index[start_idx: end_idx, ...]
        batch_x5_index = X5_index[start_idx: end_idx, ...]

        '''# 创建索引数组，这里我们返回起始索引和结束索引（或者只返回起始索引，看您的需求）
        x1_index = np.arange(start_idx, end_idx)
        # 假设其他视图的索引与X1相同（如果不同，需要分别计算）
        x2_index = x1_index.copy()  # 或者根据X2的具体情况生成不同的索引
        x3_index = x1_index.copy()  # 同上
        x4_index = x1_index.copy()  # 同上

        # 批次编号，这里我们返回从0开始的批次编号（或者继续返回i+1，看您的需求）
        batch_No = i  # 如果需要从1开始，则使用i+1'''

        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x1_index, batch_x2_index, batch_x3_index, batch_x4_index, batch_x5_index, (i + 1))


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        print('ACC:'+ str(arg[0]))
        print('NMI:'+ str(arg[1]))
        print('ARI:'+ str(arg[2]))
        output = "ACC {:.3f} std {:.3f} NMI {:.3f} std {:.3f} ARI {:.3f} std {:.3f}".format( np.mean(arg[0]),
                                                                                             np.std(arg[0]),
                                                                                             np.mean(arg[1]),
                                                                                             np.std(arg[1]),
                                                                                             np.mean(arg[2]),
                                                                                             np.std(arg[2]))
    elif len(arg) == 1:
        print(arg)
        output = "ACC {:.3f} std {:.3f}".format(np.mean(arg), np.std(arg))

    print(output)
    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
