import os
import scipy.io
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import random
import warnings
warnings.filterwarnings("ignore")


# 需要h5py读取
ALL_data = dict(
    Caltech101_7= {1: 'Caltech101_7', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [1984, 512, 928, 254, 40]},
    HandWritten = {1: 'handwritten1031_v73', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6]},
    Caltech101_20={1: 'Caltech101-20_v73', 'N': 2386, 'K': 20, 'V': 6, 'n_input': [48, 40, 254, 1984, 512, 928]},
    LandUse_21 = {1: 'LandUse_21_v73','N': 2100, 'K': 21, 'V': 3, 'n_input': [20,59,40]},
    Scene_15 = {1: 'Scene_15','N': 4485, 'K': 15, 'V': 3, 'n_input': [20,59,40]},
    COIL20 = {1: 'COIL20','N': 1440, 'K': 20, 'V': 3, 'n_input': [1024,1024,324]},
    leaves_3v={1: 'leaves_3v', 'N': 1600, 'K': 100, 'V': 3, 'n_input': [64, 64, 64]},
    ORL = {1: 'ORL', 'N': 400, 'K': 40, 'V': 4, 'n_input': [512, 59, 864, 254]},
    MSRC_v1= {1: 'MSRC_v1', 'N': 210, 'K': 7, 'V': 5, 'n_input': [24, 576, 512, 256, 254]},
    ALOI_100 = {1: 'ALOI_100_7', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125]},
    YouTubeFace10_4Views={1: 'YTF10_4', 'N': 38654, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640]},
    AWA={1: 'AWA_73', 'N': 10158, 'K': 50, 'V': 7, 'n_input': [2688, 2000, 2000, 2000, 2000, 4096,4096]},
    # AWA={1: 'AWA_73', 'N': 30735, 'K': 50, 'V': 6, 'n_input': [2688, 2000, 252, 2000, 2000, 2000]},
    EMNIST_digits_4Views={1: 'EMNIST_digits_4Views_v73', 'N': 280000, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640]}
)
# ALL_data =Caltech101_7 = {1: 'Caltech101_7', 'N': 1474, 'K': 6, 'V': 56, 'n_input': [1984, 512, 928, 254, 40]}

path = './Dataset/'


def get_mask(view_num, alldata_len, missing_rate):
    '''生成缺失矩阵：
    view_num为视图数
    alldata_len为数据长度
    missing_rate为缺失率
    return 缺失矩阵 alldata_len*view_num大小的0和1矩阵
    '''
    missindex = np.ones((alldata_len, view_num))
    b=((10 - 10*missing_rate)/10) * alldata_len
    miss_begin = int(b)  #将b转换成整数 作为缺失开始的索引
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num,
                                     size=view_num - 1)

        missindex[i, missdata] = 0

    return missindex


def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    np.random.seed(1)

    size = len(Y[0])
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    ##########################获取缺失矩阵###########################################
    missindex = get_mask(view_num, size, missrate)

    index_complete = []
    index_partial = []
    for i in range(view_num):
        index_complete.append([])
        index_partial.append([])
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index_complete[j].append(i)
            else:
                index_partial[j].append(i)

    filled_index_com = []
    for i in range(view_num):
        filled_index_com.append([])
    max_len = 0
    for v in range(view_num):
        if max_len < len(index_complete[v]):
            max_len = len(index_complete[v])
    for v in range(view_num):
        if len(index_complete[v]) < max_len:
            diff_len = max_len - len(index_complete[v])

            diff_value = random.sample(index_complete[v], diff_len)
            filled_index_com[v] = index_complete[v] + diff_value
        elif len(index_complete[v]) == max_len:
            filled_index_com[v] = index_complete[v]

    filled_X_complete = []
    filled_Y_complete = []
    for i in range(view_num):
        filled_X_complete.append([])
        filled_Y_complete.append([])
        filled_X_complete[i] = X[i][filled_index_com[i]]
        filled_Y_complete[i] = Y[i][filled_index_com[i]]
    for v in range(view_num):

        X[v] = torch.from_numpy(X[v])
        filled_X_complete[v] = torch.from_numpy(filled_X_complete[v])

    return X, Y, missindex, filled_X_complete, filled_Y_complete, index_complete, index_partial

'''def load_data(dataset, missrate):
    #data = h5py.File(path + dataset[1] + ".mat")
    # 先尝试h5py
    try:
        data = h5py.File(dataset, 'r')
        return data
    except:
        # 再尝试scipy
        try:
            data = scipy.io.loadmat(dataset)
            return data
        except Exception as e:
            print(f"两种方式都失败: {e}")
            return None
    X = []
    Y = []
    Label = np.array(data['Y']).T

    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)
    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)

    return X, Y, missindex, X_com, Y_com, index_com, index_incom'''

#这个可以跑通scene15
def load_data(dataset_config, missrate):
    """加载数据"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集名称
    dataset_name = dataset_config[1]

    # 可能的文件路径列表
    possible_paths = [
        os.path.join(current_dir, 'Dataset', dataset_name + ".mat"),
        os.path.join(current_dir, 'Dataset', dataset_name + "_v73.mat"),
        os.path.join(current_dir, dataset_name + ".mat"),
        os.path.join('./Dataset/', dataset_name + ".mat"),
        os.path.join('./data/', dataset_name + ".mat"),
    ]

    # 查找存在的文件
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            print(f"找到数据文件: {file_path}")
            break

    if file_path is None:
        print("尝试的文件路径:")
        for p in possible_paths:
            print(f"  {p}")
        raise FileNotFoundError(f"找不到数据文件: {dataset_name}")

    # 读取数据
    try:
        # 先尝试h5py
        data = h5py.File(file_path, 'r')
        print("使用h5py读取成功")
    except:
        try:
            # 再尝试scipy
            data = scipy.io.loadmat(file_path)
            print("使用scipy.io.loadmat读取成功")
        except Exception as e:
            print(f"两种方式都失败: {e}")
            raise

    # 处理数据...
    X = []
    Y = []
    Label = np.array(data['gt'])#.T
    #Label = Label.reshape(Label.shape[0])
    Label = Label.flatten()
    mm = MinMaxScaler()

    # 方法1: 查找X1, X2, X3...等视图
    view_data_found = False
    for i in range(1, 10):  # 假设最多9个视图
        view_key = f'X{i}'
        if view_key in data:
            view_data = data[view_key].astype(np.float32)
            if view_data.shape[0] != len(Label):
                view_data = view_data.T
            print(f"找到视图{i}: {view_key} 形状: {view_data.shape}")

            std_view = mm.fit_transform(view_data)
            X.append(std_view)
            Y.append(Label)
            view_data_found = True

    # 方法2: 如果找到了视图数据
    if view_data_found:
        print(f"通过X1,X2,...键找到 {len(X)} 个视图")
    else:
        # 方法3: 尝试从X cell数组中提取
        if 'X' in data and isinstance(data['X'], np.ndarray):
            print(f"X的数据类型: {data['X'].dtype}")
            print(f"X的形状: {data['X'].shape}")

            # 修复这里：使用object而不是np.object
            if data['X'].dtype == object:  # 修改这里
                print("从X cell数组中提取视图...")
                X_cell = data['X']
                for i in range(X_cell.shape[1]):
                    cell_item = X_cell[0, i]
                    if isinstance(cell_item, np.ndarray):
                        view_data = cell_item.astype(np.float32)
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                        print(f"cell视图{i + 1}形状: {view_data.shape}")

                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
            else:
                # X不是cell数组，可能是其他格式
                print("X不是cell数组，尝试其他处理方式...")
                X_data = data['X'].astype(np.float32)
                print(f"X_data形状: {X_data.shape}")

                # 如果X是2D数组，尝试根据配置分割
                if X_data.ndim == 2:
                    view_dims = dataset_config.get('n_input', [20, 59, 40])
                    total_dims = sum(view_dims)

                    if X_data.shape[1] == total_dims:
                        print(f"按配置分割视图: {view_dims}")
                        start_idx = 0
                        for dim in view_dims:
                            end_idx = start_idx + dim
                            view_data = X_data[:, start_idx:end_idx]
                            print(f"分割视图形状: {view_data.shape}")

                            std_view = mm.fit_transform(view_data)
                            X.append(std_view)
                            Y.append(Label)
                            start_idx = end_idx

    # 方法4: 尝试其他可能的视图键名
    if len(X) == 0:
        print("尝试其他可能的视图键名...")
        possible_view_keys = []
        for key in data.keys():
            if not key.startswith('__') and key not in ['Y', 'y', 'label', 'gnd']:
                possible_view_keys.append(key)

        print(f"可能的视图键: {possible_view_keys}")

        # 尝试按字母顺序或数字顺序排序
        possible_view_keys.sort()
        for key in possible_view_keys:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
                view_data = data[key].astype(np.float32)
                if view_data.shape[0] == len(Label) or view_data.shape[1] == len(Label):
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T
                    print(f"使用键 '{key}': 形状{view_data.shape}")

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)

    if len(X) == 0:
        print("数据文件详细内容:")
        for key in data.keys():
            if not key.startswith('__'):
                val = data[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: 形状{val.shape}, 类型{val.dtype}")
                else:
                    print(f"  {key}: {type(val)}")
        raise ValueError("无法提取视图数据。请检查.mat文件结构")

    print(f"成功加载 {len(X)} 个视图")
    print(f"每个视图特征维度: {[x.shape[1] for x in X]}")

    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(
        missrate=missrate, X=X, Y=Y
    )

    return X, Y, missindex, X_com, Y_com, index_com, index_incom

#这个可以跑通100leaves
'''def load_data(dataset_config, missrate):
    """加载数据"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集名称
    dataset_name = dataset_config[1]

    # 可能的文件路径列表
    possible_paths = [
        os.path.join(current_dir, 'Dataset', dataset_name + ".mat"),
        os.path.join(current_dir, 'Dataset', dataset_name + "_v73.mat"),
        os.path.join(current_dir, dataset_name + ".mat"),
        os.path.join('./Dataset/', dataset_name + ".mat"),
        os.path.join('./data/', dataset_name + ".mat"),
    ]

    # 查找存在的文件
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            print(f"找到数据文件: {file_path}")
            break

    if file_path is None:
        print("尝试的文件路径:")
        for p in possible_paths:
            print(f"  {p}")
        raise FileNotFoundError(f"找不到数据文件: {dataset_name}")

    # 读取数据
    try:
        # 先尝试h5py
        data = h5py.File(file_path, 'r')
        print("使用h5py读取成功")
        use_h5py = True
    except:
        try:
            # 再尝试scipy
            data = scipy.io.loadmat(file_path)
            print("使用scipy.io.loadmat读取成功")
            use_h5py = False
        except Exception as e:
            print(f"两种方式都失败: {e}")
            raise

    # 处理数据...
    X = []
    Y = []

    # 打印所有键名以便调试
    if not use_h5py:
        print("数据文件中的键:", [key for key in data.keys() if not key.startswith('__')])

    # 处理标签
    Label = None
    label_keys = ['gt', 'Y', 'y', 'label', 'gnd', 'L']
    for key in label_keys:
        if key in data:
            Label = data[key]
            print(f"使用键 '{key}' 作为标签")
            break

    if Label is None:
        raise KeyError("找不到标签数据")

    Label = np.array(Label)
    print(f"原始标签形状: {Label.shape}")
    Label = Label.flatten()
    print(f"处理后标签形状: {Label.shape}")
    print(f"标签样本数: {len(Label)}")
    print(f"标签唯一值: {np.unique(Label)}")

    mm = MinMaxScaler()

    # 方法1: 处理fea键（对于100leaves数据集）
    if 'fea' in data:
        print("找到'fea'键，处理多视图数据...")
        fea_data = data['fea']
        print(f"'fea'类型: {type(fea_data)}")

        if not use_h5py and isinstance(fea_data, np.ndarray) and fea_data.dtype == object:
            print("'fea'是cell数组，包含多个视图")
            # 遍历cell数组中的每个元素
            for i in range(fea_data.shape[1]):
                cell_item = fea_data[0, i]
                if isinstance(cell_item, np.ndarray):
                    view_data = cell_item.astype(np.float32)
                    # 确保正确的形状 (n_samples x n_features)
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T
                    print(f"视图{i + 1}形状: {view_data.shape}")

                    # 检查数据是否有效
                    if np.any(np.isnan(view_data)) or np.any(np.isinf(view_data)):
                        print(f"警告: 视图{i + 1}包含NaN或Inf值")
                        # 处理异常值
                        view_data = np.nan_to_num(view_data)

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)

        elif use_h5py:
            # h5py格式处理
            print("h5py格式的fea处理")
            # 需要根据实际h5py结构处理
            pass
        else:
            # fea是普通数组
            print("'fea'是普通数组")
            fea_data = fea_data.astype(np.float32)
            if fea_data.shape[0] != len(Label):
                fea_data = fea_data.T
            print(f"'fea'形状: {fea_data.shape}")

            # 检查是否应该分割为多视图
            view_dims = dataset_config.get('n_input', [64, 64, 64])
            if len(view_dims) > 1 and fea_data.shape[1] == sum(view_dims):
                print(f"按配置分割为 {len(view_dims)} 个视图")
                start_idx = 0
                for j, dim in enumerate(view_dims):
                    end_idx = start_idx + dim
                    view_data = fea_data[:, start_idx:end_idx]
                    print(f"视图{j + 1}形状: {view_data.shape}")

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)
                    start_idx = end_idx
            else:
                # 作为单视图处理
                print("作为单视图处理")
                std_view = mm.fit_transform(fea_data)
                X.append(std_view)
                Y.append(Label)

    # 方法2: 查找X1, X2, X3...等视图
    elif not use_h5py:
        view_data_found = False
        for i in range(1, 10):  # 假设最多9个视图
            view_key = f'X{i}'
            if view_key in data:
                view_data = data[view_key]
                if isinstance(view_data, np.ndarray) and view_data.dtype == object:
                    # 处理cell数组
                    for j in range(view_data.shape[1]):
                        cell_item = view_data[0, j]
                        if isinstance(cell_item, np.ndarray):
                            view_data_item = cell_item.astype(np.float32)
                            if view_data_item.shape[0] != len(Label):
                                view_data_item = view_data_item.T
                            print(f"视图{i}.{j + 1}形状: {view_data_item.shape}")

                            std_view = mm.fit_transform(view_data_item)
                            X.append(std_view)
                            Y.append(Label)
                            view_data_found = True
                elif isinstance(view_data, np.ndarray):
                    view_data = view_data.astype(np.float32)
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T
                    print(f"找到视图{i}: {view_key} 形状: {view_data.shape}")

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)
                    view_data_found = True

        # 方法3: 尝试从X cell数组中提取
        if not view_data_found and 'X' in data and isinstance(data['X'], np.ndarray):
            print(f"X的数据类型: {data['X'].dtype}")
            print(f"X的形状: {data['X'].shape}")

            if data['X'].dtype == object:  # cell数组
                print("从X cell数组中提取视图...")
                X_cell = data['X']
                for i in range(X_cell.shape[1]):
                    cell_item = X_cell[0, i]
                    if isinstance(cell_item, np.ndarray):
                        view_data = cell_item.astype(np.float32)
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                        print(f"cell视图{i + 1}形状: {view_data.shape}")

                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
            else:
                # X不是cell数组，可能是其他格式
                print("X不是cell数组，尝试其他处理方式...")
                X_data = data['X'].astype(np.float32)
                print(f"X_data形状: {X_data.shape}")

                # 如果X是2D数组，尝试根据配置分割
                if X_data.ndim == 2:
                    view_dims = dataset_config.get('n_input', [64, 64, 64])
                    total_dims = sum(view_dims)

                    if X_data.shape[1] == total_dims:
                        print(f"按配置分割视图: {view_dims}")
                        start_idx = 0
                        for dim in view_dims:
                            end_idx = start_idx + dim
                            view_data = X_data[:, start_idx:end_idx]
                            print(f"分割视图形状: {view_data.shape}")

                            std_view = mm.fit_transform(view_data)
                            X.append(std_view)
                            Y.append(Label)
                            start_idx = end_idx

    # 方法4: h5py格式的特殊处理
    elif use_h5py and 'X' in data:
        print("h5py格式的X数据处理")
        X_data = data['X']
        print(f"X_data形状: {X_data.shape}")

        # 根据h5py结构处理
        for i in range(X_data.shape[1]):
            try:
                cell_ref = X_data[0, i]
                view_data = np.array(data[cell_ref], dtype=np.float32).T
                print(f"视图{i + 1}形状: {view_data.shape}")

                std_view = mm.fit_transform(view_data)
                X.append(std_view)
                Y.append(Label)
            except Exception as e:
                print(f"处理视图{i + 1}时出错: {e}")
                continue

    # 方法5: 尝试其他可能的视图键名
    if len(X) == 0 and not use_h5py:
        print("尝试其他可能的视图键名...")
        possible_view_keys = []
        for key in data.keys():
            if not key.startswith('__') and key not in ['gt', 'Y', 'y', 'label', 'gnd']:
                possible_view_keys.append(key)

        print(f"可能的视图键: {possible_view_keys}")

        # 尝试按字母顺序或数字顺序排序
        possible_view_keys.sort()
        for key in possible_view_keys:
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"处理键 '{key}': 形状{val.shape}, 类型{val.dtype}")

                if val.dtype == object:
                    # 处理cell数组
                    print(f"  '{key}'是cell数组")
                    for i in range(val.shape[1]):
                        cell_item = val[0, i]
                        if isinstance(cell_item, np.ndarray):
                            view_data = cell_item.astype(np.float32)
                            if view_data.shape[0] != len(Label):
                                view_data = view_data.T
                            print(f"  子视图{i + 1}形状: {view_data.shape}")

                            std_view = mm.fit_transform(view_data)
                            X.append(std_view)
                            Y.append(Label)
                elif val.ndim == 2:
                    # 处理普通2D数组
                    view_data = val.astype(np.float32)
                    if view_data.shape[0] == len(Label) or view_data.shape[1] == len(Label):
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                        print(f"使用键 '{key}': 形状{view_data.shape}")

                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)

    if len(X) == 0:
        print("数据文件详细内容:")
        if not use_h5py:
            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        print(f"  {key}: 形状{val.shape}, 类型{val.dtype}")
                    else:
                        print(f"  {key}: {type(val)}")
        raise ValueError("无法提取视图数据。请检查.mat文件结构")

    print(f"成功加载 {len(X)} 个视图")
    print(f"每个视图特征维度: {[x.shape[1] for x in X]}")

    # 验证视图数据
    for i, view in enumerate(X):
        if view.shape[0] != len(Label):
            print(f"警告: 视图{i + 1}样本数({view.shape[0]})与标签数({len(Label)})不匹配!")
            # 尝试修复
            if view.shape[1] == len(Label):
                X[i] = view.T
                print(f"已转置视图{i + 1}")

    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(
        missrate=missrate, X=X, Y=Y
    )

    return X, Y, missindex, X_com, Y_com, index_com, index_incom'''


#可以跑COIL20 ORL
'''def load_data(dataset_config, missrate):
    """加载数据"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集名称
    dataset_name = dataset_config[1]

    # 可能的文件路径列表
    possible_paths = [
        os.path.join(current_dir, 'Dataset', dataset_name + ".mat"),
        os.path.join(current_dir, 'Dataset', dataset_name + "_v73.mat"),
        os.path.join(current_dir, dataset_name + ".mat"),
        os.path.join('./Dataset/', dataset_name + ".mat"),
        os.path.join('./data/', dataset_name + ".mat"),
    ]

    # 查找存在的文件
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            print(f"找到数据文件: {file_path}")
            break

    if file_path is None:
        print("尝试的文件路径:")
        for p in possible_paths:
            print(f"  {p}")
        raise FileNotFoundError(f"找不到数据文件: {dataset_name}")

    # 读取数据
    try:
        # 先尝试h5py
        data = h5py.File(file_path, 'r')
        print("使用h5py读取成功")
        use_h5py = True
    except:
        try:
            # 再尝试scipy
            data = scipy.io.loadmat(file_path)
            print("使用scipy.io.loadmat读取成功")
            use_h5py = False
        except Exception as e:
            print(f"两种方式都失败: {e}")
            raise

    # 处理数据...
    X = []
    Y = []

    # 打印所有键名以便调试
    if not use_h5py:
        print("数据文件中的键:", [key for key in data.keys() if not key.startswith('__')])

    # 处理标签
    Label = None
    label_keys = ['Y', 'y', 'label', 'gnd', 'L']
    for key in label_keys:
        if key in data:
            Label = data[key]
            print(f"使用键 '{key}' 作为标签")
            break

    if Label is None:
        raise KeyError("找不到标签数据")

    Label = np.array(Label)
    print(f"原始标签形状: {Label.shape}")
    Label = Label.flatten()
    print(f"处理后标签形状: {Label.shape}")
    print(f"标签样本数: {len(Label)}")
    print(f"标签唯一值: {np.unique(Label)}")

    mm = MinMaxScaler()

    # 方法1: 处理X cell数组（对于COIL20数据集）
    if 'X' in data and isinstance(data['X'], np.ndarray):
        print(f"X的数据类型: {data['X'].dtype}")
        print(f"X的形状: {data['X'].shape}")

        if data['X'].dtype == object:  # cell数组
            print("从X cell数组中提取视图...")
            X_cell = data['X']

            for i in range(X_cell.shape[1]):
                cell_item = X_cell[0, i]
                if isinstance(cell_item, np.ndarray):
                    view_data = cell_item.astype(np.float32)

                    # 调试信息
                    print(f"cell视图{i + 1}原始形状: {view_data.shape}")
                    print(f"cell视图{i + 1}维度: {view_data.ndim}")

                    # 处理3D数据（图像数据）
                    if view_data.ndim == 3:
                        print(f"视图{i + 1}是3D图像数据，需要展平")
                        # 展平图像：将(1440, 32, 32)转换为(1440, 1024)
                        n_samples = view_data.shape[0]
                        # 检查是否为图像数据
                        if view_data.shape[1] == view_data.shape[2]:  # 方形图像
                            image_size = view_data.shape[1]
                            view_data_flat = view_data.reshape(n_samples, -1)
                            print(f"展平后形状: {view_data_flat.shape}")
                            print(f"每个样本特征数: {view_data_flat.shape[1]}")

                            # 确保特征数与配置匹配
                            if 'n_input' in dataset_config:
                                expected_dim = dataset_config['n_input'][i] if i < len(
                                    dataset_config['n_input']) else None
                                if expected_dim and view_data_flat.shape[1] != expected_dim:
                                    print(
                                        f"警告: 视图{i + 1}特征数({view_data_flat.shape[1]})与配置({expected_dim})不匹配")
                                    # 可以选择调整或跳过
                                    if view_data_flat.shape[1] > expected_dim:
                                        print(f"使用PCA降维到{expected_dim}维")
                                        from sklearn.decomposition import PCA
                                        pca = PCA(n_components=expected_dim)
                                        view_data_flat = pca.fit_transform(view_data_flat)
                                    elif view_data_flat.shape[1] < expected_dim:
                                        print(f"使用零填充到{expected_dim}维")
                                        padding = np.zeros((n_samples, expected_dim - view_data_flat.shape[1]))
                                        view_data_flat = np.hstack([view_data_flat, padding])
                        else:
                            # 非方形图像，直接展平
                            view_data_flat = view_data.reshape(n_samples, -1)
                            print(f"非方形图像展平后形状: {view_data_flat.shape}")

                        view_data = view_data_flat
                    elif view_data.ndim == 2:
                        # 已经是2D数据
                        print(f"视图{i + 1}是2D数据")
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                    else:
                        raise ValueError(f"视图{i + 1}维度({view_data.ndim})不支持")

                    print(f"cell视图{i + 1}处理后形状: {view_data.shape}")

                    # 数据标准化
                    try:
                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                        print(f"视图{i + 1}标准化后形状: {std_view.shape}")
                    except Exception as e:
                        print(f"视图{i + 1}标准化失败: {e}")
                        # 尝试其他标准化方法
                        print("尝试使用StandardScaler...")
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        std_view = scaler.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                        print(f"视图{i + 1}标准化后形状: {std_view.shape}")

    # 方法2: 尝试其他可能的视图键名
    if len(X) == 0 and not use_h5py:
        print("尝试其他可能的视图键名...")
        possible_view_keys = []
        for key in data.keys():
            if not key.startswith('__') and key not in ['Y', 'y', 'label', 'gnd', 'L']:
                possible_view_keys.append(key)

        print(f"可能的视图键: {possible_view_keys}")

        # 尝试按字母顺序或数字顺序排序
        possible_view_keys.sort()
        for key in possible_view_keys:
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"处理键 '{key}': 形状{val.shape}, 类型{val.dtype}, 维度{val.ndim}")

                if val.ndim == 3:
                    print(f"  '{key}'是3D数据，需要展平")
                    # 展平3D数据
                    n_samples = val.shape[0]
                    val_flat = val.reshape(n_samples, -1)
                    print(f"  展平后形状: {val_flat.shape}")

                    view_data = val_flat.astype(np.float32)
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)
                elif val.ndim == 2:
                    # 处理普通2D数组
                    view_data = val.astype(np.float32)
                    if view_data.shape[0] == len(Label) or view_data.shape[1] == len(Label):
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                        print(f"使用键 '{key}': 形状{view_data.shape}")

                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                else:
                    print(f"  警告: '{key}'的维度{val.ndim}不支持")

    # 方法3: h5py格式的特殊处理
    elif use_h5py and 'X' in data:
        print("h5py格式的X数据处理")
        X_data = data['X']
        print(f"X_data形状: {X_data.shape}")

        # 根据h5py结构处理
        for i in range(X_data.shape[1]):
            try:
                cell_ref = X_data[0, i]
                view_data = np.array(data[cell_ref], dtype=np.float32)
                print(f"视图{i + 1}原始形状: {view_data.shape}")

                # 处理3D数据
                if view_data.ndim == 3:
                    # 展平图像数据
                    n_samples = view_data.shape[0]
                    view_data = view_data.reshape(n_samples, -1)
                    print(f"视图{i + 1}展平后形状: {view_data.shape}")

                if view_data.shape[0] != len(Label):
                    view_data = view_data.T

                std_view = mm.fit_transform(view_data)
                X.append(std_view)
                Y.append(Label)
                print(f"视图{i + 1}处理后形状: {std_view.shape}")
            except Exception as e:
                print(f"处理视图{i + 1}时出错: {e}")
                continue

    if len(X) == 0:
        print("数据文件详细内容:")
        if not use_h5py:
            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        print(f"  {key}: 形状{val.shape}, 类型{val.dtype}, 维度{val.ndim}")
                    else:
                        print(f"  {key}: {type(val)}")
        raise ValueError("无法提取视图数据。请检查.mat文件结构")

    print(f"成功加载 {len(X)} 个视图")
    print(f"每个视图特征维度: {[x.shape[1] for x in X]}")

    # 验证视图数据
    for i, view in enumerate(X):
        if view.shape[0] != len(Label):
            print(f"警告: 视图{i + 1}样本数({view.shape[0]})与标签数({len(Label)})不匹配!")
            # 尝试修复
            if view.shape[1] == len(Label):
                X[i] = view.T
                print(f"已转置视图{i + 1}")

    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(
        missrate=missrate, X=X, Y=Y
    )

    return X, Y, missindex, X_com, Y_com, index_com, index_incom'''


'''def load_data(dataset_config, missrate):
    """加载数据"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集名称
    dataset_name = dataset_config[1]

    # 可能的文件路径列表
    possible_paths = [
        os.path.join(current_dir, 'Dataset', dataset_name + ".mat"),
        os.path.join(current_dir, 'Dataset', dataset_name + "_v73.mat"),
        os.path.join(current_dir, dataset_name + ".mat"),
        os.path.join('./Dataset/', dataset_name + ".mat"),
        os.path.join('./data/', dataset_name + ".mat"),
    ]

    # 查找存在的文件
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            print(f"找到数据文件: {file_path}")
            break

    if file_path is None:
        print("尝试的文件路径:")
        for p in possible_paths:
            print(f"  {p}")
        raise FileNotFoundError(f"找不到数据文件: {dataset_name}")

    # 读取数据
    try:
        # 先尝试h5py
        data = h5py.File(file_path, 'r')
        print("使用h5py读取成功")
        use_h5py = True
    except:
        try:
            # 再尝试scipy
            data = scipy.io.loadmat(file_path)
            print("使用scipy.io.loadmat读取成功")
            use_h5py = False
        except Exception as e:
            print(f"两种方式都失败: {e}")
            raise

    # 处理数据...
    X = []
    Y = []

    # 打印所有键名以便调试
    if not use_h5py:
        print("数据文件中的键:", [key for key in data.keys() if not key.startswith('__')])

    # 处理标签
    Label = None
    label_keys = ['gt', 'Y', 'y', 'label', 'gnd', 'L']
    for key in label_keys:
        if key in data:
            Label = data[key]
            print(f"使用键 '{key}' 作为标签")
            break

    if Label is None:
        raise KeyError("找不到标签数据")

    Label = np.array(Label)
    print(f"原始标签形状: {Label.shape}")
    Label = Label.flatten()
    print(f"处理后标签形状: {Label.shape}")
    print(f"标签样本数: {len(Label)}")
    print(f"标签唯一值: {np.unique(Label)}")

    mm = MinMaxScaler()

    # MSRC_v1数据集特殊处理
    if dataset_name == 'MSRC_v1' and 'fea' in data:
        print("处理MSRC_v1数据集...")
        fea_data = data['fea']
        print(f"fea数据形状: {fea_data.shape}")
        print(f"fea数据类型: {fea_data.dtype}")

        if fea_data.dtype == object and fea_data.shape == (1, 5):
            print("从fea cell数组中提取5个视图...")
            for i in range(fea_data.shape[1]):
                cell_item = fea_data[0, i]
                if isinstance(cell_item, np.ndarray):
                    view_data = cell_item.astype(np.float32)
                    print(f"视图{i + 1}原始形状: {view_data.shape}")

                    # 确保正确的形状 (n_samples x n_features)
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T
                    print(f"视图{i + 1}处理后形状: {view_data.shape}")

                    try:
                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                        print(f"视图{i + 1}标准化后形状: {std_view.shape}")
                    except Exception as e:
                        print(f"视图{i + 1}标准化失败: {e}")
                        # 直接使用原始数据
                        X.append(view_data)
                        Y.append(Label)
                        print(f"视图{i + 1}使用原始数据")

    # 通用处理：处理X cell数组
    elif 'X' in data and isinstance(data['X'], np.ndarray):
        print(f"X的数据类型: {data['X'].dtype}")
        print(f"X的形状: {data['X'].shape}")

        if data['X'].dtype == object:  # cell数组
            print("从X cell数组中提取视图...")
            X_cell = data['X']

            for i in range(X_cell.shape[1]):
                cell_item = X_cell[0, i]
                if isinstance(cell_item, np.ndarray):
                    view_data = cell_item.astype(np.float32)

                    print(f"cell视图{i + 1}原始形状: {view_data.shape}")

                    # 处理3D数据（图像数据）
                    if view_data.ndim == 3:
                        print(f"视图{i + 1}是3D图像数据，需要展平")
                        n_samples = view_data.shape[0]
                        view_data_flat = view_data.reshape(n_samples, -1)
                        print(f"展平后形状: {view_data_flat.shape}")
                        view_data = view_data_flat
                    elif view_data.ndim == 2:
                        # 已经是2D数据
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                    else:
                        raise ValueError(f"视图{i + 1}维度({view_data.ndim})不支持")

                    print(f"cell视图{i + 1}处理后形状: {view_data.shape}")

                    try:
                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                        print(f"视图{i + 1}标准化后形状: {std_view.shape}")
                    except Exception as e:
                        print(f"视图{i + 1}标准化失败: {e}")
                        # 尝试其他标准化方法
                        print("尝试使用StandardScaler...")
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        std_view = scaler.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                        print(f"视图{i + 1}标准化后形状: {std_view.shape}")

    # 通用处理：尝试其他可能的视图键名
    if len(X) == 0 and not use_h5py:
        print("尝试其他可能的视图键名...")
        possible_view_keys = []
        for key in data.keys():
            if not key.startswith('__') and key not in ['gt', 'Y', 'y', 'label', 'gnd', 'L']:
                possible_view_keys.append(key)

        print(f"可能的视图键: {possible_view_keys}")

        # 尝试按字母顺序或数字顺序排序
        possible_view_keys.sort()
        for key in possible_view_keys:
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"处理键 '{key}': 形状{val.shape}, 类型{val.dtype}, 维度{val.ndim}")

                if val.dtype == object:
                    # 处理cell数组
                    print(f"  '{key}'是cell数组，提取内部数据")
                    if val.shape == (1, 5):  # MSRC_v1结构
                        for i in range(val.shape[1]):
                            cell_item = val[0, i]
                            if isinstance(cell_item, np.ndarray):
                                view_data = cell_item.astype(np.float32)
                                if view_data.shape[0] != len(Label):
                                    view_data = view_data.T
                                print(f"  子视图{i + 1}形状: {view_data.shape}")

                                std_view = mm.fit_transform(view_data)
                                X.append(std_view)
                                Y.append(Label)
                    else:
                        # 通用cell数组处理
                        for i in range(val.shape[1]):
                            cell_item = val[0, i]
                            if isinstance(cell_item, np.ndarray):
                                view_data = cell_item.astype(np.float32)
                                if view_data.shape[0] != len(Label):
                                    view_data = view_data.T
                                print(f"  子视图{i + 1}形状: {view_data.shape}")

                                std_view = mm.fit_transform(view_data)
                                X.append(std_view)
                                Y.append(Label)

                elif val.ndim == 3:
                    print(f"  '{key}'是3D数据，需要展平")
                    n_samples = val.shape[0]
                    val_flat = val.reshape(n_samples, -1)
                    print(f"  展平后形状: {val_flat.shape}")

                    view_data = val_flat.astype(np.float32)
                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)
                elif val.ndim == 2:
                    # 处理普通2D数组
                    view_data = val.astype(np.float32)
                    if view_data.shape[0] == len(Label) or view_data.shape[1] == len(Label):
                        if view_data.shape[0] != len(Label):
                            view_data = view_data.T
                        print(f"使用键 '{key}': 形状{view_data.shape}")

                        std_view = mm.fit_transform(view_data)
                        X.append(std_view)
                        Y.append(Label)
                else:
                    print(f"  警告: '{key}'的维度{val.ndim}不支持")

    # h5py格式的特殊处理
    elif use_h5py:
        print("h5py格式数据处理")
        # 尝试找到视图数据
        view_keys = []
        for key in data.keys():
            if key not in ['gt', 'Y', 'y', 'label', 'gnd', 'L']:
                view_keys.append(key)

        print(f"可能的视图键: {view_keys}")

        for key in view_keys:
            try:
                if isinstance(data[key], h5py.Group):
                    print(f"处理h5py组 '{key}'")
                    # 这里需要根据具体结构处理
                elif isinstance(data[key], h5py.Dataset):
                    view_data = np.array(data[key], dtype=np.float32)
                    print(f"处理h5py数据集 '{key}': 形状{view_data.shape}")

                    # 处理3D数据
                    if view_data.ndim == 3:
                        n_samples = view_data.shape[0]
                        view_data = view_data.reshape(n_samples, -1)
                        print(f"展平后形状: {view_data.shape}")

                    if view_data.shape[0] != len(Label):
                        view_data = view_data.T

                    std_view = mm.fit_transform(view_data)
                    X.append(std_view)
                    Y.append(Label)
                    print(f"处理后形状: {std_view.shape}")
            except Exception as e:
                print(f"处理键 '{key}' 时出错: {e}")
                continue

    if len(X) == 0:
        print("数据文件详细内容:")
        if not use_h5py:
            for key in data.keys():
                if not key.startswith('__'):
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        print(f"  {key}: 形状{val.shape}, 类型{val.dtype}, 维度{val.ndim}")
                        if val.dtype == object and val.shape[0] == 1:
                            print(f"    cell数组内容:")
                            for i in range(val.shape[1]):
                                cell_item = val[0, i]
                                if isinstance(cell_item, np.ndarray):
                                    print(f"      元素{i}: 形状{cell_item.shape}, 类型{cell_item.dtype}")
                    else:
                        print(f"  {key}: {type(val)}")
        raise ValueError("无法提取视图数据。请检查.mat文件结构")

    print(f"成功加载 {len(X)} 个视图")
    print(f"每个视图特征维度: {[x.shape[1] for x in X]}")

    # 验证视图数据
    for i, view in enumerate(X):
        if view.shape[0] != len(Label):
            print(f"警告: 视图{i + 1}样本数({view.shape[0]})与标签数({len(Label)})不匹配!")
            # 尝试修复
            if view.shape[1] == len(Label):
                X[i] = view.T
                print(f"已转置视图{i + 1}")

    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(
        missrate=missrate, X=X, Y=Y
    )

    return X, Y, missindex, X_com, Y_com, index_com, index_incom'''
