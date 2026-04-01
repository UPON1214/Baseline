import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

colorList = [
    'darkblue',
    'darkviolet',
    'red',
    'deeppink',
    'darkslategray',
    'darkgreen',
    'fuchsia',
    'crimson',
    'aqua',
    'steelblue',
    'dodgerblue',
    'royalblue',
    'greenyellow',
    'lightblue',
    'lightgreen',
    'slateblue',
    'orange',
    'orangered',
    'yellow',
    'gold',
    'sandybrown',
    'gray',
]


def generate_random_color():
    """生成一个随机的十六进制颜色代码"""
    return '#{:02x}{:02x}{:02x}'.format(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))


def TSNE_show2D(z, y):
    """use t-SNE to visualize the latent representation"""
    t_sne = TSNE(n_components=2, learning_rate='auto') #初始化t-SNE：使用 TSNE(n_components=2, learning_rate='auto') 创建一个t-SNE对象，设置目标维度为2，学习率自动调整。
    data = t_sne.fit_transform(z) #数据降维：通过 fit_transform(z) 将高维数据 z 降维到二维空间，结果存储在 data 中。
    data = pd.DataFrame(data, index=y) #将降维后的数据 data 转换为 pandas.DataFrame，使用 y 作为索引。这里 y 应该是与 z 中的数据点一一对应的标签或标识符。
   # color = [colorList[i - 1] for i in data.index] # 颜色映射：根据 data 的索引（即 y 的值）来为每个数据点分配颜色。这里假设 colorList 是一个预定义的颜色列表，其长度应至少与 y 中的唯一值数量相同。

    # 使用随机颜色
    unique_labels = data.index.unique()
    color_map = {label: generate_random_color() for label in unique_labels}
    color = [color_map[label] for label in data.index]

    plt.scatter(data[0], data[1], c=color, marker='.', s=12) # 绘制散点图：使用 plt.scatter 绘制二维空间中的点，颜色由 color 数组决定，点的大小设置为12。
    plt.axis('off') #关闭坐标轴，设置x轴和y轴的刻度字体大小为10，调整布局以紧凑显示图表，并保存为PDF文件。文件名包含当前时间戳。
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    #plt.savefig('../imgs/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%d %H-%M-%S')) + '.pdf')
    plt.show()
