import sys
import torch


def compute_joint(view1, view2):
    """Compute the joint probability matrix P计算两个视图的联合概率矩阵 """

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k) # 断言 view2 的尺寸和 view1 一致，确保两个视图的样本数量和特征数量相同。

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1) # 在 view1 的第二个维度上增加一个维度，在 view2 的第一个维度上增加一个维度。计算 view1 和 view2 的外积，结果是一个三维张量。
    p_i_j = p_i_j.sum(dim=0) # 对第一个维度进行求和，得到联合概率矩阵
    p_i_j = (p_i_j + p_i_j.t()) / 2.   # symmetrise 计算对称化后的联合概率矩阵
    p_i_j = p_i_j / p_i_j.sum()        # normalise 对矩阵进行归一化，使得矩阵中所有元素的和为 1

    return p_i_j # 返回计算得到的联合概率矩阵


def MMI(view1, view2, lamb=10.0, EPS=sys.float_info.epsilon):
    """MMI loss用于计算最大互信息（MMI）损失"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k)) # 断言 p_i_j 的尺寸是 (k, k)，确保联合概率矩阵的尺寸正确。

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone() # 计算边缘概率 p_i，并扩展成 k x k 的矩阵。
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone() # 计算边缘概率 p_j，并扩展成 k x k 的矩阵。

    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb) * torch.log(p_j) \
                      - (lamb) * torch.log(p_i))

    loss = loss.sum()

    return loss
