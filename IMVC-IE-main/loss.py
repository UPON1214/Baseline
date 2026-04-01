import torch
import torch.nn as nn
import math


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N): # 旨在生成一个掩码（mask）张量，这个掩码张量是一个布尔类型的矩阵，用于标记哪些样本是相互关联的。具体来说，该方法生成一个 N x N 的矩阵，其中矩阵中的某些元素被设为 False（表示相关样本），其他元素设为 True（表示非相关样本）。
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0) # 使用 fill_diagonal_ 方法将 mask 矩阵的对角线元素填充为 0。_ 后缀表示该方法是就地操作，会直接修改 mask 张量，而不会创建新的张量。
        for i in range(N//2): # N 的一半（向下取整）; 在每次循环中，设置两个特定位置的元素为 0。
            mask[i, N//2 + i] = 0 # 将第 i 行第 N//2 + i 列的元素设为 0。
            mask[N//2 + i, i] = 0 # 这两行代码确保矩阵的对称位置也设为 0
        mask = mask.bool() # 将 mask 转换为布尔类型，即将所有非零元素变为 True，零元素变为 False。
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)                           

        # sim = torch.matmul(h, h.T) / self.temperature_f          
        sim = self.similarity(h.unsqueeze(1), h.unsqueeze(0)) / self.temperature_f # 计算样本之间的相似性。 h.unsqueeze(1) 和 h.unsqueeze(0) 是为了扩展张量的维度，以便相似性计算。
        sim_i_j = torch.diag(sim, self.batch_size)                 
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)      
        mask = self.mask_correlated_samples(N)          # 调用 mask_correlated_samples 方法，生成一个掩码矩阵 mask。mask 是一个布尔矩阵，用于标记哪些样本是负样本
        negative_samples = sim[mask].reshape(N, -1)     # 使用掩码 mask 从相似性矩阵 sim 中提取负样本对的相似性分数。

        labels = torch.zeros(N).to(positive_samples.device).long() # 创建一个全零的标签张量 labels，形状为 (N,)。labels 的类型为 long，并将其移动到 positive_samples 所在的设备（如 GPU）。
        logits = torch.cat((positive_samples, negative_samples), dim=1) # 将 positive_samples 和 negative_samples 沿第 1 维拼接，形成一个包含正负样本对的相似性分数的张量 logits。logits 的形状为 (N, 1 + negative_samples.size(1))
        loss = self.criterion(logits, labels)                                      
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)  # view(-1) 将 p_i 转换为一维张量。
        p_i /= p_i.sum() # 归一化处理，使其和为 1。
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum() # 计算 p_i 的熵，先求出 p_i 的元素个数的对数，再加上 p_i 的每个元素乘以其对数的和
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()                        
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)     

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l     
        sim_i_j = torch.diag(sim, self.class_num)                                      
        sim_j_i = torch.diag(sim, -self.class_num)                                     

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)         
        mask = self.mask_correlated_samples(N)                                         
        negative_clusters = sim[mask].reshape(N, -1)                                   

        labels = torch.zeros(N).to(positive_clusters.device).long()                   
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)              
        loss = self.criterion(logits, labels)                                          
        loss /= N
        return loss + entropy
