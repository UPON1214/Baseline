import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np

from mmi_loss import MMI
import evaluation
from util import next_batch, next_batch_COIL20
from mmd_loss import MMD
from visualization import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, feature Z^v.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, feature Z^v.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction samples.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, feature Z^v.
              x_hat:  [num, feat_dim] float tensor, reconstruction samples.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Apadc():
    """Apadc module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        #if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
           # raise ValueError('Inconsistent latent dim!')

        # 检查所有自动编码器的潜在维度是否一致
        latent_dims = [config['Autoencoder'][f'arch{i}'][-1] for i in range(1, 4)]
        if not all(dim == latent_dims[0] for dim in latent_dims):
            raise ValueError('Inconsistent latent dimensions!')

       # self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._latent_dim = latent_dims[0]

        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder3 = Autoencoder(config['Autoencoder']['arch3'], config['Autoencoder']['activations3'],
                                        config['Autoencoder']['batchnorm'])

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.autoencoder3.to(device)


    '''def train(self, config, logger, x1_train, x2_train, x3_train, Y_list, mask, optimizer, device):
        """Training the model.
            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """

        # Get complete data for training
        #if (torch.LongTensor([1, 1, 1, 1]).to(device) == mask).int():
           # print('yes')
        flag_1 = (torch.LongTensor([1, 1, 1]).to(device) == mask).int()
       # print(flag_1)
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0
        for epoch in range(config['training']['epoch'] + 1):
            #X1, X2, X3, X4 = shuffle(x1_train, x2_train, flag_1[:, 0], flag_1[:, 1])
            #X3, X4 = shuffle(x3_train, x4_train, flag_1[:, 2], flag_1[:, 3])
            X1, X1_index = shuffle(x1_train, flag_1[:, 0])
            X2, X2_index = shuffle(x2_train, flag_1[:, 1])
            X3, X3_index = shuffle(x3_train, flag_1[:, 2])

            loss_all, loss_rec1, loss_rec2, loss_rec3, loss_mmi, loss_mmd = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, x1_index, x2_index, x3_index, batch_No in next_batch_COIL20(X1, X2, X3, X1_index, X2_index, X3_index, config['training']['batch_size']):
                if len(batch_x1) == 1:
                    continue
                index_both = x1_index + x2_index + x3_index  == 3                    # C in indicator matrix A of complete multi-view data
                index_peculiar1 = (x1_index + x1_index + x2_index + x3_index == 3) & (x1_index == 1)     # I^1 in indicator matrix A of incomplete multi-view data
                index_peculiar2 = (x1_index + x2_index + x2_index + x3_index == 3) & (x2_index == 1)    # I^2 in indicator matrix A of incomplete multi-view data
                index_peculiar3 = (x1_index + x2_index + x3_index + x3_index == 3) & (x3_index == 1)


                #print(batch_x1.shape)
                z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])   # [Z_C^1;Z_I^1]
                z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])   # [Z_C^2;Z_I^2]
                z_3 = self.autoencoder3.encoder(batch_x3[x3_index == 1])



                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1[x1_index == 1])
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2[x2_index == 1])
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_3), batch_x3[x3_index == 1])

                rec_loss = (recon1 + recon2 + recon3)
                #rec_loss = (recon1 + recon2)                               # reconstruction losses \sum L_REC^v

                z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
                z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])
                z_view3_both = self.autoencoder3.encoder(batch_x3[index_both])


                if len(batch_x2[index_peculiar2]) % config['training']['batch_size'] == 1: # 首先检查 batch_x2[index_peculiar2] 的长度。如果该长度除以批量大小后余数为1（即表示当前批次中剩余的样本数为1），则跳过本次迭代。这是为了避免在批次中处理非常小的批量，这种情况在训练过程中可能不稳定。
                    continue
                z_view2_peculiar = self.autoencoder2.encoder(batch_x2[index_peculiar2]) # 如果不满足跳过条件，代码会对 batch_x2[index_peculiar2] 进行编码，得到 z_view2_peculiar，这表示 view2 的独特部分的编码表示
                if len(batch_x1[index_peculiar1]) % config['training']['batch_size'] == 1:
                    continue
                z_view1_peculiar = self.autoencoder1.encoder(batch_x1[index_peculiar1])
                if len(batch_x3[index_peculiar3]) % config['training']['batch_size'] == 1:
                    continue
                z_view3_peculiar = self.autoencoder3.encoder(batch_x3[index_peculiar3])


                w1 = torch.var(z_view1_both) # 计算张量元素的方差（variance）的函数 方差表示编码表示的分布离散程度。通过计算方差，模型可以确定各视图在联合表示中应当占据的权重比例
                w2 = torch.var(z_view2_both)
                w3 = torch.var(z_view3_both)

                total_w = w1 + w2 + w3
                a1 = w1 /total_w  #(w1 + w2) # a1 是 z_view1_both 的权重，计算方法是 w1 占 w1 + w2 的比例
                a2 = w2 / total_w
                a3 = w3 / total_w

                #a2 = 1 - a1
                # the weight matrix is only used in MMI loss to explore the common cluster information
                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v)) for MMI loss
                #Z = torch.add(z_view1_both * a1, z_view2_both * a2) # 使用 torch.add 对加权后的两个视图进行加法操作，得到联合表示 Z
                temp_sum = torch.add(z_view1_both * a1, z_view2_both * a2)
                Z = torch.add(temp_sum ,z_view3_both * a3)
                # mutual information losses \sum L_MMI^v (Z_C, Z_I^v)
                mmi_loss = MMI(z_view1_both, Z) + MMI(z_view2_both, Z) + MMI(z_view3_both, Z)

                view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)
                view2 = torch.cat([z_view2_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)
                view3 = torch.cat([z_view3_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)

                #view_all = torch.add(view1, view2, view3, view4).div(4)
                #print(view1.shape)
                #print(view2.shape)
                #print(view3.shape)
                #print(view4.shape)
                total_sum = torch.add(view1, view2)
                total_sum = torch.add(total_sum, view3)

                view_all = total_sum.div(4)

                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv for MMD loss
                #view_both = torch.add(view1, view2).div(2)
                # mean discrepancy losses   \sum L_MMD^v (Z_C, Z_I^v)
                mmd_loss = MMD(view1, view_all, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num']) + \
                           MMD(view2, view_all, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num']) + \
                           MMD(view3, view_all, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num'])


                # total loss
                loss = mmi_loss + mmd_loss * config['training']['lambda1'] + rec_loss * config['training']['lambda2']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_rec3 += recon3.item()

                loss_mmd += mmd_loss.item()
                loss_mmi += mmi_loss.item()

            if (epoch) % config['print_num'] == 0:
                # output = "Epoch: {:.0f}/{:.0f} " \
                #          "==> REC loss = {:.4f} " \
                #          "==> MMD loss = {:.4f} " \
                #          "==> MMI loss = {:.4e} " \
                #     .format(epoch, config['training']['epoch'], (loss_rec1 + loss_rec2), loss_mmd, loss_mmi)
                output = "Epoch: {:.0f}/{:.0f} " \
                    .format(epoch, config['training']['epoch'])
                print(output)
                # evalution
                scores = self.evaluation(config, logger, mask, x1_train, x2_train, x3_train, Y_list, device)


                if scores['kmeans']['ACC'] >= Tmp_acc:
                    Tmp_acc = scores['kmeans']['ACC']
                    Tmp_nmi = scores['kmeans']['NMI']
                    Tmp_ari = scores['kmeans']['ARI']
        return Tmp_acc, Tmp_nmi, Tmp_ari'''

    def train(self, config, logger, x1_train, x2_train, x3_train, Y_list, mask, optimizer, device):
        """Training the model."""

        flag_1 = (torch.LongTensor([1, 1, 1]).to(device) == mask).int()
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)

        Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0

        # ===== 新增：loss 收敛记录 =====
        loss_curve = {
            'total': [],
            'rec': [],
            'mmi': [],
            'mmd': []
        }

        for epoch in range(config['training']['epoch'] + 1):

            X1, X1_index = shuffle(x1_train, flag_1[:, 0])
            X2, X2_index = shuffle(x2_train, flag_1[:, 1])
            X3, X3_index = shuffle(x3_train, flag_1[:, 2])

            loss_all = 0.0
            loss_rec = 0.0
            loss_mmi = 0.0
            loss_mmd = 0.0
            batch_count = 0

            for batch_x1, batch_x2, batch_x3, x1_index, x2_index, x3_index, batch_No in \
                    next_batch_COIL20(X1, X2, X3, X1_index, X2_index, X3_index,
                                      config['training']['batch_size']):

                if len(batch_x1) <= 1:
                    continue

                index_both = (x1_index + x2_index + x3_index) == 3
                index_peculiar1 = (x1_index + x1_index + x2_index + x3_index == 3) & (x1_index == 1)
                index_peculiar2 = (x1_index + x2_index + x2_index + x3_index == 3) & (x2_index == 1)
                index_peculiar3 = (x1_index + x2_index + x3_index + x3_index == 3) & (x3_index == 1)

                z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])
                z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])
                z_3 = self.autoencoder3.encoder(batch_x3[x3_index == 1])

                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1[x1_index == 1])
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2[x2_index == 1])
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_3), batch_x3[x3_index == 1])
                rec_loss = recon1 + recon2 + recon3

                z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
                z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])
                z_view3_both = self.autoencoder3.encoder(batch_x3[index_both])

                if len(batch_x1[index_peculiar1]) <= 1 or \
                        len(batch_x2[index_peculiar2]) <= 1 or \
                        len(batch_x3[index_peculiar3]) <= 1:
                    continue

                z_view1_peculiar = self.autoencoder1.encoder(batch_x1[index_peculiar1])
                z_view2_peculiar = self.autoencoder2.encoder(batch_x2[index_peculiar2])
                z_view3_peculiar = self.autoencoder3.encoder(batch_x3[index_peculiar3])

                w1, w2, w3 = torch.var(z_view1_both), torch.var(z_view2_both), torch.var(z_view3_both)
                total_w = w1 + w2 + w3
                a1, a2, a3 = w1 / total_w, w2 / total_w, w3 / total_w

                Z = z_view1_both * a1 + z_view2_both * a2 + z_view3_both * a3
                mmi_loss = MMI(z_view1_both, Z) + MMI(z_view2_both, Z) + MMI(z_view3_both, Z)

                view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)
                view2 = torch.cat([z_view2_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)
                view3 = torch.cat([z_view3_both, z_view1_peculiar, z_view2_peculiar, z_view3_peculiar], dim=0)

                view_all = (view1 + view2 + view3) / 4.0

                mmd_loss = (
                        MMD(view1, view_all, config['training']['kernel_mul'], config['training']['kernel_num']) +
                        MMD(view2, view_all, config['training']['kernel_mul'], config['training']['kernel_num']) +
                        MMD(view3, view_all, config['training']['kernel_mul'], config['training']['kernel_num'])
                )

                loss = (
                        mmi_loss +
                        config['training']['lambda1'] * mmd_loss +
                        config['training']['lambda2'] * rec_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec += rec_loss.item()
                loss_mmi += mmi_loss.item()
                loss_mmd += mmd_loss.item()
                batch_count += 1

            # ===== 记录 epoch 平均 loss =====
            if batch_count > 0:
                loss_curve['total'].append(loss_all / batch_count)
                loss_curve['rec'].append(loss_rec / batch_count)
                loss_curve['mmi'].append(loss_mmi / batch_count)
                loss_curve['mmd'].append(loss_mmd / batch_count)

            if epoch % config['print_num'] == 0:
                print(f"Epoch [{epoch}/{config['training']['epoch']}]")
                scores = self.evaluation(config, logger, mask,
                                         x1_train, x2_train, x3_train,
                                         Y_list, device)

                if scores['kmeans']['ACC'] >= Tmp_acc:
                    Tmp_acc = scores['kmeans']['ACC']
                    Tmp_nmi = scores['kmeans']['NMI']
                    Tmp_ari = scores['kmeans']['ARI']

        # ===== 准备总体训练损失数据 =====
        losses = loss_curve['total']

        # ===== 绘制总体训练损失函数 =====
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制损失曲线
        ax.plot(
            range(len(losses)),
            losses,
            color='blue',
            linewidth=2,
            label='Total Loss'
        )

        # 设置坐标轴标签及字体
        ax.set_xlabel('Epochs', fontsize=30, weight='bold')  # X轴标签放大加粗
        ax.set_ylabel('Loss', fontsize=30, weight='bold')  # Y轴标签放大加粗

        # 设置刻度字体
        ax.tick_params(axis='both', which='major',
                       labelsize=28, width=3, length=12)
        ax.tick_params(axis='both', which='minor',
                       labelsize=24, width=2.5, length=8)

        # 设置外框线宽度
        ax.spines['top'].set_linewidth(3.5)
        ax.spines['right'].set_linewidth(3.5)
        ax.spines['left'].set_linewidth(3.5)
        ax.spines['bottom'].set_linewidth(3.5)

        plt.tight_layout()

        # ===== 保存图片（高分辨率）=====
        save_dir = "loss_plots"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        save_path = f"{save_dir}/total_loss_{timestamp}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 同时保存最新版本
        latest_path = f"{save_dir}/total_loss_latest.png"
        fig.savefig(latest_path, dpi=300, bbox_inches='tight')

        plt.show()

        return Tmp_acc, Tmp_nmi, Tmp_ari

    def evaluation(self, config, logger, mask, x1_train, x2_train, x3_train, Y_list, device):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
            
            #flag = mask[:, 0] + mask[:, 1] == 2           # complete multi-view data
            flag = (mask.sum(dim=1) == 3)
            view2_missing_idx_eval = mask[:, 0] == 0      # incomplete multi-view data
            view1_missing_idx_eval = mask[:, 1] == 0      # incomplete multi-view data
            view3_missing_idx_eval = mask[:, 2] == 0


            # 处理完全多视图数据
            if flag.any():  # 如果有完全多视图数据
                common_view1 = x1_train[flag]
                common_view1 = self.autoencoder1.encoder(common_view1)
                common_view2 = x2_train[flag]
                common_view2 = self.autoencoder2.encoder(common_view2)
                common_view3 = x3_train[flag]
                common_view3 = self.autoencoder3.encoder(common_view3.to(device))

                y_common = Y_list[flag]

            view1_exist = x1_train[view1_missing_idx_eval]
            view1_exist = self.autoencoder1.encoder(view1_exist)
            y_view1_exist = Y_list[view1_missing_idx_eval]

            view2_exist = x2_train[view2_missing_idx_eval]
            view2_exist = self.autoencoder2.encoder(view2_exist)
            y_view2_exist = Y_list[view2_missing_idx_eval]

            view3_exist = x3_train[view3_missing_idx_eval]
            view3_exist = self.autoencoder3.encoder(view3_exist.to(device))
            y_view3_exist = Y_list[view3_missing_idx_eval]



            # Since the distributions of different views have been aligned,
            # it is OK to obtain the common features for clustering by the following two approaches

            # (1) z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v))
            # w1 = torch.var(common_view1)
            # w2 = torch.var(common_view2)
            # a1 = w1 / (w1 + w2)
            # a2 = 1 - a1
            # common = torch.add(common_view1 * a1, common_view2 * a2)

            # (2) z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv
            add_common = torch.add(common_view1, common_view2)
            common = torch.add(add_common, common_view3).div(4)




            #latent_fusion = torch.cat([common, view1_exist, view2_exist], dim=0).cpu().detach().numpy()
            # 融合所有视图的特征（包括完全和不完全的数据）
            latent_fusion = []
            if common is not None:
                latent_fusion.append(common.cpu().detach().numpy())
            latent_fusion.append(view1_exist.cpu().detach().numpy())
            latent_fusion.append(view2_exist.cpu().detach().numpy())
            latent_fusion.append(view3_exist.cpu().detach().numpy())

            latent_fusion = np.concatenate(latent_fusion, axis=0)

            #Y_list = torch.cat([y_common, y_view1_exist, y_view2_exist], dim=0).cpu().detach().numpy()
            Y_list_all = []
            if common is not None:
                Y_list_all.append(y_common.cpu().detach().numpy())
            Y_list_all.append(y_view1_exist.cpu().detach().numpy())
            Y_list_all.append(y_view2_exist.cpu().detach().numpy())
            Y_list_all.append(y_view3_exist.cpu().detach().numpy())

            Y_list = np.concatenate(Y_list_all, axis=0)

            #TSNE_show2D(latent_fusion, Y_list[:, 0])

            scores, _ = evaluation.clustering([latent_fusion], Y_list[:, 0])
            # print("\033[2;29m" + 'Common features ' + str(scores) + "\033[0m")
            self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
            
        return scores

