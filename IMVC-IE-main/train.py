'''import torch
from network import Network
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy import stats
import h5py
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST_USPS
# BDGP
# my_UCI

#Dataname = 'MNIST_USPS'
Dataname = 'COIL20'
# Dataname = 'my_UCI'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.1)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=50)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)

dataset_pretrain = copy.deepcopy(dataset)
dataset_contrain = copy.deepcopy(dataset)


tot_sample = dataset.V1.shape[0]
p = 0.5
np.random.seed(10)

miss_mark1 = np.random.choice(tot_sample, size=int(tot_sample * p), replace=False)
miss_mark1.sort()
available_mark1 = []
for i in range(tot_sample):
    if i not in miss_mark1:
        available_mark1.append(i)
available_mark1 = np.array(available_mark1)

miss_mark2 = np.random.choice(tot_sample, size=int(tot_sample * p), replace=False)
miss_mark2.sort()
available_mark2 = []
for i in range(tot_sample):
    if i not in miss_mark2:
        available_mark2.append(i)
available_mark2 = np.array(available_mark2)

pair_mark = []
for i in available_mark1:
    if i in available_mark2:
        pair_mark.append(i)
pair_mark = np.array(pair_mark)
print("pair_mark: ",len(pair_mark))

dataset_pretrain.percentage_dele(1, available_mark1, available_mark2, pair_mark)
dataset_contrain.percentage_dele(2, available_mark1, available_mark2, pair_mark)


data_loader_pretrain = torch.utils.data.DataLoader(
    dataset_pretrain,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
data_loader_contrain = torch.utils.data.DataLoader(
    dataset_contrain,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True




pretrain_losses = []  # <<< 用于绘图

def pretrain(epoch):
    tot_loss = 0.
    pretrain_criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_pretrain):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(pretrain_criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    epoch_loss = tot_loss / len(data_loader_pretrain)
    pretrain_losses.append(epoch_loss)  # <<< 保存
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(epoch_loss))




contrastive_losses = []  # <<< 用于绘图

def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_contrain):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    epoch_loss = tot_loss / len(data_loader_contrain)
    contrastive_losses.append(epoch_loss)  # <<< 保存
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(epoch_loss))

def plot_loss_curve(losses, save_name):
    import time
    import os
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(losses)), losses, color='blue', linewidth=2, label='Loss')
    ax.set_xlabel('Epochs', fontsize=30, weight='bold')
    ax.set_ylabel('Loss', fontsize=30, weight='bold')

    ax.tick_params(axis='both', which='major', labelsize=28, width=3, length=12)
    ax.tick_params(axis='both', which='minor', labelsize=24, width=2.5, length=8)

    ax.spines['top'].set_linewidth(3.5)
    ax.spines['right'].set_linewidth(3.5)
    ax.spines['left'].set_linewidth(3.5)
    ax.spines['bottom'].set_linewidth(3.5)

    plt.tight_layout()

    save_dir = "loss_plots"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{save_name}_{timestamp}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    latest_path = os.path.join(save_dir, f"{save_name}_latest.png")
    fig.savefig(latest_path, dpi=300, bbox_inches='tight')

    fig.canvas.draw()
    plt.show(block=True)



def generate_prompt_box(view_mean, mu, sigma, number):
    view_mean = view_mean.reshape(1, -1)
    vm_nor = (view_mean - np.min(view_mean)) / (np.max(view_mean) - np.min(view_mean))                      
    simulated_box = []
    for i in range(number):
        noise = np.random.normal(mu, sigma, view_mean.shape)
        simulated_sample = vm_nor + noise
        simulated_sample = simulated_sample * (np.max(view_mean) - np.min(view_mean)) + np.min(view_mean)   
        simulated_sample = simulated_sample.reshape(view_mean.shape[1])
        simulated_box.append(simulated_sample)
    simulated_box = np.array(simulated_box).astype(np.float32)
    simulated_box = torch.from_numpy(simulated_box)
    return simulated_box


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)



if not os.path.exists('./models'):
    os.makedirs('./models')
seed = 10
T = 1
for i in range(T):
    print(Dataname)
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)


    # ===== 训练循环 =====
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1

    #plot_loss_curve(pretrain_losses, 'AE Pretraining Loss Convergence', 'pretrain_loss')

    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        epoch += 1

    #plot_loss_curve(contrastive_losses, 'Contrastive Loss Convergence', 'contrastive_loss')
    # 绘制对比学习损失，不显示标题
    plot_loss_curve(contrastive_losses, save_name='contrastive_loss')


view1_mean, view2_mean = dataset_pretrain.sample_mean()
r = 1
mu = 0
tot_sample1_pretrain = len(available_mark1)
tot_sample2_pretrain = len(available_mark2)
sigma1, sigma2 = dataset_pretrain.pretrain_sigma()
sigma1 = sigma1.reshape(1, -1)
sigma2 = sigma2.reshape(1, -1)
dataset_rec = copy.deepcopy(dataset)
dataset_rec_noise = copy.deepcopy(dataset)
dataset_rec_NOE = copy.deepcopy(dataset)
dataset_rec_mean = copy.deepcopy(dataset)
dataset_rec_zero = copy.deepcopy(dataset)
count = 0
for i in miss_mark1:
    view2_correspond = (torch.from_numpy(dataset.V2[i]).reshape(1, -1)).to(device)                              
    prompt_box = generate_prompt_box(view1_mean, mu, sigma1, number=int(r * tot_sample1_pretrain)).to(device)   
    vp = [prompt_box, view2_correspond]   
    hs, _, _, _ = model(vp)  
       
    similarity = torch.nn.CosineSimilarity(dim=1)
    sim = similarity(hs[1], hs[0])
    sim_value, sim_mark = torch.topk(sim, k=1)      
    best_simulated_cpu = prompt_box[sim_mark].cpu().numpy().reshape(view1_mean.shape).astype(np.float32)       
    dataset_rec.V1[i] = best_simulated_cpu

print()
for i in miss_mark2:
    view1_correspond = (torch.from_numpy(dataset.V1[i]).reshape(1, -1)).to(device)
    prompt_box = generate_prompt_box(view2_mean, mu, sigma2, number=int(r * tot_sample1_pretrain)).to(device)
    vp = [view1_correspond, prompt_box]
    hs, _, _, _ = model(vp)

    similarity = torch.nn.CosineSimilarity(dim=1)
    sim = similarity(hs[0], hs[1])
    sim_value, sim_mark = torch.topk(sim, k=1)
    best_simulated_cpu = prompt_box[sim_mark].cpu().numpy().reshape(view2_mean.shape).astype(np.float32)
    dataset_rec.V2[i] = best_simulated_cpu


data_loader_st = torch.utils.data.DataLoader(
     dataset_rec,
     batch_size=args.batch_size,
     shuffle=True,
     drop_last=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def semantic_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader_st):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader_st)))

while epoch <= args.mse_epochs + args.con_epochs + 100:
    semantic_train(epoch)
    epoch += 1

rec_v1 = torch.from_numpy(dataset_rec.V1).reshape(data_size, -1).to(device)
rec_v2 = torch.from_numpy(dataset_rec.V2).reshape(data_size, -1).to(device)
_, qs, _, _ = model([rec_v1, rec_v2])
confi1, y_pred1 = qs[0].topk(k=1, dim=1)
y_pred1 = y_pred1.cpu().numpy()
confi2, y_pred2 = qs[1].topk(k=1, dim=1)
y_pred2 = y_pred2.cpu().numpy()

pur1 = purity_score(dataset.Y.reshape(data_size), y_pred1.reshape(data_size))
pur2 = purity_score(dataset.Y.reshape(data_size), y_pred2.reshape(data_size))
print('semantic_V1_acc: ', cluster_acc(dataset.Y, y_pred1))
print('semantic_V2_acc: ', cluster_acc(dataset.Y, y_pred2))
print("semantic_V1_pur1: ", pur1)
print("semantic_V2_pur2: ", pur2)'''

"The above is the original code, and the following is the code we modified to adapt to other datasets. "

import argparse
import copy
import os
import random
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

from dataloader import MultiViewArrayDataset, load_data
from loss import Loss
from network import Network


Dataname = 'COIL20'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--data_path', default='data/COIL20.mat', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--missing_rate', default=0.5, type=float)
parser.add_argument('--temperature_f', default=0.1, type=float)
parser.add_argument('--temperature_l', default=1.0, type=float)
parser.add_argument('--learning_rate', default=0.0003, type=float)
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--mse_epochs', default=50, type=int)
parser.add_argument('--con_epochs', default=200, type=int)
parser.add_argument('--semantic_epochs', default=100, type=int)
parser.add_argument('--feature_dim', default=512, type=int)
parser.add_argument('--high_feature_dim', default=128, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--prompt_multiplier', default=1.0, type=float)
parser.add_argument('--prompt_pool_size', default=64, type=int)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_missing_indices(total_samples: int, num_views: int, missing_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    missing_marks = []
    available_marks = []
    n_miss = int(total_samples * missing_rate)
    for _ in range(num_views):
        miss = np.sort(rng.choice(total_samples, size=n_miss, replace=False))
        avail = np.setdiff1d(np.arange(total_samples), miss)
        missing_marks.append(miss)
        available_marks.append(avail)
    complete_idx = available_marks[0]
    for avail in available_marks[1:]:
        complete_idx = np.intersect1d(complete_idx, avail)
    return missing_marks, available_marks, complete_idx


def build_subset_datasets(dataset: MultiViewArrayDataset, available_marks: List[np.ndarray], complete_idx: np.ndarray):
    dataset_pretrain = dataset.subset_per_view(available_marks)
    dataset_contrain = dataset.subset_same_indices(complete_idx)
    return dataset_pretrain, dataset_contrain


def make_loader(ds, shuffle=True):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=(len(ds) >= args.batch_size),
    )


def plot_loss_curve(losses, save_name):
    if len(losses) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(losses)), losses, linewidth=2, label='Loss')
    ax.set_xlabel('Epochs', fontsize=30, weight='bold')
    ax.set_ylabel('Loss', fontsize=30, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=28, width=3, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(3.5)
    plt.tight_layout()
    save_dir = 'loss_plots'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{save_name}_latest.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_prompt_box(view_mean, mu, sigma, number):
    view_mean = view_mean.reshape(1, -1)
    vmax = np.max(view_mean)
    vmin = np.min(view_mean)
    if vmax > vmin:
        vm_nor = (view_mean - vmin) / (vmax - vmin)
    else:
        vm_nor = np.zeros_like(view_mean)
    simulated_box = []
    for _ in range(number):
        noise = np.random.normal(mu, sigma, view_mean.shape)
        simulated_sample = vm_nor + noise
        simulated_sample = simulated_sample * max(vmax - vmin, 1e-8) + vmin
        simulated_box.append(simulated_sample.reshape(view_mean.shape[1]))
    simulated_box = np.array(simulated_box).astype(np.float32)
    return torch.from_numpy(simulated_box)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(row, col)) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    y_true = y_true.copy()
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def pretrain(model, optimizer, data_loader_pretrain, view, epoch, pretrain_losses):
    tot_loss = 0.
    pretrain_criterion = torch.nn.MSELoss()
    model.train()
    for xs, _, _ in data_loader_pretrain:
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss = sum(pretrain_criterion(xs[v], xrs[v]) for v in range(view))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    denom = max(len(data_loader_pretrain), 1)
    epoch_loss = tot_loss / denom
    pretrain_losses.append(epoch_loss)
    print(f'Pretrain Epoch {epoch} Loss:{epoch_loss:.6f}')


def contrastive_train(model, optimizer, criterion, data_loader_contrain, view, epoch, contrastive_losses):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    model.train()
    for xs, _, _ in data_loader_contrain:
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        hs, qs, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    denom = max(len(data_loader_contrain), 1)
    epoch_loss = tot_loss / denom
    contrastive_losses.append(epoch_loss)
    print(f'Contrastive Epoch {epoch} Loss:{epoch_loss:.6f}')


def reconstruct_missing_views(model, dataset, dataset_pretrain, missing_marks, view):
    model.eval()
    view_means = dataset_pretrain.sample_mean()
    sigmas = dataset_pretrain.pretrain_sigma()
    sigmas = [np.asarray(s, dtype=np.float32).reshape(1, -1) for s in sigmas]
    dataset_rec = dataset.clone()
    mu = 0
    similarity = torch.nn.CosineSimilarity(dim=1)

    with torch.no_grad():
        for v in range(view):
            prompt_count = max(1, min(args.prompt_pool_size, int(args.prompt_multiplier * len(dataset_pretrain.Vs[v]))))
            print(f'Reconstructing view {v+1} with prompt_count={prompt_count}, missing={len(missing_marks[v])}')
            for idx in missing_marks[v]:
                prompt_box = generate_prompt_box(view_means[v], mu, sigmas[v], number=prompt_count).to(device)
                model_inputs = []
                for t in range(view):
                    if t == v:
                        model_inputs.append(prompt_box)
                    else:
                        corr = torch.from_numpy(dataset.Vs[t][idx]).reshape(1, -1).repeat(prompt_count, 1).to(device)
                        model_inputs.append(corr)
                hs, _, _, _ = model(model_inputs)
                sims = []
                for t in range(view):
                    if t != v:
                        sims.append(similarity(hs[v], hs[t]))
                score = torch.stack(sims, dim=0).mean(dim=0)
                _, best_idx = torch.topk(score, k=1)
                best_simulated_cpu = prompt_box[best_idx].cpu().numpy().reshape(view_means[v].shape).astype(np.float32)
                dataset_rec.Vs[v][idx] = best_simulated_cpu
                setattr(dataset_rec, f'V{v+1}', dataset_rec.Vs[v])
    return dataset_rec


def semantic_train(model, optimizer, criterion, data_loader_st, view, epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    model.train()
    for xs, _, _ in data_loader_st:
        xs = [x.to(device) for x in xs]
        optimizer.zero_grad()
        _, qs, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    denom = max(len(data_loader_st), 1)
    print(f'Semantic Epoch {epoch} Loss:{tot_loss / denom:.6f}')


def evaluate(model, dataset_rec, data_size, view):
    model.eval()
    xs = [torch.from_numpy(v).reshape(data_size, -1).to(device) for v in dataset_rec.Vs]
    with torch.no_grad():
        _, qs, _, _ = model(xs)
    preds = []
    for v in range(view):
        _, pred = qs[v].topk(k=1, dim=1)
        pred = pred.cpu().numpy().reshape(data_size)
        preds.append(pred)
        print(f'View {v+1} acc: {cluster_acc(dataset_rec.Y, pred):.4f}')
        print(f'View {v+1} purity: {purity_score(dataset_rec.Y, pred):.4f}')
    q_mean = torch.stack(qs, dim=0).mean(dim=0)
    fused_pred = torch.argmax(q_mean, dim=1).cpu().numpy().reshape(data_size)
    print(f'Fused acc: {cluster_acc(dataset_rec.Y, fused_pred):.4f}')
    print(f'Fused purity: {purity_score(dataset_rec.Y, fused_pred):.4f}')
    return preds, fused_pred


def main():
    setup_seed(args.seed)
    dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_path)
    print(f'Dataset: {args.dataset}, dims={dims}, view={view}, data_size={data_size}, class_num={class_num}')

    missing_marks, available_marks, complete_idx = build_missing_indices(
        data_size, view, args.missing_rate, args.seed
    )
    print('Complete paired samples:', len(complete_idx))
    dataset_pretrain, dataset_contrain = build_subset_datasets(dataset, available_marks, complete_idx)

    data_loader_pretrain = make_loader(dataset_pretrain, shuffle=True)
    data_loader_contrain = make_loader(dataset_contrain, shuffle=True)

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    pretrain_losses = []
    contrastive_losses = []
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(model, optimizer, data_loader_pretrain, view, epoch, pretrain_losses)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(model, optimizer, criterion, data_loader_contrain, view, epoch, contrastive_losses)
        epoch += 1
    plot_loss_curve(pretrain_losses, save_name='pretrain_loss')
    plot_loss_curve(contrastive_losses, save_name='contrastive_loss')

    dataset_rec = reconstruct_missing_views(model, dataset, dataset_pretrain, missing_marks, view)
    data_loader_st = make_loader(dataset_rec, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for semantic_epoch in range(1, args.semantic_epochs + 1):
        semantic_train(model, optimizer, criterion, data_loader_st, view, semantic_epoch)

    evaluate(model, dataset_rec, data_size, view)


if __name__ == '__main__':
    main()


