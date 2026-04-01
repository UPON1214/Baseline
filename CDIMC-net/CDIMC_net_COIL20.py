# -*- coding: utf-8 -*-
"""
CDIMC-net adaptation for COIL20.

This part of the code was independently modified from the original code
for the COIL20 dataset and was not provided by the original author.

This regenerated version is based on the original handwritten-5view PyTorch
implementation, but modified so it can run on the COIL20 dataset with 3 views:
    n_input = [1024, 1024, 324]
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import StandardScaler, normalize
from torch.nn import Linear
from torch.optim import Adam, SGD

from idecutils import cluster_acc, purity_score


def set_seed(seed: int = 20) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def weighted_mse_loss(recon: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    diff2 = (recon - target) ** 2
    diff2 = diff2.mean(dim=1)
    weight = weight.float()
    denom = torch.clamp(weight.sum(), min=1.0)
    return (diff2 * weight).sum() / denom


def get_knn_graph(data: np.ndarray, k_num: int) -> np.ndarray:
    x_norm = np.sum(np.square(data), axis=1, keepdims=True)
    dists = x_norm - 2 * np.matmul(data, data.T) + x_norm.T
    n = data.shape[0]
    graph = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argsort(dists[i])[: k_num + 1]
        graph[i, idx] = 1.0
    graph = graph - np.diag(np.diag(graph))
    graph = np.maximum(graph, graph.T)
    return graph


def compute_hidden_dims(n_input: int, n_stacks: int) -> List[int]:
    dims = []
    current = n_input
    for _ in range(max(n_stacks - 1, 1)):
        current = max(64, int(round(current * 0.8)))
        dims.append(current)
    dims[-1] = max(dims[-1], min(1500, max(128, n_input)))
    return dims


def load_coil20_views(mat_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    data = scipy.io.loadmat(mat_path)
    if "X" not in data or "Y" not in data:
        raise ValueError(f"{mat_path} must contain keys 'X' and 'Y'.")
    X = data["X"]
    Y = data["Y"].reshape(-1).astype(np.int64)
    views: List[np.ndarray] = []
    for i in range(X.shape[1]):
        arr = np.array(X[0, i])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        views.append(arr.astype(np.float32))
    return views, Y


def generate_missing_mask(n_samples: int, n_views: int, miss_rate: float, seed: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - miss_rate
    WE = (rng.random((n_samples, n_views)) < keep_prob).astype(np.float32)
    for i in np.where(WE.sum(axis=1) == 0)[0]:
        WE[i, rng.integers(0, n_views)] = 1.0
    for v in np.where(WE.sum(axis=0) == 0)[0]:
        WE[rng.integers(0, n_samples), v] = 1.0
    return WE


def preprocess_views(views: Sequence[np.ndarray], WE: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    train_views: List[np.ndarray] = []
    reorder_views: List[np.ndarray] = []
    for v, x in enumerate(views):
        x = np.asarray(x, dtype=np.float32)
        observed = WE[:, v] == 1
        missing = ~observed
        x_reorder = x.copy()
        mean_feat = x_reorder[observed].mean(axis=0)
        x_reorder[missing] = mean_feat
        x_reorder = normalize(x_reorder)
        reorder_views.append(x_reorder.astype(np.float32))

        x_train = x.copy()
        scaler = StandardScaler()
        x_train[observed] = scaler.fit_transform(x_train[observed])
        x_train[missing] = 0.0
        train_views.append(np.nan_to_num(x_train).astype(np.float32))
    return train_views, reorder_views


def reorder_by_initial_clustering(
    train_views: Sequence[np.ndarray],
    reorder_views: Sequence[np.ndarray],
    y: np.ndarray,
    WE: np.ndarray,
    n_clusters: int,
    seed: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.concatenate(reorder_views, axis=1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    init_pred = kmeans.fit_predict(stacked)
    order = np.argsort(init_pred, kind="stable")
    train_views_r = [x[order].copy() for x in train_views]
    reorder_views_r = [x[order].copy() for x in reorder_views]
    y_r = y[order].copy()
    WE_r = WE[order].copy()
    return train_views_r, reorder_views_r, y_r, WE_r


def build_graphs(reorder_views: Sequence[np.ndarray], WE: np.ndarray, knn: int) -> List[np.ndarray]:
    graphs: List[np.ndarray] = []
    n = WE.shape[0]
    for v, x in enumerate(reorder_views):
        observed_idx = np.where(WE[:, v] == 1)[0]
        x_obs = x[observed_idx]
        g_obs = get_knn_graph(x_obs, k_num=min(knn, max(1, len(observed_idx) - 1)))
        g_full = np.zeros((n, n), dtype=np.float32)
        g_full[np.ix_(observed_idx, observed_idx)] = g_obs
        graphs.append(g_full)
    return graphs


class ViewAE(nn.Module):
    def __init__(self, input_dim: int, n_stacks: int, n_z: int):
        super().__init__()
        hidden_dims = compute_hidden_dims(input_dim, n_stacks)
        self.enc1 = Linear(input_dim, hidden_dims[0])
        self.enc2 = Linear(hidden_dims[0], hidden_dims[1])
        self.enc3 = Linear(hidden_dims[1], hidden_dims[2])
        self.z_layer = Linear(hidden_dims[2], n_z)
        self.dec0 = Linear(n_z, n_z)
        self.dec1 = Linear(n_z, hidden_dims[2])
        self.dec2 = Linear(hidden_dims[2], hidden_dims[1])
        self.dec3 = Linear(hidden_dims[1], hidden_dims[0])
        self.x_bar = Linear(hidden_dims[0], input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        h3 = F.relu(self.enc3(h2))
        return self.z_layer(h3)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        r = F.relu(self.dec0(z))
        h1 = F.relu(self.dec1(r))
        h2 = F.relu(self.dec2(h1))
        h3 = F.relu(self.dec3(h2))
        return self.x_bar(h3)


class MultiViewAE(nn.Module):
    def __init__(self, n_stacks: int, n_input: Sequence[int], n_z: int):
        super().__init__()
        self.n_views = len(n_input)
        self.view_aes = nn.ModuleList([ViewAE(int(d), n_stacks, n_z) for d in n_input])

    def forward(self, views: Sequence[torch.Tensor], we: torch.Tensor):
        view_z = [ae.encode(x) for ae, x in zip(self.view_aes, views)]
        weights = we.float()
        denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
        fused = sum(view_z[v] * weights[:, v:v+1] for v in range(self.n_views)) / denom
        recons = [ae.decode(fused) for ae in self.view_aes]
        return recons, fused, view_z


class IDEC(nn.Module):
    def __init__(self, n_stacks: int, n_input: Sequence[int], n_z: int, n_clusters: int, pretrain_path: str):
        super().__init__()
        self.pretrain_path = pretrain_path
        self.ae = MultiViewAE(n_stacks=n_stacks, n_input=n_input, n_z=n_z)
        self.n_clusters = n_clusters

    def pretrain(self, train_data: "TrainData", args: argparse.Namespace, device: torch.device) -> None:
        if args.pretrain_flag == 0 or (not os.path.exists(self.pretrain_path)):
            pretrain_ae(self.ae, train_data, args, device, self.pretrain_path)
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path, map_location=device))
            print(f"Loaded pretrained AE from {self.pretrain_path}")

    @torch.no_grad()
    def update_label(self, views: Sequence[torch.Tensor], we: torch.Tensor, cluster_layer: torch.Tensor):
        _, z, _ = self.ae(views, we)
        x_norm = torch.sum(z ** 2, dim=1, keepdim=True)
        c_norm = torch.sum(cluster_layer ** 2, dim=1, keepdim=True).T
        dists = x_norm - 2 * torch.mm(z, cluster_layer.T) + c_norm
        labels = torch.argmin(dists, dim=1)
        losses = torch.min(dists, dim=1).values
        return labels, losses

    def forward(self, views, we, y_pred, cluster_layer, sample_weight):
        _, z, view_z = self.ae(views, we)
        centers = cluster_layer[y_pred]
        dist2 = torch.sum((z - centers) ** 2, dim=1)
        denom = torch.clamp(sample_weight.sum(), min=1.0)
        kl_loss = (dist2 * sample_weight).sum() / denom
        return z, kl_loss, view_z


@dataclass
class TrainData:
    views_np: List[np.ndarray]
    views_torch: List[torch.Tensor]
    y: np.ndarray
    we_np: np.ndarray
    we_torch: torch.Tensor
    graphs_torch: List[torch.Tensor]


def graph_loss_from_batch(z_batch: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
    deg = torch.diag(graph_batch.sum(dim=1))
    lap = deg - graph_batch
    return torch.trace(z_batch.T @ lap @ z_batch) / max(1, z_batch.shape[0])


def pretrain_ae(model: MultiViewAE, train_data: TrainData, args: argparse.Namespace, device: torch.device, save_path: str):
    print("Start AE pretraining...")
    ensure_dir(os.path.dirname(save_path))
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    optimizer = SGD(model.parameters(), lr=args.lrae, momentum=args.momentumae)
    n_samples = train_data.views_np[0].shape[0]
    indices = np.arange(n_samples)
    loss_curve = []
    model.train()
    for epoch in range(args.aeepochs):
        if args.ae_shuffle:
            np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, n_samples, args.batch_size):
            idx = indices[start:start + args.batch_size]
            batch_views = [x[idx].to(device) for x in train_data.views_torch]
            batch_we = train_data.we_torch[idx].to(device)
            batch_graphs = [g[np.ix_(idx, idx)].to(device) for g in train_data.graphs_torch]
            optimizer.zero_grad()
            recons, _, view_z = model(batch_views, batch_we)
            rec_loss = 0.0
            for v in range(len(batch_views)):
                rec_loss = rec_loss + weighted_mse_loss(recons[v], batch_views[v], batch_we[:, v])
            g_loss = 0.0
            for v in range(len(batch_views)):
                g_loss = g_loss + graph_loss_from_batch(view_z[v], batch_graphs[v])
            g_loss = g_loss / len(batch_views)
            loss = rec_loss + args.gammaae * g_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        epoch_loss = total_loss / max(1, batches)
        loss_curve.append(epoch_loss)
        torch.save(model.state_dict(), save_path)
        print(f"AE epoch {epoch + 1}/{args.aeepochs}, loss={epoch_loss:.6f}")
    ensure_dir(args.plot_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AE Pretraining Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "ae_pretrain_loss.png"), dpi=300)
    plt.close()


@torch.no_grad()
def get_fused_features(model: IDEC, train_data: TrainData, device: torch.device) -> np.ndarray:
    model.eval()
    _, z, _ = model.ae([x.to(device) for x in train_data.views_torch], train_data.we_torch.to(device))
    return z.cpu().numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    return (
        cluster_acc(y_true, y_pred),
        nmi_score(y_true, y_pred),
        ari_score(y_true, y_pred),
        purity_score(y_true, y_pred),
    )


def train_idec(train_data: TrainData, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model = IDEC(args.n_stacks, args.n_input, args.n_z, args.n_clusters, args.pretrain_path).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    model.pretrain(train_data, args, device)
    optimizer = Adam(model.parameters(), lr=args.lrkl)

    hidden_np = get_fused_features(model, train_data, device)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=args.seed)
    y_pred = kmeans.fit_predict(np.nan_to_num(hidden_np))
    cluster_layer = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    sample_weight = torch.ones(train_data.views_torch[0].shape[0], dtype=torch.float32)
    y_pred_last = np.copy(y_pred)
    loss_curve = []
    n_samples = train_data.views_np[0].shape[0]

    for epoch in range(args.maxiter):
        model.train()
        idx_all = np.arange(n_samples)
        if args.ae_shuffle:
            np.random.shuffle(idx_all)
        total_loss_epoch = 0.0
        updates = 0
        for _ in range(args.maxkl_epoch):
            for start in range(0, n_samples, args.batch_size):
                idx = idx_all[start:start + args.batch_size]
                batch_views = [x[idx].to(device) for x in train_data.views_torch]
                batch_we = train_data.we_torch[idx].to(device)
                batch_y = torch.tensor(y_pred[idx], dtype=torch.long, device=device)
                batch_sw = sample_weight[idx].to(device)
                batch_graphs = [g[np.ix_(idx, idx)].to(device) for g in train_data.graphs_torch]
                optimizer.zero_grad()
                _, kl_loss, view_z = model(batch_views, batch_we, batch_y, cluster_layer, batch_sw)
                g_loss = 0.0
                for v in range(len(batch_views)):
                    g_loss = g_loss + graph_loss_from_batch(view_z[v], batch_graphs[v])
                g_loss = g_loss / len(batch_views)
                fusion_loss = kl_loss + args.gammakl * g_loss
                fusion_loss.backward()
                optimizer.step()
                total_loss_epoch += float(fusion_loss.item())
                updates += 1
        total_loss_epoch /= max(1, updates)
        loss_curve.append(total_loss_epoch)

        model.eval()
        new_labels, prelosses = model.update_label([x.to(device) for x in train_data.views_torch], train_data.we_torch.to(device), cluster_layer)
        y_pred = new_labels.cpu().numpy()
        prelosses = prelosses.detach().cpu()
        lam = prelosses.mean() + (epoch / max(1, args.maxiter)) * prelosses.std()
        sample_weight = torch.where(prelosses <= lam, torch.ones_like(prelosses), torch.zeros_like(prelosses))

        fused = get_fused_features(model, train_data, device)
        centers = []
        for c in range(args.n_clusters):
            mask = y_pred == c
            if np.any(mask):
                centers.append(fused[mask].mean(axis=0))
            else:
                centers.append(kmeans.cluster_centers_[c])
        cluster_layer = torch.tensor(np.stack(centers), dtype=torch.float32, device=device)

        acc, nmi, ari, pur = evaluate(train_data.y, y_pred)
        delta_y = np.mean(y_pred != y_pred_last)
        y_pred_last = np.copy(y_pred)
        print(
            f"Epoch {epoch + 1}/{args.maxiter} | loss={total_loss_epoch:.6f} | "
            f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} PUR={pur:.4f} | "
            f"delta={delta_y:.6f} | selected={sample_weight.mean().item():.4f}"
        )
        if epoch > 10 and delta_y < args.tol:
            print(f"Training stopped early at epoch {epoch + 1}")
            break

    ensure_dir(args.plot_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CDIMC Fine-tuning Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "cdimc_finetune_loss.png"), dpi=300)
    plt.close()

    acc, nmi, ari, pur = evaluate(train_data.y, y_pred)
    print("=" * 70)
    print(f"Final ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, PUR={pur:.4f}")
    print("=" * 70)
    return y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDIMC-net adaptation for COIL20")
    parser.add_argument("--dataset", type=str, default="COIL20")
    parser.add_argument("--data-path", type=str, default="data/COIL20.mat")
    parser.add_argument("--save-dir", type=str, default="./cdimc_coil20_results")
    parser.add_argument("--plot-dir", type=str, default="./cdimc_coil20_results/loss_plots")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--percentDel", type=int, default=3)
    parser.add_argument("--n-stacks", type=int, default=4)
    parser.add_argument("--n-z", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--knn", type=int, default=11)
    parser.add_argument("--aeepochs", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--maxkl-epoch", type=int, default=5)
    parser.add_argument("--lrae", type=float, default=0.01)
    parser.add_argument("--lrkl", type=float, default=0.01)
    parser.add_argument("--momentumae", type=float, default=0.95)
    parser.add_argument("--gammaae", type=float, default=1e-3)
    parser.add_argument("--gammakl", type=float, default=1e-2)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--ae-shuffle", action="store_true")
    parser.add_argument("--pretrain-flag", type=int, default=0, help="0=train AE, 1=load pretrained model if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(args.plot_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.dataset.upper() != "COIL20":
        raise ValueError("This regenerated version is prepared specifically for COIL20.")

    views_raw, y = load_coil20_views(args.data_path)
    args.n_clusters = int(len(np.unique(y)))
    args.n_input = [int(v.shape[1]) for v in views_raw]
    print(f"Loaded COIL20: N={len(y)}, K={args.n_clusters}, V={len(views_raw)}, n_input={args.n_input}")

    WE = generate_missing_mask(len(y), len(views_raw), miss_rate=args.percentDel / 10.0, seed=args.seed)
    train_views, reorder_views = preprocess_views(views_raw, WE)
    train_views, reorder_views, y, WE = reorder_by_initial_clustering(train_views, reorder_views, y, WE, args.n_clusters, seed=args.seed)
    graphs = build_graphs(reorder_views, WE, knn=args.knn)

    train_data = TrainData(
        views_np=train_views,
        views_torch=[torch.tensor(v, dtype=torch.float32) for v in train_views],
        y=y,
        we_np=WE,
        we_torch=torch.tensor(WE, dtype=torch.float32),
        graphs_torch=[torch.tensor(g, dtype=torch.float32) for g in graphs],
    )

    args.pretrain_path = os.path.join(args.save_dir, f"coil20_pretrained_mr{args.percentDel}_ae{args.aeepochs}.pt")
    y_pred = train_idec(train_data, args, device)
    np.save(os.path.join(args.save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(args.save_dir, "labels.npy"), y)
    np.save(os.path.join(args.save_dir, "WE.npy"), WE)
    print(f"Saved outputs to {args.save_dir}")


if __name__ == "__main__":
    main()
