from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from .encoder import BayesianTransformerEncoder
from .bayesian_layers import BayesianLinear

# ========== Dataset & utilities ==========

class TripletDataset(Dataset):
    """Holds (x1, x2_pos, x2_neg) triplets; each is shape (N_features,)"""
    def __init__(self, X1, X2_pos, X2_neg):
        super().__init__()
        self.X1 = X1
        self.X2_pos = X2_pos
        self.X2_neg = X2_neg

    def __len__(self):
        return self.X1.shape[0]

    def __getitem__(self, idx):
        return self.X1[idx], self.X2_pos[idx], self.X2_neg[idx]

def prepare_triplets(frame_pairs, df, features_to_use, normalize_features=True):
    """
    Gather all (x1, x2_pos, x2_neg) from given frame pairs.
    Positive: nearest centroid in t+1; Negative: random different.
    Returns torch.FloatTensors or (None, None, None) if empty.
    """
    X1_list, X2_pos_list, X2_neg_list = [], [], []

    if normalize_features:
        all_data = df[features_to_use].values.astype(np.float32)
        mean_ = all_data.mean(axis=0)
        std_ = all_data.std(axis=0) + 1e-8
    else:
        mean_, std_ = 0.0, 1.0

    for (frame1, frame2) in frame_pairs:
        sec1 = df[df["FrameID"] == int(frame1)]
        sec2 = df[df["FrameID"] == int(frame2)]
        if len(sec1) == 0 or len(sec2) == 0:
            continue

        x1_full_np = sec1[features_to_use].to_numpy(dtype=np.float32)
        x2_full_np = sec2[features_to_use].to_numpy(dtype=np.float32)

        x1_full_np = (x1_full_np - mean_) / std_
        x2_full_np = (x2_full_np - mean_) / std_

        c1 = sec1[["Centroid_X", "Centroid_Y"]].to_numpy(dtype=np.float32)
        c2 = sec2[["Centroid_X", "Centroid_Y"]].to_numpy(dtype=np.float32)
        # L2 distances
        dists = np.linalg.norm(c1[:,None,:] - c2[None,:,:], axis=-1)
        pos_idx = np.argmin(dists, axis=1)

        N = x1_full_np.shape[0]
        neg_idx = np.random.randint(0, x2_full_np.shape[0], size=N)
        for i in range(N):
            while neg_idx[i] == pos_idx[i]:
                neg_idx[i] = np.random.randint(0, x2_full_np.shape[0])

        X1_list.append(x1_full_np)
        X2_pos_list.append(x2_full_np[pos_idx])
        X2_neg_list.append(x2_full_np[neg_idx])

    if len(X1_list) == 0:
        return None, None, None

    X1 = torch.from_numpy(np.concatenate(X1_list, axis=0)).float()
    X2p = torch.from_numpy(np.concatenate(X2_pos_list, axis=0)).float()
    X2n = torch.from_numpy(np.concatenate(X2_neg_list, axis=0)).float()
    return X1, X2p, X2n

# ========== Model ==========

class BayesianTransformerForCellTracking(nn.Module):
    """
    The same backbone you wrote, with 'feature_attention' gating + encoder + (mu, logvar) heads.
    """
    def __init__(self, input_dim=10, embed_dim=64, num_heads=2, ff_hidden_dim=256,
                 num_layers=2, output_dim=2, prior_mu=0.0, prior_sigma=0.1,
                 dropout=0.1, use_layernorm=True):
        super().__init__()
        self.feature_attention = BayesianLinear(input_dim, input_dim, prior_mu, prior_sigma)
        self.input_proj = BayesianLinear(input_dim, embed_dim, prior_mu, prior_sigma)
        self.encoder = BayesianTransformerEncoder(
            embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma,
            dropout=dropout, use_layernorm=use_layernorm
        )
        self.out_mu = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)
        self.out_logvar = BayesianLinear(embed_dim, output_dim, prior_mu, prior_sigma)
        self.alpha = 0.9
        self.register_buffer('prev_attention', torch.zeros(input_dim))

    def forward(self, x, sample=True, frame_idx=None, collect_attn=False):
        attn_scores = self.feature_attention(x, sample=sample)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        if self.training:
            mean_attn = attn_scores.mean(dim=0)
            smoothed = self.alpha * self.prev_attention + (1 - self.alpha) * mean_attn.squeeze(0)
            self.prev_attention = smoothed.detach()
        else:
            smoothed = attn_scores.mean(dim=0).squeeze(0)

        x = x * smoothed.unsqueeze(0)
        embed = self.input_proj(x, sample=sample)
        enc_out = self.encoder(embed, sample=sample)
        mu = self.out_mu(enc_out, sample=sample)
        logvar = self.out_logvar(enc_out, sample=sample)
        return mu, logvar, (smoothed if collect_attn else None)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_bnn(
    bnn: BayesianTransformerForCellTracking,
    frame_pairs: List[Tuple[int,int]],
    df,
    features_to_use: List[str],
    num_epochs=100,
    lr=1e-3,
    margin=0.2,
    weight_decay=1e-5,
    batch_size=128,
    early_stopping_patience=10,
    reduce_lr_patience=5,
    device="auto",
    kl_beta: float = 0.0
):
    """
    Advanced training loop with triplets, margin-based contrastive loss,
    LR scheduling, early stopping, and optional KL regularization.
    """
    X1_t, X2_pos_t, X2_neg_t = prepare_triplets(frame_pairs, df, features_to_use, normalize_features=True)
    if X1_t is None:
        print("[ERROR] No triplets to train on.")
        return bnn

    dataset = TripletDataset(X1_t, X2_pos_t, X2_neg_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device == "auto":
        device = "cuda" if (torch.cuda.is_available()) else "cpu"
    bnn.to(device)

    opt = torch.optim.Adam(bnn.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=reduce_lr_patience, verbose=True)

    best_loss, no_impr = float("inf"), 0
    for epoch in range(num_epochs):
        bnn.train()
        total = 0.0
        for (x1, x2p, x2n) in loader:
            x1 = x1.to(device).unsqueeze(1)
            x2p = x2p.to(device).unsqueeze(1)
            x2n = x2n.to(device).unsqueeze(1)

            opt.zero_grad()

            mu1, logv1, _ = bnn(x1, sample=True)
            mu2p, logv2p, _ = bnn(x2p, sample=True)
            mu2n, logv2n, _ = bnn(x2n, sample=True)

            z1 = reparameterize(mu1, logv1)
            z2p = reparameterize(mu2p, logv2p)
            z2n = reparameterize(mu2n, logv2n)

            pos = (z1 - z2p).pow(2).mean(dim=1)
            neg = (z1 - z2n).pow(2).mean(dim=1)
            contrast = torch.clamp(pos - neg + margin, min=0.0).mean()

            kl = 0.0
            if kl_beta > 0.0:
                kl_terms = []
                for m in bnn.modules():
                    if hasattr(m, "kl"):
                        kl_terms.append(m.kl())
                kl = sum(kl_terms) if len(kl_terms) > 0 else 0.0
            loss = contrast + kl_beta * kl

            loss.backward()
            clip_grad_norm_(bnn.parameters(), 1.0)
            opt.step()

            total += loss.item()

        avg = total / (len(loader) + 1e-8)
        sched.step(avg)

        improved = avg < (best_loss - 1e-6)
        best_loss = min(best_loss, avg)
        no_impr = 0 if improved else (no_impr + 1)
        print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg:.4f}  no_improvement={no_impr}")
        if no_impr >= early_stopping_patience:
            print("Early stopping.")
            break

    return bnn
