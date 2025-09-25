import copy
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from model.embedding import FullyConnect
from model.school import SchoolImage
from utils.logger import Logger
from utils.loss import SpectralNetLoss
from utils.metrics import run_evaluate_with_labels
from utils.utils import pairwise_distance

INF = 1e-8


class PaddedDataset(Dataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.original_length = len(dataset)
        # Calculate how many samples we need to add to make the last batch full
        self.padded_length = (
            (self.original_length + self.batch_size - 1) // self.batch_size
        ) * self.batch_size

    def __len__(self):
        return self.padded_length

    def __getitem__(self, idx):
        if idx < self.original_length:
            return self.dataset[idx]
        else:
            # For indices beyond the original length, wrap around to the beginning
            return self.dataset[idx % self.original_length]


class Trainer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        self.lr = config["trainer"]["lr"]
        self.num_epochs = config["trainer"]["num_epochs"]
        self.weight_path = config["trainer"]["weight_path"]
        self.batch_size = config["trainer"]["batch_size"]
        self.cluster = config["cluster"]

        os.makedirs(self.weight_path, exist_ok=True)

        self.g_dim = config["school"]["g_dim"]
        self.gamma = config["school"]["gamma"]
        self.mu = config["school"]["mu"]
        self.delta = config["school"]["delta"]

        self.feat_size = config["school"]["feat_size"]
        self.out_feat = config["school"]["out_feat"]
        self.k = config["school"]["k"]

        self.model = SchoolImage(config, device)
        self.logger = Logger(name=self.__class__.__name__)

        self.criterion = SpectralNetLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.consistency_encoder = nn.Sequential(
            nn.Linear(
                self.out_feat,
                self.g_dim,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ).to(self.device)

        self.fully_connect = FullyConnect(
            in_ft=config["input_dim"],
            out_ft=self.out_feat,
        ).to(self.device)

    def _get_data_loader(self, X: torch.Tensor, train_ratio: float = 0.9):
        train_size = int(train_ratio * len(X))
        valid_size = len(X) - train_size
        dataset = TensorDataset(X)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        trainset, valset = (
            copy.deepcopy(train_dataset),
            copy.deepcopy(valid_dataset),
        )

        padded_train_dataset = PaddedDataset(trainset, self.batch_size)
        padded_valid_dataset = PaddedDataset(valset, self.batch_size)

        train_loader = DataLoader(
            padded_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        ortho_loader = DataLoader(
            padded_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            padded_valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return train_loader, ortho_loader, valid_loader

    def compute_graph_parameters(self, X_grad: torch.Tensor):
        """Compute graph parameters from input features.

        Args:
            X_grad (torch.Tensor): Input features tensor

        Returns:
            tuple: (A, alpha, beta, idx) where:
                - A: Adjacency matrix
                - alpha: Mean of radius values
                - beta: Mean of radius values (same as alpha)
                - idx: Sorted indices of pairwise distances
        """
        X = X_grad.cpu()
        distX = pairwise_distance(X)
        # Sort the distances and get the sorted indices
        distX_sorted, idx = torch.sort(distX, dim=1)
        num = X_grad.shape[0]
        A = torch.zeros(num, num)
        rr = torch.zeros(num)
        for i in range(num):
            di = distX_sorted[i, 1 : self.k + 1]
            rr[i] = 0.5 * (self.k * di[self.k - 1] - torch.sum(di[: self.k]))
            id = idx[i, 1 : self.k + 1]
            A[i, id] = (di[self.k - 1] - di) / (
                self.k * di[self.k - 1]
                - torch.sum(di[: self.k])
                + torch.finfo(torch.float).eps
            )
        alpha = rr.mean()
        # r = 0

        beta = rr.mean()

        return A, alpha, beta, idx

    def train(
        self,
        features: Optional[torch.Tensor] = None,
    ):
        if features is not None:
            self.features = features.view(features.size(0), -1).to(self.device)

        train_loader, ortho_loader, _ = self._get_data_loader(
            self.features, train_ratio=1
        )
        start_time = time.time()
        pbar = tqdm(range(self.num_epochs), desc="Training")

        for epoch in pbar:
            train_loss = 0
            epoch_spectral_loss = 0
            epoch_node_consistency_loss = 0
            epoch_cluster_consistency_loss = 0

            self.model.train()
            epoch_start_time = time.time()
            for X_grad, X_orth in zip(train_loader, ortho_loader):
                X_grad = X_grad[0]
                X_orth = X_orth[0]
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.view(X_orth.size(0), -1)

                A, alpha, beta, idx = self.compute_graph_parameters(X_grad)

                embs_hom, A_updated, Y = self.model(
                    x=X_grad,
                    x_orth=X_orth,
                    idx=idx,
                    alpha=alpha,
                    beta=beta,
                    is_training=True,
                )

                spectral_loss = self.criterion(A_updated, Y)

                p_i = Y.sum(0).view(-1)
                p_i = (p_i + INF) / (p_i.sum() + INF)
                p_i = torch.abs(p_i)
                # The second term in Eq. (13): entropy loss
                entrpoy_loss = (
                    math.log(p_i.size(0) + INF)
                    + ((p_i + INF) * torch.log(p_i + INF)).sum()
                )
                spectral_loss = spectral_loss + self.gamma * entrpoy_loss

                features_emb = self.fully_connect(X_grad)

                features_emb = self.consistency_encoder(features_emb)
                embs_hom = self.consistency_encoder(embs_hom)

                # The first term in Eq. (15): invariance loss
                inter_c = embs_hom.T @ features_emb.detach()
                inter_c = F.normalize(inter_c, p=2, dim=1)
                loss_inv = -torch.diagonal(inter_c).sum()

                intra_c = (embs_hom).T @ (embs_hom).contiguous() + (features_emb).T @ (
                    features_emb
                ).contiguous()
                intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
                loss_uni = torch.log(intra_c)
                loss_consistency = loss_inv + self.mu * loss_uni

                # The second term in Eq. (13): cluster-level loss
                Y_hat = torch.argmax(Y, dim=1)
                cluster_center = torch.stack([
                    torch.mean(embs_hom[Y_hat == i], dim=0) for i in range(self.cluster)
                ])  # Shape: (num_clusters, embedding_dim)
                # Gather positive cluster centers
                positive = cluster_center[Y_hat]
                # The first term in Eq. (11)
                inter_c = positive.T @ features_emb
                inter_c = F.normalize(inter_c, p=2, dim=1)
                loss_spe_inv = -torch.diagonal(
                    inter_c
                ).sum()  # Shape: (num_clusters, embedding_dim)

                loss = (
                    spectral_loss
                    + self.mu * loss_consistency
                    + self.delta * (loss_spe_inv)
                )

                # Backward pass
                loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                epoch_spectral_loss += spectral_loss.item()
                epoch_node_consistency_loss += loss_consistency.item()
                epoch_cluster_consistency_loss += loss_spe_inv.item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "spec_loss": f"{spectral_loss.item():.4f}",
                        "loss_consistency": f"{loss_consistency.item():.4f}",
                        "loss_spe_inv": f"{loss_spe_inv.item():.4f}",
                    }
                )

            train_loss /= len(train_loader)
            epoch_spectral_loss /= len(train_loader)
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            self.scheduler.step(train_loss)

            # Log detailed metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.num_epochs}] "
                    f"Loss: {train_loss:.4f} "
                    f"Spectral Loss: {epoch_spectral_loss:.4f} "
                    f"Node Consistency Loss: {epoch_node_consistency_loss:.4f} "
                    f"Cluster Loss: {epoch_cluster_consistency_loss:.4f} "
                    f"Time: {epoch_time:.2f}s"
                )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

    def evaluate(self, X: torch.Tensor, labels: torch.Tensor) -> dict:
        X = X.view(X.size(0), -1)
        X = X.to(self.device)

        with torch.no_grad():
            self.embeddings_, _, _ = self.model.spectral_net(
                X, should_update_orth_weights=True
            )
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.cluster, n_init='auto').fit(self.embeddings_)
            cluster_assignments = kmeans.labels_
            results = run_evaluate_with_labels(
                cluster_assignments, labels.cpu().numpy(), self.cluster
            )

        return results
