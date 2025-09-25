import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import trange
from sklearn.cluster import KMeans

from model.spectral_net import SpectralNetModel
from utils.logger import Logger
from utils.loss import SpectralNetLoss
from utils.metrics import run_evaluate_with_labels
from utils.utils import get_nearest_neighbors, get_gaussian_kernel, compute_scale


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

class SpectralNetTrainer(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.is_sparse = self.config["spectral_net"]["is_sparse"]
        self.config = config
        self.lr = self.config["trainer"]["lr"]
        self.n_nbg = self.config["spectral_net"]["n_nbg"]
        self.min_lr = self.config["spectral_net"]["min_lr"]
        self.epochs = self.config["trainer"]["num_epochs"]
        self.scale_k = self.config["spectral_net"]["scale_k"]
        self.lr_decay = self.config["spectral_net"]["lr_decay"]
        self.patience = self.config["spectral_net"]["patience"]
        self.architecture = self.config["spectral_net"]["architecture"]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.is_local_scale = self.config["spectral_net"]["is_local_scale"]
        self.cluster = self.config["cluster"]
        self.model = SpectralNetModel(
            architecture=self.architecture,
            input_dim=self.config["input_dim"],
            orthonorm_weights=self.config["spectral_net"]["orthonorm_weights"],
        ).to(self.device)

        self.logger = Logger(name=self.__class__.__name__)

        self.criterion = SpectralNetLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def create_affingity_matrix_from_scale(
        self, X: torch.Tensor, scale: float
    ) -> torch.Tensor:
        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        # scale = compute_scale(Dis, k=scale, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

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

    def train(self, X: torch.Tensor):
        X = X.view(X.size(0), -1)

        self.counter = 0
        train_loader, ortho_loader, _ = self._get_data_loader(X, train_ratio=1)

        t = trange(self.epochs, desc="Training")
        total_train_loss = []
        for epoch in t:
            train_loss = 0.0
            for X_grad, X_orth in zip(train_loader, ortho_loader):
                X_grad = X_grad[0]
                X_orth = X_orth[0]
                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device=self.device)
                X_orth = X_orth.view(X_orth.size(0), -1)

                # Orthogonalization step
                self.model.eval()
                self.model(X_orth, should_update_orth_weights=True)

                # Gradient step
                self.model.train()
                self.optimizer.zero_grad()

                Y, _, _ = self.model(X_grad, should_update_orth_weights=False)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.scheduler.step(train_loss)

            t.set_description(
                "Train Loss: {:.7f}".format(
                    train_loss
                )
            )
            total_train_loss.append(train_loss)
            t.refresh()

    def evaluate(self, X: torch.Tensor, labels: torch.Tensor) -> dict:
        X = X.view(X.size(0), -1)
        X = X.to(self.device)

        with torch.no_grad():
            self.embeddings_, _, _ = self.model(
                X, should_update_orth_weights=True
            )
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.cluster, n_init='auto').fit(self.embeddings_)
            cluster_assignments = kmeans.labels_
            results = run_evaluate_with_labels(
                cluster_assignments, labels.cpu().numpy(), self.cluster
            )

        return results
