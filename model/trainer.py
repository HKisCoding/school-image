import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model.embedding import EmbeddingModel
from model.school import SchoolImage
from utils.loss import SpectralNetLoss
from utils.utils import pairwise_distance

INF = 1e-8


class Trainer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        self.lr = config["trainer"]["lr"]
        self.num_epochs = config["trainer"]["num_epochs"]
        self.weight_path = config["trainer"]["weight_path"]

        os.makedirs(self.weight_path, exist_ok=True)

        self.g_dim = config["school"]["g_dim"]
        self.gamma = config["school"]["gamma"]
        self.mu = config["school"]["mu"]
        self.delta = config["school"]["delta"]

        self.feat_size = config["school"]["feat_size"]
        self.out_feat = config["school"]["out_feat"]
        self.k = config["school"]["k"]

        self.model = SchoolImage(config, device)
        self.criterion = SpectralNetLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e4,
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

        self.embedding = EmbeddingModel(
            architecture=config["embedding"]["architecture"],
            input_dim=self.out_feat,
        ).to(self.device)

    def train(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if features is not None:
            self.features = features.view(features.size(0), -1).to(self.device)
            self.labels = labels.to(self.device)
