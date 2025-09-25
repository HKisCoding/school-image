import torch

from model.trainer import Trainer
from model.spectral_net_trainer import SpectralNetTrainer


def train_school():
    config = {
        "dataset": "coil-20",
        "trainer": {
            "lr": 0.001,
            "num_epochs": 100,
            "weight_path": "weights",
            "batch_size": 1024,
        },
        "school": {
            "g_dim": 10,
            "gamma": 1.0,
            "mu": 0.1,
            "delta": 1.0,
            "feat_size": 128,
            "out_feat": 256,
            "k": 10,
        },
        "spectral_net": {
            "architecture": [1024, 1024, 256],
            "orthonorm_weights": None,
        },
        "embedding": {
            "architecture": [1024, 256],
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_path = f"dataset/{config['dataset']}_Feature.pt"
    label_path = f"dataset/{config['dataset']}_Label.pt"
    features = torch.load(feature_path, map_location=device)
    labels = torch.load(label_path, map_location=device)

    config["trainer"]["batch_size"] = features.shape[0]
    config["input_dim"] = features.shape[1]
    config["cluster"] = len(torch.unique(labels))
    config["spectral_net"]["architecture"].append(len(torch.unique(labels)))
    trainer = Trainer(config, device)
    trainer.train(features)
    results = trainer.evaluate(features, labels)
    print(results)


def train_spectral_net():
    config = {
        "dataset": "coil-20",
        "trainer": {
            "lr": 0.001,
            "num_epochs": 100,
            "weight_path": "weights",
            "batch_size": 256,
        },
        "spectral_net": {
            "architecture": [1024, 1024, 256],
            "orthonorm_weights": None,
            "n_nbg": 10,
            "min_lr": 1e-8,
            "scale_k": 10,
            "lr_decay": 0.1,
            "patience": 10,
            "is_sparse": False,
            "is_local_scale": False,
        },
        "cluster": 20,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_path = f"dataset/{config['dataset']}_Feature.pt"
    label_path = f"dataset/{config['dataset']}_Label.pt"
    features = torch.load(feature_path, map_location=device)
    labels = torch.load(label_path, map_location=device)
    config["input_dim"] = features.shape[1]
    config["cluster"] = len(torch.unique(labels))
    config["spectral_net"]["architecture"].append(len(torch.unique(labels)))
    trainer = SpectralNetTrainer(config, device)
    trainer.train(features)
    results = trainer.evaluate(features, labels)
    print(results)



if __name__ == "__main__":
    train_spectral_net()
