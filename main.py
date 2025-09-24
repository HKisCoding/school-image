import torch

from model.trainer import Trainer


def main():
    config = {
        "dataset": "coil-20",
        "trainer": {
            "lr": 0.001,
            "num_epochs": 100,
            "weight_path": "weights",
            "batch_size": 128,
        },
        "school": {
            "g_dim": 10,
            "gamma": 1.0,
            "mu": 1.0,
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

    config["input_dim"] = features.shape[1]
    config["cluster"] = len(torch.unique(labels))
    trainer = Trainer(config, device)
    trainer.train(features, labels)


if __name__ == "__main__":
    main()
