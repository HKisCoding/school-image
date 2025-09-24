import torch
import torch.nn as nn

from model.spectral_net import SpectralNetModel
from utils.utils import EProjSimplex_new_matrix


class SchoolImage(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.spectral_net = SpectralNetModel(
            architecture=config["spectral_net"]["architecture"],
            input_dim=config["input_dim"],
            orthonorm_weights=config["spectral_net"]["orthonorm_weights"],
        ).to(device)

    def forward(
        self,
        x: torch.Tensor,
        x_orth: torch.Tensor,
        idx,
        alpha,
        beta,
        is_training: bool = True,
    ):
        if is_training:
            self.spectral_net(
                x=x_orth,
                semantic_out_dim=self.config["school"]["out_feat"],
                should_update_orth_weights=True,
            )

        device = self.config["device"] or torch.device("cpu")
        # affinity_matrix = create_affinity_matrix(
        #     X=x,
        #     n_neighbors=self.config.school.n_neighbors,
        #     scale_k=self.config.school.k,
        #     device=device,
        # )
        # self.init_graph = affinity_to_adjacency(affinity_matrix)

        Y, semantic_H, ortho_H = self.spectral_net(
            x=x,
            semantic_out_dim=self.config["school"]["out_feat"],
            should_update_orth_weights=False,
        )

        num = x.shape[0]
        A = torch.zeros(num, num, device=device)
        idxa0 = idx[:, 1 : self.config["school"]["k"] + 1]
        dfi = torch.sqrt(torch.sum((Y.unsqueeze(1) - Y[idxa0]) ** 2, dim=2) + 1e-8).to(
            device
        )
        dxi = torch.sqrt(
            torch.sum((ortho_H.unsqueeze(1) - ortho_H[idxa0]) ** 2, dim=2) + 1e-8
        ).to(device)
        ad = -(dxi + beta * dfi) / (2 * alpha)
        A.scatter_(1, idxa0.to(device), EProjSimplex_new_matrix(ad))

        embs_hom = torch.mm(A, semantic_H)

        return embs_hom, A, Y
