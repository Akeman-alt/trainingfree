import torch
import torch.nn as nn
import sys
import os

RESTYPES = "ACDEFGHIKLMNPQRSTVWY"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPNN_PATH = os.path.join(PROJECT_ROOT, "ProteinMPNN")

if MPNN_PATH not in sys.path:
    sys.path.insert(0, MPNN_PATH)

from protein_mpnn_utils import ProteinMPNN


class BaseReward(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, seq_samples, structure=None):
        raise NotImplementedError


class MPNNReward(BaseReward):
    """
    ProteinMPNN log P(seq | structure)
    支持：
        seq  : [B,L] 或 [N,B,L]
        struc: [B,L,3] 或 [B,L,4,3]
    """

    def __init__(self, device, checkpoint_path=None):

        super().__init__(device)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PROJECT_ROOT,
                "ProteinMPNN",
                "vanilla_model_weights",
                "v_48_020.pt",
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.model = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=checkpoint["num_edges"],
            augment_eps=0.0,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False


    def _prepare_structure(self, structure):
        """
        convert structure to [B,L,4,3]
        """

        if structure.ndim == 4:
            return structure

        if structure.ndim != 3:
            raise ValueError(f"Invalid structure shape {structure.shape}")

        B, L, _ = structure.shape

        X = torch.zeros(B, L, 4, 3, device=structure.device)

        # CA index = 1
        X[:, :, 1, :] = structure

        return X


    def forward(self, seq_samples, structure):

        if structure is None:
            raise ValueError("MPNNReward requires structure")

        # ------------------------------------------------
        # flatten sampling dimension
        # ------------------------------------------------

        if seq_samples.ndim == 3:

            N, B, L = seq_samples.shape

            seq_samples = seq_samples.reshape(N * B, L)

            if structure.ndim == 3:
                structure = structure.unsqueeze(0).expand(N, -1, -1, -1)
                structure = structure.reshape(N * B, L, 3)

            elif structure.ndim == 4:
                structure = structure.unsqueeze(0).expand(N, -1, -1, -1, -1)
                structure = structure.reshape(N * B, L, 4, 3)

        B, L = seq_samples.shape

        # ------------------------------------------------
        # convert CA -> full backbone tensor
        # ------------------------------------------------

        X = self._prepare_structure(structure)

        X = X.to(self.device)
        seq_samples = seq_samples.to(self.device)

        mask = torch.ones(B, L, device=self.device)

        chain_M = torch.ones(B, L, device=self.device)

        residue_idx = torch.arange(L, device=self.device).unsqueeze(0).repeat(B, 1)

        chain_encoding_all = torch.zeros(B, L, device=self.device)

        randn = torch.zeros(B, L, device=self.device)

        # ------------------------------------------------
        # forward MPNN
        # ------------------------------------------------

        logits = self.model(
            X,
            seq_samples,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            randn,
        )

        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs,
            -1,
            seq_samples.unsqueeze(-1),
        ).squeeze(-1)

        score = token_log_probs.mean(dim=-1)

        return score