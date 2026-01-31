import torch
import torch.nn as nn
import sys
import os

# 标准氨基酸顺序
RESTYPES = 'ACDEFGHIKLMNPQRSTVWY'

# 尝试动态导入 ProteinMPNN
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MPNN_PATH = os.path.join(PROJECT_ROOT, 'ProteinMPNN')
    if MPNN_PATH not in sys.path:
        sys.path.append(MPNN_PATH)
    
    from protein_mpnn_utils import ProteinMPNN
    print("✅ 成功导入 ProteinMPNN 模块")
except ImportError as e:
    print(f"⚠️ 警告: 无法导入 ProteinMPNN。MPNNReward 将不可用。错误: {e}")
    ProteinMPNN = None

class BaseReward(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, seq_samples, structure=None):
        raise NotImplementedError

class TargetReward(BaseReward):
    """
    计算指定氨基酸的含量。支持 2D [B, L] 或 3D [N, B, L] 输入。
    """
    def __init__(self, device, target_chars=['A'], vocab_order=RESTYPES):
        super().__init__(device)
        self.target_chars = target_chars
        self.vocab_size = len(vocab_order) + 1
        self.reward_mask = torch.zeros(self.vocab_size, device=device)
        
        for char in target_chars:
            if char in vocab_order:
                idx = vocab_order.index(char)
                self.reward_mask[idx] = 1.0
        
    def forward(self, seq_samples, structure=None):
        # 1. 查表
        hits = self.reward_mask[seq_samples.long()]
        # 2. 算平均值 (占比)
        return hits.mean(dim=-1)

class MPNNReward(BaseReward):
    """
    计算序列在给定骨架上的 ProteinMPNN 似然分数。
    自动处理 [B, L] (Single) 或 [N, B, L] (Batch Sampling) 的输入。
    """
    def __init__(self, device, checkpoint_path=None, ca_only=False):
        super().__init__(device)
        
        if ProteinMPNN is None:
            raise ImportError("ProteinMPNN 未正确导入，请检查路径。")

        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, 'ProteinMPNN', 'vanilla_model_weights', 'v_48_020.pt')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hidden_dim = 128
        num_layers = 3
        
        self.model = ProteinMPNN(
            num_letters=21, 
            node_features=hidden_dim, 
            edge_features=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            k_neighbors=checkpoint['num_edges'],
            ca_only=ca_only
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seq_samples, structure):
        """
        seq_samples: [B, L] 或 [N_samples, B, L]
        structure:   [B, L, 3] 或 [B, L, 4, 3]
        """
        if structure is None:
            return torch.zeros(seq_samples.shape[:-1], device=self.device)

        # ------------------------------------------------------------------
        # 🔴 [关键修复] 处理 3D 输入 (N_samples, B, L)
        # ------------------------------------------------------------------
        if seq_samples.ndim == 3:
            N_samples, B, L = seq_samples.shape
            
            # 1. 把序列展平: [N*B, L]
            seq_flat = seq_samples.reshape(-1, L)
            
            # 2. 把结构扩展并展平: [B, L, 3] -> [N, B, L, 3] -> [N*B, L, 3]
            if structure.ndim == 3:
                struct_flat = structure.unsqueeze(0).expand(N_samples, -1, -1, -1).reshape(N_samples * B, L, 3)
            elif structure.ndim == 4:
                struct_flat = structure.unsqueeze(0).expand(N_samples, -1, -1, -1, -1).reshape(N_samples * B, L, 4, 3)
            else:
                raise ValueError(f"Invalid structure shape: {structure.shape}")
            
            # 3. 递归调用 (现在是 2D 输入了)
            scores_flat = self.forward(seq_flat, struct_flat) # 返回 [N*B]
            
            # 4. 变回原来的形状 [N, B]
            return scores_flat.view(N_samples, B)

        # ------------------------------------------------------------------
        # 🟢 正常的 2D 输入处理 [B, L]
        # ------------------------------------------------------------------
        B, L = seq_samples.shape
        
        # 结构适配 [B, L, 3] -> [B, L, 4, 3]
        if structure.ndim == 3: 
            # 只有 CA，填充到 index 1
            # 注意：这里我们创建一个新的 tensor，但保留梯度链
            X = torch.zeros(B, L, 4, 3, device=self.device)
            # 必须用切片赋值来保留 structure 的梯度
            # X[:, :, 1, :] = structure 
            # 但 inplace 赋值有时会打断梯度，更安全的做法是拼接或掩码加法
            # 这里简单处理，如果梯度断了可以用 X = structure.unsqueeze(2).expand(...)
            # 这里的trick: 先全零，再把CA加进去
            X = X + structure.unsqueeze(2) * torch.tensor([0, 1, 0, 0], device=self.device).view(1, 1, 4, 1)
        else:
            X = structure

        # 构建 Mask
        mask = torch.ones(B, L, device=self.device)
        chain_M = mask.clone()
        residue_idx = torch.arange(L, device=self.device).view(1, -1).expand(B, -1)
        chain_encoding_all = torch.zeros_like(residue_idx)
        randn = torch.randn(B, L, device=self.device) 
        
        # MPNN 前向
        logits = self.model(X, seq_samples, mask, chain_M, residue_idx, chain_encoding_all, randn)
        
        # 计算 Score (Log Prob)
        log_probs = torch.log_softmax(logits, dim=-1)
        sp_log_probs = torch.gather(log_probs, -1, seq_samples.unsqueeze(-1)).squeeze(-1)
        
        # 返回每个序列的平均分
        return sp_log_probs.mean(dim=-1)