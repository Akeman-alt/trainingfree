import torch
import torch.nn as nn

# 标准氨基酸顺序
RESTYPES = 'ACDEFGHIKLMNPQRSTVWY'

class BaseReward(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, seq_samples):
        raise NotImplementedError

class TargetReward(BaseReward):
    """
    简单粗暴：计算指定的一组氨基酸的总占比。
    例如 target_chars=['A', 'V']，则 Reward = (A的数量 + V的数量) / 总长度
    """
    def __init__(self, device, target_chars=['A'], vocab_order=RESTYPES):
        super().__init__(device)
        
        self.target_chars = target_chars
        self.vocab_size = len(vocab_order) + 1
        
        # 创建一个查分表：是目标氨基酸的位置填 1.0，否则填 0.0
        self.reward_mask = torch.zeros(self.vocab_size, device=device)
        
        print(f"🎯 初始化奖励函数: 增加 {target_chars} 的含量")
        
        found_any = False
        for char in target_chars:
            if char in vocab_order:
                idx = vocab_order.index(char)
                self.reward_mask[idx] = 1.0
                found_any = True
            else:
                print(f"⚠️ 警告: 氨基酸 {char} 不在词表中！")
        
        if not found_any:
            raise ValueError("目标氨基酸列表无效，无法计算奖励！")

    def forward(self, seq_samples):
        # seq_samples: [N, B, L] (整数索引)
        
        # 1. 查表：直接把 token ID 变成 0 或 1
        # [N, B, L] -> [N, B, L] (float)
        hits = self.reward_mask[seq_samples.long()]
        
        # 2. 算平均值 (占比)
        # [N, B]
        return hits.mean(dim=-1)