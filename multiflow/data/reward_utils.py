"""
统一的 Reward 计算工具模块

支持两种输入格式：
1. 字符串序列（字母表示）：如 "ACDEFG..."
2. 数字索引张量（数字表示）：如 torch.tensor([0, 1, 2, ...])
"""

import torch
from typing import Union, List
from multiflow.data import residue_constants


def calculate_reward(
    sequence: Union[str, torch.Tensor],
    target_chars: Union[str, List[str]] = 'A',
    device: str = None
) -> Union[float, torch.Tensor]:
    """
    计算序列中目标氨基酸的占比（Reward）
    
    Args:
        sequence: 输入序列，可以是：
            - str: 字符串序列，如 "ACDEFG..."
            - torch.Tensor: 数字索引张量，形状可以是 [L], [B, L], [N, B, L] 等
        target_chars: 目标氨基酸字符，可以是单个字符 'A' 或列表 ['A', 'V']
        device: 当输入是张量时，指定设备（可选）
    
    Returns:
        - 如果输入是字符串：返回 float，表示目标氨基酸的占比
        - 如果输入是张量：返回 torch.Tensor，形状为去除最后一个维度后的形状
          例如 [N, B, L] -> [N, B]
    
    Examples:
        >>> # 字符串输入
        >>> reward = calculate_reward("ACDEFG", target_chars='A')
        >>> # 返回 1/6 ≈ 0.167
        
        >>> # 张量输入（单个序列）
        >>> seq_tensor = torch.tensor([0, 1, 2, 3, 4, 5])  # A, R, N, D, C, Q
        >>> reward = calculate_reward(seq_tensor, target_chars='A')
        >>> # 返回 tensor(0.167)
        
        >>> # 张量输入（批量）
        >>> seq_batch = torch.tensor([[0, 1, 2], [0, 0, 1]])  # [B=2, L=3]
        >>> reward = calculate_reward(seq_batch, target_chars='A')
        >>> # 返回 tensor([0.333, 0.667])
    """
    # 统一 target_chars 为列表格式
    if isinstance(target_chars, str):
        target_chars = [target_chars]
    
    # 获取目标氨基酸的索引
    target_indices = []
    for char in target_chars:
        if char in residue_constants.restype_order:
            target_indices.append(residue_constants.restype_order[char])
        elif char == 'X' and 'X' in residue_constants.restypes_with_x:
            target_indices.append(residue_constants.restypes_with_x.index('X'))
        else:
            raise ValueError(f"目标氨基酸 '{char}' 不在标准词表中！")
    
    # 转换为张量以便后续计算
    if device is None:
        device = 'cpu'
    target_indices_tensor = torch.tensor(target_indices, device=device, dtype=torch.long)
    
    # 处理字符串输入
    if isinstance(sequence, str):
        if len(sequence) == 0:
            return 0.0
        count = sum(1 for char in sequence if char in target_chars)
        return count / len(sequence)
    
    # 处理张量输入
    elif isinstance(sequence, torch.Tensor):
        # 确保在正确的设备上
        if device is None:
            device = sequence.device
        target_indices_tensor = target_indices_tensor.to(device)
        sequence = sequence.to(device)
        
        # 将目标索引转换为 mask
        # 对于每个目标索引，检查序列中哪些位置匹配
        # 使用广播机制：sequence [..., L] 与 target_indices [num_targets] 比较
        # 结果: [..., L, num_targets] -> 在最后一个维度上取 max -> [..., L]
        matches = torch.zeros_like(sequence, dtype=torch.float)
        for idx in target_indices:
            matches = matches + (sequence == idx).float()
        matches = matches.clamp(0, 1)  # 确保每个位置最多匹配一次
        
        # 计算平均值（占比），在最后一个维度上求平均
        reward = matches.mean(dim=-1)
        
        return reward
    
    else:
        raise TypeError(f"不支持的输入类型: {type(sequence)}，期望 str 或 torch.Tensor")


def create_reward_fn(
    target_chars: Union[str, List[str]] = 'A',
    device: str = None
):
    """
    创建一个 reward 函数，用于 guided_interpolant.py 中的 reward_fn
    
    Args:
        target_chars: 目标氨基酸字符，可以是单个字符 'A' 或列表 ['A', 'V']
        device: 设备（可选，如果为 None 则从输入张量自动获取）
    
    Returns:
        一个函数，接收 [N, B, L] 形状的张量，返回 [N, B] 形状的 reward
    
    Example:
        >>> reward_fn = create_reward_fn(target_chars='A')
        >>> seq_samples = torch.randint(0, 20, (8, 2, 100))  # [N=8, B=2, L=100]
        >>> rewards = reward_fn(seq_samples)  # [8, 2]
    """
    def reward_fn(seq_samples: torch.Tensor) -> torch.Tensor:
        """
        计算 reward
        
        Args:
            seq_samples: [N, B, L] 形状的张量，表示 N 个样本，每个样本有 B 个批次，每个批次有 L 个位置
        
        Returns:
            [N, B] 形状的张量，表示每个样本每个批次的 reward
        """
        # 如果 device 未指定，从输入张量自动获取
        actual_device = device if device is not None else seq_samples.device
        return calculate_reward(seq_samples, target_chars=target_chars, device=actual_device)
    
    return reward_fn
