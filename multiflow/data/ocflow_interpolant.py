import torch
import numpy as np
from torch import autograd
from torch.distributions import Categorical
from multiflow.data.interpolant import Interpolant, _centered_gaussian, _uniform_so3, _masked_categorical
from multiflow.data import utils as du
from multiflow.data import all_atom

# ======================================================================
# 📊 诊断记录器 (Guidance Logger)
# ======================================================================
class GuidanceLogger:
    def __init__(self):
        self.history = {
            # --- 步级别 (Step-level) 记录 ---
            'iter': [],             # 记录当前是第几次迭代
            't': [],                # 记录时间 t
            'reward': [],           # 记录当前结构的 reward
            'grad_l2_norm': [],     # 记录梯度的 L2 范数
            'grad_max_norm': [],    # 记录梯度的 Max 范数
            'vf_update_norm': [],   # 记录主模型预测的更新幅度
            'guidance_update_norm': [], # 记录 guidance (theta) 的更新幅度
            'grad_spatial': [],     # 记录残基维度的梯度空间分布
            # --- 轮次级别 (Iteration-level) 记录 ---
            'iter_idx': [],         # 记录第几轮 [0, 1, 2, 3...]
            'iter_final_reward': [] # 记录该轮生成结束时的确切 Reward
        }

    def log_step(self, iter_idx, t, reward, grad, vf_update, guidance_update):
        self.history['iter'].append(iter_idx)
        self.history['t'].append(t.item() if isinstance(t, torch.Tensor) else t)
        self.history['reward'].append(reward)
        self.history['grad_l2_norm'].append(grad.norm(p=2, dim=-1).mean().item())
        self.history['grad_max_norm'].append(grad.abs().max().item())
        self.history['vf_update_norm'].append(vf_update.norm(p=2, dim=-1).mean().item())
        self.history['guidance_update_norm'].append(guidance_update.norm(p=2, dim=-1).mean().item())
        spatial_grad = grad.norm(p=2, dim=-1).mean(dim=0)
        self.history['grad_spatial'].append(spatial_grad.detach().cpu().numpy())

    def log_iteration(self, iter_idx, final_reward):
        self.history['iter_idx'].append(iter_idx)
        self.history['iter_final_reward'].append(final_reward)

    def save(self, save_path="guidance_diagnostics.npz"):
        np.savez(
            save_path,
            iter=np.array(self.history['iter']),
            t=np.array(self.history['t']),
            reward=np.array(self.history['reward']),
            grad_l2_norm=np.array(self.history['grad_l2_norm']),
            grad_max_norm=np.array(self.history['grad_max_norm']),
            vf_update_norm=np.array(self.history['vf_update_norm']),
            guidance_update_norm=np.array(self.history['guidance_update_norm']),
            grad_spatial=np.array(self.history['grad_spatial']),
            iter_idx=np.array(self.history['iter_idx']),
            iter_final_reward=np.array(self.history['iter_final_reward'])
        )
        print(f"✅ Logger saved successfully to {save_path}")


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

# ======================================================================
# 🚨 官方 Trick：极其暴力的步级平方和截断函数（防止 ODE 链式求导梯度爆炸）
# ======================================================================
def clip_norm(x, max_norm):
    norm = x.square().sum(dim=list(range(1, x.ndim)), keepdim=True)
    cond = norm > max_norm
    scale = max_norm / (norm + 1e-6)
    return torch.where(cond, x * scale, x)


class GuidedInterpolant(Interpolant):
    """
    OC-Flow Guided Interpolant (terminal reward, VJP)
    """
    def __init__(self, cfg, guidance_config=None, reward_fn=None, **kwargs):
        super().__init__(cfg)

        self.guidance_cfg = None
        if guidance_config is not None:
            self.guidance_cfg = guidance_config
        elif hasattr(cfg, "guidance") and cfg.guidance is not None:
            from omegaconf import OmegaConf
            self.guidance_cfg = OmegaConf.to_container(cfg.guidance, resolve=True)

        self.num_guidance_iters = 1
        self.step_size = 0.01
        self.momentum = 0.9
        self.seq_samples = 8
        self.seed = None

        if isinstance(self.guidance_cfg, dict):
            enabled = self.guidance_cfg.get("enabled", False)
            if enabled:
                self.num_guidance_iters = self.guidance_cfg.get("num_iters", 1)
                self.step_size = self.guidance_cfg.get("step_size", 0.01)
                self.momentum = self.guidance_cfg.get("momentum", 0.9)
                self.seq_samples = self.guidance_cfg.get("seq_samples", 8)
                self.seed = self.guidance_cfg.get("seed", None)

        self.reward_fn = reward_fn

    def _compute_reward(self, logits, backbone):
        # 修复3：使用 argmax 消除采样随机性导致的目标跳变 (Moving Target)
        # 取概率最大的序列，并展开为 (seq_samples, B, L) 适配 MPNN 接口
        seq_samples = torch.argmax(logits, dim=-1).unsqueeze(0).expand(self.seq_samples, -1, -1)
        # compute MPNN reward
        scores = self.reward_fn(seq_samples, backbone)
        # 取均值 [B]
        reward = scores.mean(dim=0)
        # NaN protection
        if torch.isnan(reward).any():
            reward = torch.nan_to_num(reward, nan=0.0)
        return reward

    def sample(
        self,
        num_batch,
        num_res,
        model,
        num_timesteps=None,
        trans_0=None,
        rotmats_0=None,
        aatypes_0=None,
        trans_1=None,
        rotmats_1=None,
        aatypes_1=None,
        diffuse_mask=None,
        chain_idx=None,
        res_idx=None,
        t_nn=None,
        forward_folding=False,
        inverse_folding=False,
        separate_t=False,
        reward_model=None,
        step_size=0.05,
        log_save_dir=None,
    ):
        print("DEBUG: sample() called")
        device = self._device
        res_mask = torch.ones(num_batch, num_res, device=device)

        if diffuse_mask is None:
            diffuse_mask = res_mask
        if res_idx is None:
            res_idx = torch.arange(num_res, device=device)[None].repeat(num_batch, 1)
        if chain_idx is None:
            chain_idx = res_mask

        # ===== initial noise =====
        if trans_0 is None:
            trans_0 = _centered_gaussian(num_batch, num_res, device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, device)
        if aatypes_0 is None:
            if self._aatypes_cfg.interpolant_type == "masking":
                aatypes_0 = _masked_categorical(num_batch, num_res, device)
            else:
                aatypes_0 = torch.randint_like(res_mask, low=0, high=self.num_tokens)

        trans_t, rotmats_t, aatypes_t = trans_0, rotmats_0, aatypes_0

        if aatypes_1 is not None:
            logits_1 = torch.nn.functional.one_hot(aatypes_1, num_classes=self.num_tokens).float()

        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'trans_sc': torch.zeros(num_batch, num_res, 3, device=device),
            'aatypes_sc': torch.zeros(num_batch, num_res, self.num_tokens, device=device),
        }

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, res_mask).detach().cpu()

        # ==========================================
        # 🚨 优化核心：Asynchronous + VJP Adjoint Method
        # ==========================================
        num_iters = getattr(self, 'num_guidance_iters', 5)
        eta = getattr(self, 'step_size', 0.1)      
        beta = getattr(self, 'momentum', 0.9)      
        num_steps = len(ts) - 1
       
        group_size = 5  
        num_thetas = (num_steps + group_size - 1) // group_size
       
        theta_trans = [torch.zeros_like(trans_0) for _ in range(num_thetas)]
        prot_traj, clean_traj = [], []
        
        # 记录最优状态的变量
        best_reward = -float('inf')
        best_prot_traj, best_clean_traj = [], []

        logger = GuidanceLogger()

        for param in model.parameters():
            param.requires_grad_(False)

        for k in range(num_iters):
            states = []
            trans_t = trans_0.detach().clone()
            rotmats_t = rotmats_0.detach().clone()
            aatypes_t = aatypes_0.detach().clone()
            
            t_prev = ts[0]
            
            # ✨ 新增 1：每次迭代开始，初始化记录轨迹的列表，并放入初始噪声帧
            current_prot_traj = [(frames_to_atom37(trans_t, rotmats_t), aatypes_t.detach().cpu())]
            current_clean_traj = []
            
            # =======================================================
            # 阶段 1：前向推断 (完全 no_grad，极大节省显存)
            # =======================================================
            with torch.no_grad():
                for step_idx, t_next in enumerate(ts[1:]):
                    states.append({
                        'trans': trans_t.clone(),
                        'rotmats': rotmats_t.clone(),
                        'aatypes': aatypes_t.clone(),
                        't_prev': t_prev,
                        't_next': t_next
                    })
                    
                    dt = t_next - t_prev
                    
                    batch['trans_t'] = trans_t
                    batch['rotmats_t'] = rotmats_t
                    batch['aatypes_t'] = aatypes_t
                    t_tensor = torch.ones((num_batch,1), device=device) * t_prev
                    batch['r3_t'], batch['so3_t'], batch['cat_t'] = t_tensor, t_tensor, t_tensor
                    
                    model_out = model(batch)
                    pred_trans = model_out['pred_trans']
                    pred_rotmats = model_out['pred_rotmats']
                    pred_logits = model_out['pred_logits']

                    if forward_folding:
                        pred_logits = 100.0 * logits_1
                    
                    # ✨ 新增 2：记录每一小步预测的“干净轨迹” (Clean Trajectory)
                    pred_aatypes = torch.argmax(pred_logits, dim=-1)
                    current_clean_traj.append((frames_to_atom37(pred_trans, pred_rotmats), pred_aatypes.detach().cpu()))
                    
                    next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, trans_t)
                    rotmats_t = self._rots_euler_step(dt, t_prev, pred_rotmats, rotmats_t)
                    aatypes_t = self._aatypes_euler_step(dt, t_prev, pred_logits, aatypes_t)
                    
                    theta_idx = step_idx // group_size
                    next_trans_t = next_trans_t + theta_trans[theta_idx]
                    
                    next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)
                    rotmats_t = _rots_diffuse_mask(rotmats_t, pred_rotmats, diffuse_mask)
                    
                    trans_t = next_trans_t
                    t_prev = t_next

                    # ✨ 新增 3：记录每一小步走完 Euler step 后的“带噪轨迹” (Noisy Trajectory)
                    current_prot_traj.append((frames_to_atom37(trans_t, rotmats_t), aatypes_t.detach().cpu()))
            
            states.append({
                'trans': trans_t.clone(),
                'rotmats': rotmats_t.clone(),
                'aatypes': aatypes_t.clone()
            })
            
            # =======================================================
            # 阶段 2：计算 Reward 和 终点梯度 \nabla \Phi
            # =======================================================
            final_trans = states[-1]['trans'].detach().requires_grad_(True)
            batch['trans_t'] = final_trans
            batch['rotmats_t'] = states[-1]['rotmats']
            batch['aatypes_t'] = states[-1]['aatypes']
            
            with torch.enable_grad():
                model_out_final = model(batch)
                pred_trans_final = model_out_final['pred_trans']
                pred_logits_final = model_out_final['pred_logits']
                
                reward = self._compute_reward(pred_logits_final, final_trans)
                
                if k < num_iters - 1:
                    grad_x = autograd.grad(reward.sum(), final_trans)[0]
                    
            # ✨ 修复 6：让所有进程都计算 det_reward（因为各卡数据不同），解决多卡 DDP 变量未定义报错
            with torch.no_grad():
                det_seq = torch.argmax(pred_logits_final, dim=-1).unsqueeze(0)
                det_reward = self.reward_fn(det_seq, pred_trans_final.detach()).mean()
            
            # 📺 只让 rank 0 主进程负责打印日志，避免屏幕被刷屏
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if k == 0:
                    print(f"\n[Baseline] 无引导 (Zero-shot) 初始 Reward: {det_reward.item():.4f}", flush=True)
                else:
                    print(f"\n[OC-Flow] Iteration: {k}/{num_iters-1} | Sampled R: {reward.mean().item():.4f} | Det R: {det_reward.item():.4f}", flush=True)
            
            # ✨ 修复 4：用完整的 current_traj 更新 best_traj
            current_reward = det_reward.item() 
            
            # 📊 记录该轮迭代的最终 Reward 供收敛分析使用
            logger.log_iteration(k, current_reward)

            if k == 0 or current_reward > best_reward:
                best_reward = current_reward
                
                # 获取最后一步的结果并补齐
                pred_atom37 = frames_to_atom37(pred_trans_final, model_out_final['pred_rotmats'])
                pred_aatypes_final = torch.argmax(pred_logits_final, dim=-1).detach().cpu()
                
                best_clean_traj = current_clean_traj + [(pred_atom37, pred_aatypes_final)]
                best_prot_traj = current_prot_traj + [(pred_atom37, pred_aatypes_final)]
                
                # 📺 同样，发现更优结构也只在 rank 0 打印
                if k > 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    print(f"  --> [OC-Flow] 发现更优结构！Best Reward 刷新为: {best_reward:.4f}")

            # =======================================================
            # 阶段 3：严格对齐官方的 Vector-Jacobian Product (VJP)
            # =======================================================
            grad_thetas = [torch.zeros_like(th) for th in theta_trans]
            max_grad_norm = 5.0
            
            with torch.enable_grad():
                # 安全判断：最后一次迭代不需要推导梯度
                if k < num_iters - 1:
                    for step_idx in reversed(range(num_steps)):
                        curr_state = states[step_idx]
                        t_prev = curr_state['t_prev']
                        t_next = curr_state['t_next']
                        dt = t_next - t_prev
                        
                        curr_trans = curr_state['trans'].detach().requires_grad_(True)
                        curr_rotmats = curr_state['rotmats'].detach()
                        curr_aatypes = curr_state['aatypes'].detach()
                        
                        theta_idx = step_idx // group_size
                        curr_theta = theta_trans[theta_idx].detach()
                        
                        batch['trans_t'] = curr_trans
                        batch['rotmats_t'] = curr_rotmats
                        batch['aatypes_t'] = curr_aatypes
                        t_tensor = torch.ones((num_batch,1), device=device) * t_prev
                        batch['r3_t'], batch['so3_t'], batch['cat_t'] = t_tensor, t_tensor, t_tensor
                        
                        model_out = model(batch)
                        pred_trans = model_out['pred_trans']

                        # 拆解更新量，用于记录无条件向量场的步幅
                        vf_next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, curr_trans)
                        vf_update = vf_next_trans_t - curr_trans
                        
                        next_trans_t = vf_next_trans_t + curr_theta 
                        next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)
                        
                        vjp_out = autograd.grad(
                            outputs=next_trans_t,
                            inputs=curr_trans,
                            grad_outputs=grad_x
                        )
                        
                        lam = vjp_out[0]
                        
                        # 📊 记录微观梯度的各项指标
                        logger.log_step(
                            iter_idx=k,
                            t=t_prev,
                            reward=current_reward, # 记录该整条轨迹对应的 reward
                            grad=lam.detach(),
                            vf_update=vf_update.detach(),
                            guidance_update=curr_theta.detach()
                        )
                        
                        lam = clip_norm(lam, max_grad_norm)
                        
                        grad_thetas[theta_idx] = grad_thetas[theta_idx] + lam
                        grad_x = lam
                        #print("grad norm:", grad_x.norm())
                    
            # =======================================================
            # 阶段 4：更新全局控制项 \theta
            # =======================================================
            for i in range(num_thetas):
                lam_i = torch.nan_to_num(grad_thetas[i]) / group_size
                theta_trans[i] = (beta * theta_trans[i] + eta * lam_i).detach()
                
        # =======================================================
        # 👉 [修改处 4]：极简保存，直接使用外部计算好的绝对路径
        # =======================================================
        if log_save_dir is not None:
            import os
            os.makedirs(log_save_dir, exist_ok=True)
            save_path = os.path.join(log_save_dir, "guidance_diagnostics.npz")
            logger.save(save_path)
            print(f"\n✅ [Logger] 诊断数据已精准存入当前样本目录: {save_path}\n")

        # ✨ 修复 5：返回保存的带有完整 100 步轨迹的最佳结果
        return best_prot_traj, best_clean_traj