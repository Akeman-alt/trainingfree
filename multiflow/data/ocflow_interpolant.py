import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import autograd

from multiflow.data.interpolant import (
    Interpolant,
    _centered_gaussian,
    _uniform_so3,
    _masked_categorical,
)

from multiflow.data import utils as du
from multiflow.data import all_atom
from multiflow.rewards import MPNNReward


class GuidedInterpolant(Interpolant):
    """
    Guided version of Interpolant using Optimal Control guidance (OC-Flow).
    Ref: Training Free Guided Flow Matching with Optimal Control (ICLR 2025)
    """

    def __init__(self, cfg, guidance_config=None, reward_fn=None, **kwargs):
        super().__init__(cfg)

        # 1. 统一解析 guidance 配置
        self.guidance_cfg = None
        if guidance_config is not None:
            self.guidance_cfg = guidance_config
        elif hasattr(cfg, "guidance") and cfg.guidance is not None:
            from omegaconf import OmegaConf
            self.guidance_cfg = OmegaConf.to_container(cfg.guidance, resolve=True)

        # 2. 安全的默认值初始化
        self.num_guidance_iters = 1
        self.step_size = 0.01      # 对应论文中的 learning rate \eta
        self.momentum = 0.9       # 对应论文中的 weight decay \beta
        self.seq_samples = 32
        self.seed = None
        
        # 3. 如果开启了 guidance，则覆盖默认值
        is_enabled = False
        if isinstance(self.guidance_cfg, dict):
            is_enabled = self.guidance_cfg.get('enabled', False)

        if is_enabled:
            self.num_guidance_iters = self.guidance_cfg.get('num_iters', 1)
            self.step_size = self.guidance_cfg.get('step_size', 0.01)
            self.momentum = self.guidance_cfg.get('momentum', 0.9)
            self.seq_samples = self.guidance_cfg.get('seq_samples', 32)
            self.seed = self.guidance_cfg.get('seed', None)

        self.mpnn_reward = None

    def _compute_reward(self, logits, backbone):
        """
        Monte Carlo estimation of E_seq [ log P_MPNN(seq | backbone) ]
        """
        dist = Categorical(logits=logits)
        seq_samples = dist.sample((self.seq_samples,))
        
        scores = []
        for seq in seq_samples:
            score = self.mpnn_reward(seq, backbone)
            scores.append(score)
            
        scores = torch.stack(scores)
        reward = scores.mean(dim=0)
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
            **kwargs
        ):
        device = self._device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        res_mask = torch.ones(num_batch, num_res, device=device)
        if trans_0 is None:
            trans_0 = _centered_gaussian(num_batch, num_res, device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, device)
        if aatypes_0 is None:
            aatypes_0 = _masked_categorical(num_batch, num_res, device)
        if diffuse_mask is None:
            diffuse_mask = res_mask
        if res_idx is None:
            res_idx = torch.arange(num_res, device=device, dtype=torch.float32)[None].repeat(num_batch, 1)
        if chain_idx is None:
            chain_idx = res_mask
        
        if self.mpnn_reward is None:
            self.mpnn_reward = MPNNReward(device)

        batch = {
            "res_mask": res_mask,
            "diffuse_mask": diffuse_mask,
            "chain_idx": chain_idx,
            "res_idx": res_idx,
            "trans_sc": torch.zeros(num_batch, num_res, 3, device=device),
            "aatypes_sc": torch.zeros(num_batch, num_res, self.num_tokens, device=device),
        }

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)
        frames_to_atom37 = lambda x, y: all_atom.atom37_from_trans_rot(x, y, res_mask).detach().cpu()

        # 初始化控制变量 Theta (不再需要全局 requires_grad，通过手动 VJP 计算)
        theta = torch.zeros(num_timesteps, num_batch, num_res, 3, device=device)

        for guidance_iter in range(self.num_guidance_iters):
            
            # ==========================================================
            # 阶段 1：无梯度正向采样 (Phase 1: Gradient-free Forward Pass)
            # 通过切断计算图，大幅降低显存占用 O(ND^2) -> O(D^2)
            # ==========================================================
            trans_t = trans_0.clone()
            rotmats_t = rotmats_0.clone()
            aatypes_t = aatypes_0.clone()

            saved_states = [] # 记录状态，供反向 Adjoint VJP 使用
            prot_traj = []
            clean_traj = []
            t_prev = ts[0]

            with torch.no_grad():
                for step, t in enumerate(ts[1:]):
                    saved_states.append({
                        "trans": trans_t.clone(),
                        "rotmats": rotmats_t.clone(),
                        "aatypes": aatypes_t.clone(),
                        "trans_sc": batch["trans_sc"].clone(),
                        "aatypes_sc": batch["aatypes_sc"].clone(),
                    })

                    batch["trans_t"] = trans_t
                    batch["rotmats_t"] = rotmats_t
                    batch["aatypes_t"] = aatypes_t

                    t_tensor = torch.ones(num_batch, 1, device=device) * t_prev
                    batch["r3_t"] = t_tensor; batch["so3_t"] = t_tensor; batch["cat_t"] = t_tensor

                    dt = t - t_prev
                    model_out = model(batch)


                    pred_trans = model_out["pred_trans"]
                    pred_rotmats = model_out["pred_rotmats"]
                    pred_logits = model_out["pred_logits"]
                    pred_aatypes = model_out["pred_aatypes"]  # <--- 获取预测的类别

                    # 🚨 补充修复：记录模型预测的 clean 轨迹，防止下游 stack 报错空列表
                    clean_traj.append((
                        frames_to_atom37(pred_trans.cpu(), pred_rotmats.cpu()), 
                        pred_aatypes.detach().cpu()
                    ))

                    # Additive Control Rule
                    velocity = pred_trans - trans_t + theta[step]


                    trans_t = trans_t + velocity * dt

                    rotmats_t = self._rots_euler_step(dt, t_prev, pred_rotmats, rotmats_t)
                    aatypes_t = self._aatypes_euler_step(dt, t_prev, pred_logits, aatypes_t)

                    # Self-Conditioning Update
                    if getattr(self._cfg, 'self_condition', False):
                        batch['trans_sc'] = pred_trans.detach() * diffuse_mask[..., None]
                        if hasattr(self, '_aatypes_cfg'):
                            batch['aatypes_sc'] = torch.nn.functional.one_hot(
                                model_out['pred_aatypes'], num_classes=self.num_tokens
                            ).float() * diffuse_mask[..., None]

                    prot_traj.append((frames_to_atom37(trans_t, rotmats_t), aatypes_t.detach().cpu()))
                    t_prev = t

            # ==========================================================
            # 阶段 2：计算终端 Co-state \mu_T (Phase 2: Terminal Adjoint)
            # ==========================================================
            grad_theta = torch.zeros_like(theta)
            last_step = len(ts) - 2
            t_prev = ts[last_step]
            t = ts[last_step + 1]
            dt = t - t_prev
            
            final_state = saved_states[last_step]
            trans_t_req = final_state["trans"].requires_grad_(True)
            theta_step_req = theta[last_step].clone().requires_grad_(True)
            
            batch["trans_t"] = trans_t_req
            batch["rotmats_t"] = final_state["rotmats"]
            batch["aatypes_t"] = final_state["aatypes"]
            batch["trans_sc"] = final_state["trans_sc"]
            batch["aatypes_sc"] = final_state["aatypes_sc"]
            t_tensor = torch.ones(num_batch, 1, device=device) * t_prev
            batch["r3_t"] = t_tensor; batch["so3_t"] = t_tensor; batch["cat_t"] = t_tensor
            
            with torch.enable_grad():
                model_out = model(batch)
                velocity = model_out["pred_trans"] - trans_t_req + theta_step_req
                trans_next = trans_t_req + velocity * dt
                
                reward = self._compute_reward(model_out["pred_logits"], trans_next)
                
                # 计算出 \mu_T 以及终端 theta 的偏导
                grads = autograd.grad(reward.sum(), (trans_t_req, theta_step_req))
                grad_x = grads[0]
                grad_theta[last_step] = grads[1].detach()

            # ==========================================================
            # 阶段 3：随时间反向 VJP (Phase 3: Backprop via VJP)
            # 递推公式 \mu_t = \mu_{t+1} * Jacobian 
            # ==========================================================
            for step in reversed(range(last_step)):
                t_prev = ts[step]
                t = ts[step + 1]
                dt = t - t_prev
                
                state = saved_states[step]
                trans_t_req = state["trans"].requires_grad_(True)
                theta_step_req = theta[step].clone().requires_grad_(True)
                
                batch["trans_t"] = trans_t_req
                batch["rotmats_t"] = state["rotmats"]
                batch["aatypes_t"] = state["aatypes"]
                batch["trans_sc"] = state["trans_sc"]
                batch["aatypes_sc"] = state["aatypes_sc"]
                t_tensor = torch.ones(num_batch, 1, device=device) * t_prev
                batch["r3_t"] = t_tensor; batch["so3_t"] = t_tensor; batch["cat_t"] = t_tensor
                
                with torch.enable_grad():
                    model_out = model(batch)
                    velocity = model_out["pred_trans"] - trans_t_req + theta_step_req
                    trans_next = trans_t_req + velocity * dt
                    
                    # 使用 grad_outputs (即\mu_{t+1}) 计算雅可比乘积，更新为\mu_t
                    grads = autograd.grad(trans_next, (trans_t_req, theta_step_req), grad_outputs=grad_x)
                    grad_x = grads[0]
                    grad_theta[step] = grads[1].detach()

            # ==========================================================
            # 阶段 4：E-MSA 最优控制更新 (Phase 4: OC-Flow E-MSA Update)
            # 对应论文 Eq (19) Update rule
            # ==========================================================
            with torch.no_grad():
                # \theta_t^{k+1} = \beta * \theta_t^k + \eta * \mu_t
                theta = self.momentum * theta + self.step_size * grad_theta
            
            print(f"Guidance Iter {guidance_iter+1}/{self.num_guidance_iters} | Reward: {reward.mean().item():.4f}")

        return prot_traj, clean_traj