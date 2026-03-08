import torch
from torch import autograd
from torch.distributions import Categorical
from multiflow.data.interpolant import Interpolant, _centered_gaussian, _uniform_so3, _masked_categorical
from multiflow.data import utils as du
from multiflow.data import all_atom
from multiflow.rewards import MPNNReward
def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

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
        # sample sequences (no gradient)
        dist = Categorical(logits=logits)
        seq_samples = dist.sample((self.seq_samples,)).detach()

        # compute MPNN reward
        scores = self.reward_fn(seq_samples, backbone)

        # 取多次采样的平均分数 [B]
        reward = scores.mean(dim=0)

        # 🚨 [关键修改]：删除了 reward = reward - reward.mean() 等归一化代码
        # 因为 B=1 时减去均值会直接变成 0。
        # 外面的 grad = grad / grad.norm() 已经起到了控制步长的作用。

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


        if self.reward_fn is None:
            self.reward_fn = MPNNReward(device)

        trans_t, rotmats_t, aatypes_t = trans_0, rotmats_0, aatypes_0

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
        num_iters = self.num_guidance_iters
        eta = self.step_size      # 学习率 \eta
        beta = self.momentum      # 动量衰减 \beta
        num_steps = len(ts) - 1
        
        group_size = 5  # 每 5 步共享同一个 \theta 
        num_thetas = (num_steps + group_size - 1) // group_size 
        
        # 初始化控制项 \theta (不再需要全局挂载梯度，我们手动维护)
        theta_trans = [torch.zeros_like(trans_0) for _ in range(num_thetas)]
        
        prot_traj, clean_traj = [], []
        
        # 确保模型权重是被冻结的，不参与意外的梯度计算占用显存
        for param in model.parameters():
            param.requires_grad_(False)

        # 2. 开启外层循环，进行多轮 OC-Flow 迭代
        for k in range(num_iters):
            states = [] # 用于极低内存开销保存每一帧的离散状态(路标)
            trans_t = trans_0.detach().clone()
            rotmats_t = rotmats_0.detach().clone()
            aatypes_t = aatypes_0.detach().clone()
            
            t_prev = ts[0]
            
            # =======================================================
            # 阶段 1：前向推断 (完全 no_grad，极大节省显存！)
            # =======================================================
            with torch.no_grad():
                for step_idx, t_next in enumerate(ts[1:]):
                    # 记录当前帧状态
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
                    
                    next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, trans_t)
                    rotmats_t = self._rots_euler_step(dt, t_prev, pred_rotmats, rotmats_t)
                    aatypes_t = self._aatypes_euler_step(dt, t_prev, pred_logits, aatypes_t)
                    
                    # 施加控制项
                    theta_idx = step_idx // group_size
                    next_trans_t = next_trans_t + theta_trans[theta_idx] * dt
                    
                    # 漫反射 mask
                    next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)
                    rotmats_t = _rots_diffuse_mask(rotmats_t, pred_rotmats, diffuse_mask)
                    
                    trans_t = next_trans_t
                    t_prev = t_next
            
            # 记录终点状态用于求导
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
                
                # 计算 Reward（适配你代码里的 reward_fn）
                reward = self.reward_fn(pred_trans_final)
                reward = torch.nan_to_num(reward)
                
                if k < num_iters - 1:
                    # 获取针对 final_trans 的初始梯度
                    grad_x = autograd.grad(reward.sum(), final_trans)[0]
                    
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"\n[OC-Flow] Iteration: {k+1}/{num_iters} | Reward: {reward.mean().item():.4f}", flush=True)

            if k == num_iters - 1:
                # 最后一轮直接收集生成好的轨迹并退出
                pred_atom37 = frames_to_atom37(pred_trans_final, model_out_final['pred_rotmats'])
                clean_traj.append((pred_atom37, model_out_final['pred_aatypes'].detach().cpu()))
                prot_traj.append((pred_atom37, model_out_final['pred_aatypes'].detach().cpu()))
                break
                
            # =======================================================
            # 阶段 3：Vector-Jacobian Product (VJP) 逆向时间反向传播！
            # =======================================================
            grad_thetas = [torch.zeros_like(th) for th in theta_trans]
            
            with torch.enable_grad():
                # 从倒数第一步往前走 (Step-by-step backward)
                for step_idx in reversed(range(num_steps)):
                    curr_state = states[step_idx]
                    t_prev = curr_state['t_prev']
                    t_next = curr_state['t_next']
                    dt = t_next - t_prev
                    
                    # 重新激活输入张量，准备局部建图
                    curr_trans = curr_state['trans'].detach().requires_grad_(True)
                    curr_rotmats = curr_state['rotmats'].detach()
                    curr_aatypes = curr_state['aatypes'].detach()
                    
                    theta_idx = step_idx // group_size
                    curr_theta = theta_trans[theta_idx].detach().requires_grad_(True)
                    
                    batch['trans_t'] = curr_trans
                    batch['rotmats_t'] = curr_rotmats
                    batch['aatypes_t'] = curr_aatypes
                    t_tensor = torch.ones((num_batch,1), device=device) * t_prev
                    batch['r3_t'], batch['so3_t'], batch['cat_t'] = t_tensor, t_tensor, t_tensor
                    
                    # 仅为这 1 步重建计算图
                    model_out = model(batch)
                    pred_trans = model_out['pred_trans']
                    
                    next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, curr_trans)
                    next_trans_t = next_trans_t + curr_theta * dt
                    next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)
                    
                    # 🚨 核心 VJP：用输出梯度 grad_x 向前推算输入梯度，算完瞬间释放图内存！
                    vjp_out = autograd.grad(
                        outputs=next_trans_t,
                        inputs=(curr_trans, curr_theta),
                        grad_outputs=grad_x
                    )
                    
                    grad_curr_trans, grad_curr_theta = vjp_out[0], vjp_out[1]
                    
                    # 累积本时间步对应 \theta 的梯度
                    grad_thetas[theta_idx] = grad_thetas[theta_idx] + grad_curr_theta
                    
                    # 把计算出的 x 的梯度传递给上一时间步
                    grad_x = grad_curr_trans
                    
            # =======================================================
            # 阶段 4：更新全局控制项 \theta
            # =======================================================
            for i in range(num_thetas):
                grad_t = torch.nan_to_num(grad_thetas[i])
                grad_t = grad_t / (grad_t.norm(dim=-1, keepdim=True) + 1e-6)
                # 应用 \theta 动量更新公式
                theta_trans[i] = (beta * theta_trans[i] + eta * grad_t).detach()

        return prot_traj, clean_traj