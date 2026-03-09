import torch

from torch import autograd

from torch.distributions import Categorical

from multiflow.data.interpolant import Interpolant, _centered_gaussian, _uniform_so3, _masked_categorical

from multiflow.data import utils as du

from multiflow.data import all_atom





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

        # sample sequences (no gradient)

        dist = Categorical(logits=logits)

        seq_samples = dist.sample((self.seq_samples,)).detach()



        # compute MPNN reward

        scores = self.reward_fn(seq_samples, backbone)



        # 取多次采样的平均分数 [B]

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

        eta = getattr(self, 'step_size', 0.9)      

        beta = getattr(self, 'momentum', 1.2)      

        num_steps = len(ts) - 1

       

        group_size = 5  

        num_thetas = (num_steps + group_size - 1) // group_size

       

        theta_trans = [torch.zeros_like(trans_0) for _ in range(num_thetas)]

        prot_traj, clean_traj = [], []

       

        for param in model.parameters():

            param.requires_grad_(False)



        for k in range(num_iters):

            states = []

            trans_t = trans_0.detach().clone()

            rotmats_t = rotmats_0.detach().clone()

            aatypes_t = aatypes_0.detach().clone()

           

            t_prev = ts[0]

           

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

                   

                    next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, trans_t)

                    rotmats_t = self._rots_euler_step(dt, t_prev, pred_rotmats, rotmats_t)

                    aatypes_t = self._aatypes_euler_step(dt, t_prev, pred_logits, aatypes_t)

                   

                    theta_idx = step_idx // group_size

                    next_trans_t = next_trans_t + theta_trans[theta_idx] * dt

                   

                    next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)

                    rotmats_t = _rots_diffuse_mask(rotmats_t, pred_rotmats, diffuse_mask)

                   

                    trans_t = next_trans_t

                    t_prev = t_next

           

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

               

                reward = self._compute_reward(pred_logits_final, pred_trans_final)

               

                if k < num_iters - 1:

                    grad_x = autograd.grad(reward.sum(), final_trans)[0]

                   

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

                with torch.no_grad():

                    det_seq = torch.argmax(pred_logits_final, dim=-1).unsqueeze(0)

                    det_reward = self.reward_fn(det_seq, pred_trans_final.detach()).mean()

               

                print(f"\n[OC-Flow] Iteration: {k+1}/{num_iters} | Sampled R: {reward.mean().item():.4f} | Det R: {det_reward.item():.4f}", flush=True)

           

            if k == num_iters - 1:

                pred_atom37 = frames_to_atom37(pred_trans_final, model_out_final['pred_rotmats'])

                clean_traj.append((pred_atom37, model_out_final['pred_aatypes'].detach().cpu()))

                prot_traj.append((pred_atom37, model_out_final['pred_aatypes'].detach().cpu()))

                break

               

            # =======================================================

            # 阶段 3：严格对齐官方的 Vector-Jacobian Product (VJP)

            # =======================================================

            grad_thetas = [torch.zeros_like(th) for th in theta_trans]

            max_grad_norm = 5.0

           

            with torch.enable_grad():

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

                   

                    next_trans_t = self._trans_euler_step(dt, t_prev, pred_trans, curr_trans)

                    next_trans_t = next_trans_t + curr_theta * dt

                    next_trans_t = _trans_diffuse_mask(next_trans_t, pred_trans, diffuse_mask)

                   

                    # 🚨 核心修复：只对状态 curr_trans 求导，获取精确的共态变量

                    vjp_out = autograd.grad(

                        outputs=next_trans_t,

                        inputs=curr_trans,

                        grad_outputs=grad_x

                    )

                   

                    # 🚨 核心修复：逐步施加极其严格的平方和惩罚截断

                    lam = vjp_out[0]

                    lam = clip_norm(lam, max_grad_norm)

                   

                    grad_thetas[theta_idx] = grad_thetas[theta_idx] + lam

                    grad_x = lam

                   

            # =======================================================

            # 阶段 4：更新全局控制项 \theta

            # =======================================================

            for i in range(num_thetas):

                lam_i = torch.nan_to_num(grad_thetas[i]) / group_size

                # 应用 \theta 动量更新公式（Gradient Ascent）

                theta_trans[i] = (beta * theta_trans[i] + eta * lam_i).detach()



        return prot_traj, clean_traj
 