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

        reward = scores.mean(dim=0)  # [B]
        # normalization
        reward = reward - reward.mean()
        reward = reward / (reward.std() + 1e-6)

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

        prot_traj, clean_traj = [], []
        t_prev = ts[0]

        # ==============================
        # Forward trajectory
        # ==============================
        for t_next in ts[1:]:
            batch['trans_t'], batch['rotmats_t'], batch['aatypes_t'] = trans_t, rotmats_t, aatypes_t
            t_tensor = torch.ones((num_batch,1), device=device) * t_prev
            batch['r3_t'], batch['so3_t'], batch['cat_t'] = t_tensor, t_tensor, t_tensor
            dt = t_next - t_prev

            with torch.no_grad():
                model_out = model(batch)
            pred_trans = model_out['pred_trans']
            pred_rotmats = model_out['pred_rotmats']
            pred_logits = model_out['pred_logits']
            pred_aatypes = model_out['pred_aatypes']

            clean_traj.append((frames_to_atom37(pred_trans, pred_rotmats), pred_aatypes.detach().cpu()))

            # Euler integrator
            trans_t = self._trans_euler_step(dt, t_prev, pred_trans, trans_t)
            rotmats_t = self._rots_euler_step(dt, t_prev, pred_rotmats, rotmats_t)
            aatypes_t = self._aatypes_euler_step(dt, t_prev, pred_logits, aatypes_t)

            # diffuse mask
            trans_t = _trans_diffuse_mask(trans_t, pred_trans, diffuse_mask)
            rotmats_t = _rots_diffuse_mask(rotmats_t, pred_rotmats, diffuse_mask)

            prot_traj.append((frames_to_atom37(trans_t, rotmats_t), aatypes_t.detach().cpu()))
            t_prev = t_next

        # ==============================
        # Final reward guidance
        # ==============================
        batch['trans_t'], batch['rotmats_t'], batch['aatypes_t'] = trans_t, rotmats_t, aatypes_t
        # model forward (no_grad)
        with torch.no_grad():
            model_out = model(batch)

        pred_trans, pred_rotmats, pred_aatypes = model_out['pred_trans'], model_out['pred_rotmats'], model_out['pred_aatypes']

        if self.reward_fn is not None:
            # 1. 记录引导前的 reward
            with torch.no_grad():
                # 🚨 直接调用 self.reward_fn，只传 1 个参数（即结构坐标），彻底绕开 _compute_reward
                r_before = self.reward_fn(pred_trans.detach())
                r_before = torch.nan_to_num(r_before)

            # 2. 解除 Lightning 的强制封锁，开启计算图
            with torch.inference_mode(False), torch.enable_grad():
                
                final_trans = pred_trans.detach().clone().requires_grad_(True)

                # 🚨 同样直接调用 self.reward_fn，完美匹配你的单参数模型
                reward = self.reward_fn(final_trans)
                reward = torch.nan_to_num(reward)

                grad = autograd.grad(reward.sum(), final_trans)[0]
                grad = torch.nan_to_num(grad)
                grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)

            # 3. 应用微调
            pred_trans = pred_trans + step_size * grad.detach()

            # 4. 记录微调后的 reward
            with torch.no_grad():
                r_after = self.reward_fn(pred_trans.detach())
                r_after = torch.nan_to_num(r_after)

            # Only print on rank 0 with flush
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"\nReward before: {r_before.mean().item():.4f} | after: {r_after.mean().item():.4f}", flush=True)
                
        pred_atom37 = frames_to_atom37(pred_trans, pred_rotmats)
        clean_traj.append((pred_atom37, pred_aatypes.detach().cpu()))
        prot_traj.append((pred_atom37, pred_aatypes.detach().cpu()))

        return prot_traj, clean_traj