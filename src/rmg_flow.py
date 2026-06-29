"""
RMG's Riemannian Flow Matching (velocity) on M = R^3 (translation) x (S^3)^22 (joint quaternions).

Training (paper eq. 4): sample x0~prior, x1~data, t~U(0,1); interpolate on geodesics
  trans_t = (1-t) trans0 + t trans1 ;  quat_t = slerp(quat0, quat1, t)
target conditional velocity v_t = Log_{x_t}(x1)/(1-t) (translation reduces to trans1-trans0); the network
output is projected to the tangent space; loss = masked MSE over the product tangent.

Sampling (eq. 5): x0~prior, manifold Euler  trans += h v_trans ;  quat = Exp_quat(h * proj(v_quat)).
Classifier-free guidance combines conditional/unconditional velocities.
"""
import torch
import torch.nn.functional as F

import s3

J = 22


class RMGFlow:
    def __init__(self, sigma_trans=1.0, sigma_rot=1.0):
        self.sigma_trans = sigma_trans
        self.sigma_rot = sigma_rot

    def prior(self, B, L, device, dtype=torch.float32):
        trans = self.sigma_trans * torch.randn(B, L, 3, device=device, dtype=dtype)
        quats = s3.wrapped_gaussian((B, L, J), self.sigma_rot, device=device, dtype=dtype)
        return trans, quats

    @staticmethod
    def pack(trans, quats):
        B, L = trans.shape[:2]
        return torch.cat([trans, quats.reshape(B, L, J * 4)], dim=-1)        # (B,L,91)

    @staticmethod
    def unpack(x):
        B, L = x.shape[:2]
        return x[..., :3], x[..., 3:].reshape(B, L, J, 4)

    def training_loss(self, model, trans1, quats1, text=None, mask=None, p_drop=0.1):
        B, L = trans1.shape[:2]
        dev, dt = trans1.device, trans1.dtype
        trans0, quats0 = self.prior(B, L, dev, dt)
        t = torch.rand(B, device=dev, dtype=dt).clamp(0.0, 0.999)
        tt = t[:, None, None]
        trans_t = (1 - tt) * trans0 + tt * trans1
        quat_t = s3.geodesic(quats0, quats1, t[:, None, None, None])         # (B,L,J,4)

        trans_tgt = (trans1 - trans_t) / (1 - tt)                            # == trans1 - trans0
        quat_tgt = s3.log(quat_t, quats1) / (1 - t)[:, None, None, None]     # tangent at quat_t

        drop_mask = (torch.rand(B, device=dev) < p_drop) if text is not None else None
        pred = model(self.pack(trans_t, quat_t), t, text=text, mask=mask, drop_mask=drop_mask)
        ptrans, pquat = self.unpack(pred)
        pquat = s3.proj(quat_t, pquat)                                       # project to tangent

        tgt = self.pack(trans_tgt, quat_tgt)
        prd = self.pack(ptrans, pquat)
        se = (prd - tgt) ** 2                                                # (B,L,91)
        if mask is not None:
            m = mask[..., None].float()
            return (se * m).sum() / (m.sum() * se.shape[-1]).clamp_min(1.0)
        return se.mean()

    @torch.no_grad()
    def sample(self, model, B, L, mask=None, text=None, guidance=1.0, n_steps=50, device="cuda",
               dtype=torch.float32, return_traj=False):
        trans, quats = self.prior(B, L, device, dtype)
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)
        traj = []
        h = 1.0 / n_steps
        do_cfg = text is not None and guidance != 1.0
        for k in range(n_steps):
            tv = torch.full((B,), k * h, device=device, dtype=dtype)
            x = self.pack(trans, quats)
            if do_cfg:
                pc = model(x, tv, text=text, mask=mask)
                pu = model(x, tv, text=None, mask=mask)
                pred = pu + guidance * (pc - pu)
            else:
                pred = model(x, tv, text=text, mask=mask)
            ptrans, pquat = self.unpack(pred)
            trans = trans + h * ptrans
            quats = s3.exp(quats, h * s3.proj(quats, pquat))
            if return_traj:
                traj.append((trans.clone(), quats.clone()))
        return (trans, quats, traj) if return_traj else (trans, quats)
