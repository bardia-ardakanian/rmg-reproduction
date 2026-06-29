"""
S^3 (unit-quaternion) geometry for RMG's Riemannian flow matching — pure PyTorch, batched over
arbitrary leading dims; points are (...,4) unit quaternions [w,x,y,z].

We model joint rotations as unit quaternions on the hypersphere S^3 (per the RMG paper). Quaternions are
kept on the upper hemisphere (w >= 0) so q and -q (the same rotation) are unambiguous. A rotation by
angle alpha maps to a quaternion at S^3-angle alpha/2 from identity, so for valid rotations (alpha<=pi)
the geodesic angle theta <= pi/2 — antipodal degeneracy does not arise in-distribution.

Maps (T_x S^3 = { v : <v,x> = 0 }):
  proj(x,v)      tangent projection v - <v,x> x
  exp(x,v)       Exp_x(v) = cos|v| x + sinc|v| v
  log(x,y)       Log_x(y) = theta * (y - <x,y>x)/||.||,  ||log|| = dist = arccos<x,y>
  geodesic(...)  slerp (matches the paper's gamma(t))

Run `python s3.py` for the self-tests.
"""
import torch

EPS = 1e-7


def canonical(q):
    """Flip to the upper hemisphere (w >= 0); q and -q are the same rotation."""
    s = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * s


def normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)


def proj(x, v):
    """Project v onto the tangent space at x."""
    return v - (x * v).sum(-1, keepdim=True) * x


def exp(x, v):
    """Exp_x(v): exponential map at x of tangent vector v (projected for safety)."""
    v = proj(x, v)
    n = v.norm(dim=-1, keepdim=True)
    small = n < 1e-6
    sinc = torch.where(small, 1 - n * n / 6.0, torch.sin(n) / n.clamp_min(1e-9))
    out = torch.cos(n) * x + sinc * v
    return normalize(out)


def log(x, y):
    """Log_x(y): tangent vector at x pointing to y, with ||Log_x(y)|| = geodesic distance.
    Angle from atan2(||u||, <x,y>) — robust for small angles (arccos near 1 is ill-conditioned)."""
    c = (x * y).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    u = y - c * x                                   # tangent component, ||u|| = sin(theta)
    un = u.norm(dim=-1, keepdim=True)
    theta = torch.atan2(un, c)                      # in [0, pi]
    small = theta < 1e-6
    factor = torch.where(small, 1 + theta * theta / 6.0, theta / un.clamp_min(1e-12))
    return factor * u


def dist(x, y):
    """Geodesic distance = atan2(||y - <x,y>x||, <x,y>) (returns (...,))."""
    c = (x * y).sum(-1)
    un = (y - c.unsqueeze(-1) * x).norm(dim=-1)
    return torch.atan2(un, c)


def geodesic(x0, x1, t):
    """slerp: gamma(t) = sin((1-t)θ)/sinθ x0 + sin(tθ)/sinθ x1 (the paper's geodesic). t scalar or (...,1)."""
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=x0.device, dtype=x0.dtype)
    t = t if t.shape == () else t                       # broadcast (...,1)
    c = (x0 * x1).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS)
    theta = torch.arccos(c)
    s = torch.sin(theta)
    small = theta < 1e-4
    s0 = torch.where(small, 1 - t, torch.sin((1 - t) * theta) / s.clamp_min(1e-9))
    s1 = torch.where(small, t, torch.sin(t * theta) / s.clamp_min(1e-9))
    return normalize(s0 * x0 + s1 * x1)


def wrapped_gaussian(shape, sigma, device=None, dtype=torch.float32):
    """Mean-centered wrapped Gaussian at the identity quaternion [1,0,0,0]: sample tangent ~ N(0,sigma)
    in the 3 tangent dims and Exp it onto S^3. shape = leading dims (returns (*shape,4))."""
    mu = torch.zeros(*shape, 4, device=device, dtype=dtype)
    mu[..., 0] = 1.0
    g = torch.zeros(*shape, 4, device=device, dtype=dtype)
    g[..., 1:] = torch.randn(*shape, 3, device=device, dtype=dtype) * sigma
    return exp(mu, g)


# ---- quaternion <-> rotation matrix (data build + eval conversion; no grad needed there) ----
def quat_to_matrix(q):
    """(...,4) [w,x,y,z] unit quaternion -> (...,3,3) rotation matrix."""
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1)
    r1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1)
    r2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([r0, r1, r2], -2)


def matrix_to_quat(R):
    """(...,3,3) rotation matrix -> (...,4) [w,x,y,z] unit quaternion on the upper hemisphere."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = torch.sqrt(torch.clamp(1 + t, min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=0)) / 2
    x = torch.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = torch.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = torch.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    return canonical(normalize(torch.stack([w, x, y, z], -1)))


def _selftest():
    torch.manual_seed(0)
    dt = torch.float64
    # random unit quaternions, and tangent vectors with angle < pi/2 (valid-rotation regime)
    x = canonical(normalize(torch.randn(5000, 4, dtype=dt)))
    y = canonical(normalize(torch.randn(5000, 4, dtype=dt)))
    v = proj(x, torch.randn(5000, 4, dtype=dt))
    v = v / v.norm(dim=-1, keepdim=True) * torch.empty(5000, 1, dtype=dt).uniform_(0, 1.4)
    R = quat_to_matrix(x)
    checks = {
        "exp on manifold (|q|=1)":   (exp(x, v).norm(dim=-1) - 1).abs().max().item(),
        "log(x,exp(x,v))==v":        (log(x, exp(x, v)) - v).abs().max().item(),
        "exp(x,log(x,y))==y":        (exp(x, log(x, y)) - y).abs().max().item(),
        "||log||==dist":             (log(x, y).norm(dim=-1) - dist(x, y)).abs().max().item(),
        "tangent orthogonal":        (x * log(x, y)).sum(-1).abs().max().item(),
        "geodesic(0)==x0":           (geodesic(x, y, 0.0) - x).abs().max().item(),
        "geodesic(1)==x1":           (geodesic(x, y, 1.0) - y).abs().max().item(),
        "geodesic mid on manifold":  (geodesic(x, y, 0.5).norm(dim=-1) - 1).abs().max().item(),
        "quat->mat->quat":           (matrix_to_quat(R) - x).abs().max().item(),
        "mat->quat->mat":            (quat_to_matrix(matrix_to_quat(R)) - R).abs().max().item(),
    }
    ok = True
    print("S^3 self-tests:")
    for k, e in checks.items():
        p = e < 1e-5; ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}]  {k:26s} max_err={e:.2e}")
    return ok


if __name__ == "__main__":
    if not _selftest():
        raise SystemExit(1)
