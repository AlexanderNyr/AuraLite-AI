def rms_norm(x, weight, eps=1e-6):
    import torch
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * weight
