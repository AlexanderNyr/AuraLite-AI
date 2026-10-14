def swiglu(gate, up):
    import torch.nn.functional as F
    return F.silu(gate) * up
