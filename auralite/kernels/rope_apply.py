def rotate_half(x):
    return __import__('torch').cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

def rope_apply(x, cos, sin):
    return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)
