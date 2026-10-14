def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    import torch
    logits = logits.clone()
    if top_k and top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone(); remove[0] = False
        logits = logits.scatter(0, sorted_idx[remove], float('-inf'))
    return logits
