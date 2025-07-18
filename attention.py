import torch

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps=1e-6):
    p = p + eps
    q = q + eps
    kl = p * (p / q).log()
    return kl.sum(dim=-1).mean()

def compute_attention_alignment_loss(attn_map: torch.Tensor, n_text: int):
    attn_avg = attn_map.mean(dim=1)  # [B, L, L]
    attn_avg = torch.softmax(attn_avg, dim=-1)  # 确保是概率分布

    P = attn_avg[:, :n_text, :]
    Q = attn_avg[:, n_text:, :]

    loss_tokenwise = 0.5 * (kl_divergence(P, Q) + kl_divergence(Q, P))
    loss_totalwise = 0.5 * (kl_divergence(P.mean(dim=1), Q.mean(dim=1)) +
                            kl_divergence(Q.mean(dim=1), P.mean(dim=1)))

    return 0.5 * (loss_tokenwise + loss_totalwise)
