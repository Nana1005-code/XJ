import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAligner(nn.Module):
    def __init__(self, topk: int = 100):
        super().__init__()
        self.topk = topk
        torch.set_printoptions(threshold=float('inf'))
    #得到余弦相似度矩阵，topk清空，再进行softmax
    @staticmethod
    def cosine_similarity_batch(query: torch.Tensor, keys: torch.Tensor,topk:int,eps: float = 1e-8) -> torch.Tensor:
        query_norm = query / (query.norm(p=2) + eps)          # [d]
        keys_norm = keys / (keys.norm(p=2, dim=1, keepdim=True) + eps)  # [n, d]
        
        sims = torch.matmul(keys_norm, query_norm)            # [n]
        # 获取 top-k 的索引和值
        topk_vals, topk_idx = torch.topk(sims, topk)

        # 初始化为 0
        sims_softmax_topk = torch.zeros_like(sims)

        # 对 top-k 的值进行 softmax，然后赋值
        softmax_weights = F.softmax(topk_vals, dim=0)
        sims_softmax_topk[topk_idx] = softmax_weights

        return sims_softmax_topk  # [n]
    #前向传播
    def forward(self, V: torch.Tensor, X: torch.Tensor,is_image_token: torch.Tensor) -> torch.Tensor:
        assert is_image_token.shape[0] == X.shape[0], "Mismatch in token length"
        #根据余弦相似度选出topk个视觉token
        fused_visual_tokens = []
        
        text_token_nums = X.shape[0]  

        # 计算所有文本token与视觉token的余弦相似度 [n_text, n_visual]
        X_norm = X / (X.norm(p=2, dim=1, keepdim=True) + 1e-8)
        V_norm = V / (V.norm(p=2, dim=1, keepdim=True) + 1e-8)
        

        # 找到所有非-image位置的索引
        non_image_indices = (~is_image_token).nonzero(as_tuple=True)[0]
        # 提取非-image文本token
        non_image_indices = non_image_indices.to(X_norm.device)
        X_non_image = X_norm[non_image_indices]  # [n_nonimage, d]

        sims = torch.matmul(X_non_image, V_norm.t())  # [n_text, n_visual]
        
        # 对每个文本token，选topk视觉token并softmax
        topk_vals, topk_idx = torch.topk(sims, self.topk, dim=1)
        sims_softmax_topk = torch.zeros_like(sims)
        softmax_weights = F.softmax(topk_vals, dim=1)
        sims_softmax_topk.scatter_(1, topk_idx, softmax_weights)

        # 融合视觉token [n_text, n_visual] x [n_visual, d] -> [n_text, d]
        fused_visual_tokens = torch.matmul(sims_softmax_topk, V)
        
        print(fused_visual_tokens.shape)
        
        #拼接序列 [V', X]
        aligned_input = torch.cat([fused_visual_tokens, X], dim=0)  # [2 * n2, d2]
        print(aligned_input.shape)
        
        return aligned_input