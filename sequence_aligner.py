import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAligner(nn.Module):
    def __init__(self,visual_token_num:int, visual_dim: int,text_token_num:int, text_dim: int, top_k: int = 3):
        """
        Args:
            visual_token_num(int):视觉token数量
            visual_dim (int): 视觉嵌入维度 d1
            text_token_num(int):文本token数量
            text_dim (int): 文本嵌入维度 d2
            top_k (int): 每个文本 token 对应选择 top-k 个相似视觉 token
        """
        super().__init__()
        self.visual_to_text_linear = nn.Linear(visual_dim, text_dim)  #(n1,d1) -> (n1,d2)
        self.top_k = top_k
        self.fusion_linear = nn.Linear(top_k, 1)
    
    #余弦相似度
    @staticmethod
    def cosine_similarity_batch(query: torch.Tensor, keys: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        query_norm = query / (query.norm(p=2) + eps)          # [d]
        keys_norm = keys / (keys.norm(p=2, dim=1, keepdim=True) + eps)  # [n, d]
        sims = torch.matmul(keys_norm, query_norm)            # [n]
        return sims #获得query与每个key的相似度

    #Topk个视觉token融合 【【【【【【【【【【【【【【【【【【【【【【【【【【重写】】】】】】】】】】】】】】】】】】】】】】】】
    def vision_kFusion(self,need_tobefused):
        return self.fusion_linear(need_tobefused) #先写一个通过linear融合的

    #前向传播
    def forward(self, V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V: [n1, d1]，视觉 token 序列
            X: [n2, d2]，文本 token 序列
        Returns:
            拼接序列: [2*n2, d2]
        """
        n1, d1 = V.size()
        n2, d2 = X.size()
        V_proj = self.visual_to_text_linear(V)  # [n1, d2]
        
        #根据余弦相似度选出topk个视觉token
        fused_visual_tokens = []
        
        for i in range(n2):
            sims = self.cosine_similarity_batch(X[i], V_proj)  
            topk_idx = torch.topk(sims, 3).indices #得到最相似的向量的索引
            need_tobefused = V_proj[topk_idx]
            fused_visual_tokens.append(self.vision_kFusion(need_tobefused))
        
        V_aligned = torch.stack(fused_visual_tokens, dim=0)  # [n2, d2]

        #拼接序列 [V', X]
        sequence = torch.cat([V_aligned, X], dim=0)  # [2 * n2, d2]
        return sequence