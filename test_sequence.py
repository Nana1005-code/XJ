import torch
from aligner import SequenceAligner

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 初始化对齐模块
top_k = 3
aligner = SequenceAligner(top_k=top_k)

# 构造模拟输入：假设每个 token 的维度是 64
visual_tokens_dim = 64
text_tokens_dim = 64

# 假设有 10 个视觉 token，每个 token 维度为 64
V = torch.randn(10, visual_tokens_dim)  # [n_v, d_v]

# 假设有 5 个文本 token，每个 token 维度为 64
X = torch.randn(5, text_tokens_dim)     # [n_t, d_t]

# 执行前向传播
aligned_input = aligner(V, X)

# 检查输出形状是否正确
expected_output_dim = text_tokens_dim
expected_output_length = (V.size(0) + X.size(0))  # 即 10 + 5 = 15
assert aligned_input.shape == (expected_output_length, expected_output_dim), \
    f"Output shape mismatch: expected {(expected_output_length, expected_output_dim)}, got {aligned_input.shape}"

print("✅ 测试通过！")
print(f"输出形状: {aligned_input.shape}")