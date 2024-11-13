import torch
import numpy as np

def normalized_distribution(freq_tensor, alpha=1e-10):
    smoothed = freq_tensor + alpha
    return smoothed / torch.sum(smoothed)

# 创建一个示例频数张量
shape = (4, 150000)  # 假设是一个4x5的张量
freq_tensor = torch.zeros(shape)

# 设置一些频数值，模拟稀疏情况
freq_tensor[0, 1] = 100  # 高频
freq_tensor[1, 2] = 50   # 中频
freq_tensor[2, 3] = 10   # 低频
freq_tensor[3, 4] = 5    # 低频

print("原始频数张量:")
print(freq_tensor)
print("\n频数和:", freq_tensor.sum().item())

# 转换为概率分布
prob_dist = normalized_distribution(freq_tensor, alpha=1e-20)
from pdb import set_trace
set_trace()
print("\n转换后的概率分布:")
print(prob_dist)
print("\n概率和(应接近1):", prob_dist.sum().item())

# 打印一些主要位置的概率值
print("\n主要位置的概率值:")
print(f"高频(0,1)位置: {prob_dist[0,1]:.6f}")
print(f"中频(1,2)位置: {prob_dist[1,2]:.6f}")
print(f"低频(2,3)位置: {prob_dist[2,3]:.6f}")
print(f"零频(0,0)位置: {prob_dist[0,0]:.10f}")