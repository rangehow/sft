import torch

# 示例数据
a = 3
b = 4
weight = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)  # 假设 a = 3
data = torch.tensor([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]], dtype=torch.float32)  # 假设 a = 3, b = 4

print()
# 步骤1：归一化权重
normalized_weight = weight@torch.mean(data,dim=0)

# 步骤2：加权平均
weighted_average = torch.matmul(normalized_weight, data)

print(weighted_average)