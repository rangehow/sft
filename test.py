# import torch

# c=0.1
# a=torch.tensor([0,2,3,0,1])
# nonzero_indices = torch.nonzero(a).squeeze()
# a=(1-c)*a/sum(a)
# a[a == 0] = c / (len(a) - len(nonzero_indices))

import torch

c = 0.1
a = torch.tensor([[0, 2, 3, 0, 1], [1, 2, 3, 0, 1]], dtype=torch.float32)

nonzero_count = torch.sum(a != 0,keepdim=True,dim=-1)

print(nonzero_count)
# 计算每个非零元素应分配的概率
zero_prob = c/ nonzero_count

# 分配概率给非零元素
result = torch.where(a==0, zero_prob , a)
import pdb
pdb.set_trace()
# 计算零元素应分配的概率
zero_prob = c / (a.numel() - nonzero_count)

# 分配概率给零元素
a[a == 0] = zero_prob

print(a)
