import torch

# 假设你的张量名为tensor，索引范围名为indices
model_logits = torch.stack((torch.arange(120),torch.arange(120))).unsqueeze(-1)

print(model_logits.shape)
valid_label_index_list=[[[57, 120]], [[38, 68]]]

print(model_logits[0][120])

# 使用索引操作取出第二维的元素
result=torch.cat([row[start:end+1] for row, turn in zip(model_logits, valid_label_index_list) for start, end in turn])
print(result,result.shape)
