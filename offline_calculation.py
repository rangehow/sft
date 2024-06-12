
import torch



knns=torch.tensor([1,2,3]+[0]*25555,dtype=float)
print(torch.nn.functional.softmax(knns*2 , dim=-1))

