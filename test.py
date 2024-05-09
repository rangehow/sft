
from collections import Counter, defaultdict
import json
import math
from loguru import logger
import torch
query_length=key_length=7
from rich import print
context_position = torch.arange(query_length, dtype=torch.long,)[:, None]
memory_position = torch.arange(key_length, dtype=torch.long, )[None, :]
relative_position = memory_position - context_position  # shape (query_length, key_length)

nb=3
num_buckets=nb+1
max_distance=5

bidirectional=False

print(f'nb={nb},bucket={num_buckets},max_distance={max_distance},seq_len={query_length},llm_attention={not bidirectional}')

def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # print(relative_position)
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            print((relative_position > 0).to(torch.long) * num_buckets)
            relative_position = torch.abs(relative_position)
            print(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    


bbb=_relative_position_bucket(relative_position=relative_position,bidirectional=bidirectional,num_buckets=num_buckets,max_distance=max_distance)
# print(relative_attention_bias.weight)

print(bbb)
aaa=bbb[-1].tolist()[::-1]
cnt={}
for i,num in enumerate(aaa):
    if num not in cnt:
        cnt[num]=i

for i in range(nb):
    if cnt[i+1]-cnt[i]!=1:
        cnt[i]=f"{cnt[i]}-{cnt[i+1]-1}"
print(cnt)

from torch import nn
relative_attention_bias = nn.Embedding(num_buckets, 1)
# print(relative_attention_bias(bbb))

