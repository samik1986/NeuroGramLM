import torch
import torch.nn as nn
emb = nn.Embedding(256, 256).cuda()
idx = torch.tensor([[-100, 10, 20]]).cuda()
try:
    print(emb(idx).shape)
except Exception as e:
    print(e)
