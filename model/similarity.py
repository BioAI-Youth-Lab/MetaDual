import torch

def consinesimilar(a, b):
    a_norm = torch.norm(a, dim=1)
    a_size = a_norm.size(0)
    a_norm = a_norm.reshape(a_size, 1)
    b_norm = torch.norm(b, dim=1)
    b_size = b.size(0)
    b_norm = b_norm.reshape(1, b_size)
    ab = torch.mm(a, b.t())
    ab_norm = torch.mm(a_norm, b_norm)
    c = torch.div(ab, ab_norm + 1e-08)
    return c
