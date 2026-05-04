import torch

def linear_schedule(steps=1000, start=1e-4, end=0.02):
    return torch.linspace(start, end, steps)

def alphas(betas):
    alphas = 1 - betas
    alphas_cumulative = torch.cumprod(alphas, dim=0)
    return alphas_cumulative

def calc_betas_and_alphas():
    betas = linear_schedule()
    alphas = alphas(betas)
    return betas, alphas

