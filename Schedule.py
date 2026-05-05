import torch
from PIL import Image
import matplotlib.pyplot as plt
from PokemonDataset import PokemonDataset

def linear_schedule(steps=1000, start=1e-4, end=0.02):
    return torch.linspace(start, end, steps)

def calc_alphas(betas):
    alphas = 1 - betas
    alphas_cumulative = torch.cumprod(alphas, dim=0)
    return alphas_cumulative

def calc_linear_betas_and_alphas():
    betas = linear_schedule()
    alphas = calc_alphas(betas)
    return betas, alphas

def forward_noise(image, step, alphas):
    # should handle batches
    return torch.sqrt(alphas[step].view(-1, 1, 1, 1)) * image + torch.sqrt(1 - alphas[step].view(-1, 1, 1, 1)) * torch.randn_like(image)


if __name__ == "__main__":
    # linear beta and alpha calc
    betas, alphas = calc_linear_betas_and_alphas()
    print(f"Betas shape: {betas.shape}")
    print(f"Betas range: {betas[0]:.4f} to {betas[-1]:.4f}")
    print(f"Alphas shape: {alphas.shape}")
    print(f"Alphas range: {alphas[0]:.4f} to {alphas[-1]:.4f}")

    # linear noise visualized
    dataset = PokemonDataset('Data/pokemon_data/content/pokemon_images/', 'Data/pokemons2.csv')
    img = PokemonDataset.__getitem__(dataset, 0).unsqueeze(0)

    showSteps = [0, 50, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
    
    images = []

    
    for step in range(len(showSteps)):
        noisy = forward_noise(img, torch.tensor([showSteps[step]]), alphas).squeeze(0).permute(1, 2, 0)
        between_0_and_1 = (noisy.clamp(-1, 1) + 1) / 2
        plt.subplot(2, 6, step + 1)
        plt.title(f"Step {showSteps[step]}")
        plt.imshow(between_0_and_1.detach().numpy())
        plt.axis('off')
        
    plt.savefig('outputs/linear_noise_test.png')
    plt.show()
    
