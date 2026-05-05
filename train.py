from PokemonDataset import PokemonDataset
from Schedule import calc_linear_betas_and_alphas, forward_noise
from torch.utils.data import DataLoader
from UNet import UNet
import torch

def checkpoint_load(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


if __name__ == "__main__":
    #setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PokemonDataset('Data/pokemon_data/content/pokemon_images/', 'Data/pokemons2.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    betas, alphas = calc_linear_betas_and_alphas()
    model = UNet(base_channels=64).to(device)
    betas, alphas = betas.to(device), alphas.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Using device: {device}")
    print("Setup Complete")
    
    # training loop
    for epoch in range(500):
        print(f"Starting epoch {epoch+1}...")
        for batch in dataloader:
            images, conditions = batch
            images, conditions = images.to(device), {k: v.to(device) for k, v in conditions.items()}
            cond = {k: v.squeeze(1) for k, v in conditions.items()}
            optimizer.zero_grad()
            steps = torch.randint(0, 1000, (images.size(0),)).to(device)
            noisy_images, noise = forward_noise(images, steps, alphas)
            prediction = model(noisy_images, steps, cond)
            loss = torch.nn.functional.mse_loss(prediction, noise)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        if (epoch + 1) % 50 == 0:  # save every 50 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f'checkpoints/unet_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved at epoch {epoch+1}")
            
            


