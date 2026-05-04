import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class PokemonDataset(Dataset):
    def __init__(self, img_dir, image_size=256):
        self.img_dir = img_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # [0, 255] → [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [0, 1] → [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)
    

if __name__ == "__main__":
    dataset = PokemonDataset('Data/pokemon_data/content/pokemon_images/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    batch = next(iter(dataloader))
    print(batch.shape)   # should be [32, 3, 256, 256]
    print(batch.min(), batch.max())  # should be close to -1 and 1

