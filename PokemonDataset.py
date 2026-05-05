import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch
from UNet import TYPE_TO_IDX, RANK_TO_IDX, GENERATION_TO_IDX

class PokemonDataset(Dataset):
    def __init__(self, img_dir, csv_dir, image_size=256):
        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.df = pd.read_csv(csv_dir).set_index("index")
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
        image = Image.open(img_path).convert('RGBA')
        # add white background instead of transparent
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background.convert('RGB')
        image = self.transform(image)

        csv_row = self.df.loc[idx + 1]

        conditions = {
            "type1": torch.tensor([TYPE_TO_IDX[csv_row['type1']]]),
            "type2": torch.tensor([TYPE_TO_IDX[csv_row['type2'] if pd.notna(csv_row['type2']) else 'none']]),
            "rank": torch.tensor([RANK_TO_IDX[csv_row['rank']]]),
            "generation": torch.tensor([GENERATION_TO_IDX[csv_row['generation']]])
        }

        return image, conditions


if __name__ == "__main__":
    # for testing
    dataset = PokemonDataset('Data/pokemon_data/content/pokemon_images/', 'Data/pokemons2.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    batch = next(iter(dataloader))
    print(batch[0].shape)   # should be [32, 3, 256, 256]
    print(batch[0].min(), batch[0].max())  # should be close to -1 and 1
    for key, val in batch[1].items():
        print(f"{key}: {val.squeeze().tolist()}")

