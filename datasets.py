from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, limit=None):
        self.image_paths = glob(os.path.join(root, '*'))
        self.transforms = transforms
        if limit:
            self.image_paths = self.image_paths[:limit]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image
