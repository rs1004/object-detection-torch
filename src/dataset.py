from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

LABEL_MAP = {
}


class MyDataset(Dataset):
    def __init__(self, data_dir, data_list_file_name, imsize=448, grid_cell=7, transform=None):
        self.transform = transform
        self.data_paths = self._get_paths(data_dir, data_list_file_name)
        self.imsize = imsize
        self.grid_cell = grid_cell

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, label_path = self.data_paths[idx]

        image = Image.open(image_path).convert('RGB').resize((self.imsize, self.imsize))
        label = self._get_label(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_paths(self, data_dir, data_list_file_name):
        data_list_path = Path(data_dir) / 'ImageSets' / 'Main' / data_list_file_name

        with open(data_list_path, 'r') as f:
            ids = f.read().split('\n')

        return [[Path(data_dir) / 'JPEGImages' / f'{i}.jpg', Path(data_dir) / 'Annotations' / f'{i}.xml'] for i in ids[:-1]]

    def _get_label(label_path):
        root = ET.parse(label_path).getroot()
        org_shape = [int(s.text) for s in root.find('size')]
        org_bbox = [int(b.text) for b in obj.find('bndbox')]
        
