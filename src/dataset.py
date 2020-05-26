from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import torch

LABEL_MAP = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


class MyDataset(Dataset):
    def __init__(self, data_dir, data_list_file_name, imsize=448, grid_num=7, bbox_num=2, l_coord=5., l_noobj=0.5, transform=None):
        self.transform = transform
        self.data_paths = self._get_paths(data_dir, data_list_file_name)
        self.imsize = imsize
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.l_coord = l_coord
        self.l_noobj = l_noobj

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

    def _get_label(self, label_path):
        grid_len = self.imsize // self.grid_num

        root = ET.parse(label_path).getroot()
        width, height, _ = [int(s.text) for s in root.find('size')]
        label = torch.zeros((self.grid_num, self.grid_num, 5 * self.bbox_num + len(LABEL_MAP)))
        mask = torch.zeros_like(label)
        for obj in root.iter('object'):
            xmin, ymin, xmax, ymax = [int(b.text) for b in obj.find('bndbox')]
            xmin, xmax = self.imsize * xmin / width, self.imsize * xmax / width
            ymin, ymax = self.imsize * ymin / height, self.imsize * ymax / height
            c = LABEL_MAP[obj.find('name').text]
            cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
            lx, ly = xmax - xmin, ymax - ymin
            idx, idy = int(cx // grid_len), int(cy // grid_len)
            for j in range(self.bbox_num):
                if label[idx, idy, 5*j] == 0.:
                    label[idx, idy, 0+5*j:2+5*j] = (torch.tensor([cx, cy]) % grid_len) / grid_len
                    label[idx, idy, 2+5*j:4+5*j] = torch.sqrt((torch.tensor([lx, ly]) / self.imsize))
                    label[idx, idy, 4+5*j] = 1.
                    mask[idx, idy, 0+5*j:4+5*j] = self.l_coord
                    mask[idx, idy, 4+5*j] = 1
                    break
            label[idx, idy, 10+c] = 1.
            mask[idx, idy, 10:] = 1.
        for j in range(self.bbox_num):
            mask[:, :, 4+5*j] += (1 - mask[:, :, 4+5*j]) * self.l_noobj

        return torch.stack([label, mask])
